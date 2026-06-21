# Codebase Overview

## Structure

```
src/
  data/
    __init__.py              — exports DataSource, TimeSeriesDataPreprocessor, netcdf_to_zarr
    datasource.py            — DataSource dataclass
    zarr_store.py            — ZarrStoreManager (store cache, coord→index, patch reads)
    timeseries_dataset.py    — TimeSeriesDataset + _collate_fn
    normalization.py         — compute_norm_stats, _read_1d_df helper
    utils.py                 — netcdf_to_zarr conversion helper
    preprocessing.py         — TimeSeriesDataPreprocessor (legacy in-memory 1D path)
  forecaster/
    forecaster.py            — training loop, normalization, logging, checkpointing
  models/
    LSTM.py                  — LSTMHistoric and LSTMEncoderDecoder
  utils/
    scores_and_losses.py     — metric classes + resolve_metric
    DILATE/
      soft_dtw.py            — pure-PyTorch soft-DTW forward/backward DP
      dilate_loss.py         — DILATE loss combining shape + temporal terms
      __init__.py
```

---

## `src/data/datasource.py` — `DataSource`

Plain dataclass describing one dataset (one location, one time window, one or two file sources).

| Field | Type | Description |
|---|---|---|
| `start` | `str` | Start date, e.g. `'2000-01-01'` |
| `end` | `str` | End date, e.g. `'2014-12-31'` |
| `csv` | `str \| None` | Path to CSV file (1D tabular) |
| `netcdf_1d` | `str \| None` | Path to NetCDF file for 1D variables |
| `netcdf_1d_vars` | `list` | Variable names to read from `netcdf_1d` (required if `netcdf_1d` set) |
| `zarr` | `str \| None` | Path to Zarr store (2D grid) |
| `location` | `tuple \| None` | `(ix, iy)` pixel index OR `(cx, cy)` coordinate value (required if `zarr` set) |
| `location_is_index` | `bool` | `True` = pixel index, `False` = coordinate value |
| `coord_dims` | `tuple` | Spatial dimension names in the Zarr store, e.g. `('x', 'y')` |
| `target_col` | `str` | Column name for `y` in CSV/NetCDF 1D |
| `static_zarr_vars` | `list` | Zarr vars with no time dimension |
| `csv_index_col` | `int` | Column index to use as DataFrame index when reading CSV (default `0`) |

Validation in `__post_init__`: at least one of `csv`/`netcdf_1d`/`zarr` required; `location` required when `zarr` set; `netcdf_1d_vars` required when `netcdf_1d` set.

---

## `src/data/zarr_store.py` — `ZarrStoreManager`

Class-level store cache: multiple `DataSource` objects pointing to the same path share one open `xr.Dataset`.

**`open(path)`** — returns cached or newly opened `xr.Dataset` via `xr.open_zarr`.

**`coord_to_index(ds, location, coord_dims, location_is_index)`** — converts a coordinate pair to integer pixel indices using nearest-neighbour lookup on the dimension arrays.

**`read_patch(ds, vars, time_indices, ix, iy, patch_size, coord_dims)`** — returns `{var: array(time, H, W)}` for the given dynamic variables, integer time indices, and spatial centre `(ix, iy)`.

**`read_static_patch(ds, vars, ix, iy, patch_size, coord_dims)`** — returns `{var: array(H, W)}` for variables with no time dimension.

---

## `src/data/normalization.py`

**`_read_1d_df(source)`** — loads a CSV and/or NetCDF 1D file for a `DataSource`, slices by `start`/`end`, and returns a merged `DataFrame` (or `None` if no 1D source).

**`compute_norm_stats(sources, seq_len, horizon, historic_cols, future_cols, patch_size)`** — computes per-variable mean/std over all sources (train + val combined).

Returns a dict with keys:
- `xh_mean`, `xh_std` — shape `(n_historic_features,)`; present if any source has 1D data with `historic_cols`
- `xf_mean`, `xf_std` — shape `(n_future_features,)`
- `target_mean`, `target_std` — scalars
- `grid_mean`, `grid_std` — shape `(n_grid_channels,)`; present if any source has `zarr`
- `zarr_vars` — list of dynamic variable names (channel order for grid tensors)

---

## `src/data/timeseries_dataset.py` — `TimeSeriesDataset`

Lazy `torch.utils.data.Dataset`. Pre-loads 1D data into memory; reads Zarr patches on demand.

**Constructor** — `sources, seq_len, horizon, historic_cols, future_cols, patch_size, norm_stats, dtype`

At construction:
1. Loads 1D DataFrames into memory per source.
2. Computes per-timestep zarr NaN validity masks by reading the full patch time series once.
3. Builds a flat index `[(source_idx, t)]` of all valid windows, skipping any window with NaN in `x_h`/`x_f`/`x_h_grid`/`x_f_grid`.

**`__getitem__(i)`** — returns a `dict` containing only the keys present for the given source:

| Key | Shape | Condition |
|---|---|---|
| `x_h` | `(seq_len, n_hist)` | source has CSV or NetCDF 1D |
| `x_f` | `(horizon, n_fut)` | source has CSV or NetCDF 1D |
| `y` | `(horizon, 1)` | source has CSV or NetCDF 1D |
| `x_h_grid` | `(seq_len, C, H, W)` | source has zarr |
| `x_f_grid` | `(horizon, C, H, W)` | source has zarr |
| `x_static_grid` | `(C_static, H, W)` | source has zarr + `static_zarr_vars` |

All values are normalized using `norm_stats`. `y` NaN values are preserved for loss masking.

**`_collate_fn(batch)`** — custom collate that stacks tensors for the intersection of keys across all items in the batch. Import from `src.data.timeseries_dataset`.

---

## `src/data/utils.py` — `netcdf_to_zarr`

```python
def netcdf_to_zarr(input_path: str, output_path: str, chunk_sizes: dict):
```

Converts a NetCDF file to a Zarr store. `chunk_sizes` controls rechunking, e.g. `{'time': 1, 'x': 256, 'y': 256}`.

---

## `src/data/preprocessing.py` — `TimeSeriesDataPreprocessor` (legacy)

Original in-memory preprocessor for 1D CSV data. Kept for internal reference; new code should use `TimeSeriesDataset`.

---

## `src/forecaster/forecaster.py` — `Forecaster`

Training orchestrator. Takes `DataSource` lists, handles normalization, training, validation, checkpointing, and inference.

**Key constructor parameters (changed from prior version):**

| Parameter | Description |
|---|---|
| `training_datasets` | `list[DataSource]` — all combined into one `TimeSeriesDataset` |
| `validation_datasets` | `list[DataSource]` — each becomes a separate `TimeSeriesDataset` / DataLoader |
| `patch_size` | Spatial patch size for Zarr data (default `64`) |
| `num_workers` | DataLoader worker processes (default `0`) |
| `prefetch_factor` | Prefetch factor; applied only when `num_workers > 0` (default `2`) |
| `pin_memory` | DataLoader pin memory (default `False`) |
| `drop_last` | Drop last incomplete batch (default `False`) |
| `shuffle` | Shuffle training loader; validation always `False` (default `True`) |

Removed: `dfs_training_sets`, `dfs_validation_sets`, `target_col` (now per-`DataSource`).

**Init flow:**
1. `compute_norm_stats(training + validation sources)` → `norm_stats`
2. `TimeSeriesDataset(training_datasets, ..., norm_stats)` → training dataset
3. One `TimeSeriesDataset([src], ..., norm_stats)` per validation source → validation datasets
4. Wrap each in `DataLoader` with supplied settings

**Training / eval loops:** batches are now dicts. `y = batch['y']`; `X = {k: v for k, v in batch.items() if k != 'y'}`. NaN filtering on `y` unchanged. The model receives the full `X` dict — existing models ignore unknown keys; future spatial models will use `x_h_grid`, `x_f_grid`, `x_static_grid`.

**`predict`:** accepts either a legacy `TensorDataset` (yields tuples) or a `TimeSeriesDataset` (yields dicts). Duck-typed: checks `isinstance(batch, dict)`.

---

## `src/models/LSTM.py`

Both models receive a single `config` dict. The Forecaster injects `historic_cols`, `future_cols`, `forecasting_horizon`, and `historic_input_sequence_length` into this dict automatically.

`X` is an **open dict** — models read only the keys they need and silently ignore the rest. Existing models use `X['x_h']` and `X['x_f']`. Future spatial models will add `X['x_h_grid']`, `X['x_f_grid']`, `X['x_static_grid']`.

**`LSTMHistoric`** — encodes only `x_h`; single LSTM → last hidden → linear → `(B, horizon, 1)`.

**`LSTMEncoderDecoder`** — encoder LSTM on `x_h`; hidden downscaled and repeated over horizon, concatenated with `x_f`; decoder LSTM → linear → `(B, horizon, 1)`.

---

## `src/utils/scores_and_losses.py`

**`resolve_metric(metric)`** — string or instance → metric object. String forms: `'mae'` or `'DILATE(alpha=0.5, gamma=0.01)'`.

**`assert_differentiable(metric)`** — raises `TypeError` if metric has `differentiable = False`.

**Metric classes:** `MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `DILATE` — all wrap `nn.Module`. `DILATE` has `eval_on_normalized = True` (soft-DTW overflows on large denormalized values).

---

## `check.ipynb`

Reference notebook demonstrating a full training run using the new `DataSource` API:
1. Create `DataSource` descriptors for train/val/test time windows pointing to `data/data.csv`
2. Inspect `TimeSeriesDataset` output shapes
3. Train `LSTMHistoric` (16 hidden, 1 layer) for 5 epochs with MAE loss, RMSE val score
4. Train `LSTMEncoderDecoder` (16 hidden, 1 layer, downscale 8) same settings
