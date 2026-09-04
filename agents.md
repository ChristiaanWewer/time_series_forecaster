# Codebase Overview

## Structure

```
src/
  data/
    __init__.py              — exports DataSource, TimeSeriesDataPreprocessor
    datasource.py            — DataSource dataclass
    timeseries_dataset.py    — TimeSeriesDataset + _collate_fn
    normalization.py         — compute_norm_stats, _read_1d_df helper
    preprocessing.py         — TimeSeriesDataPreprocessor (legacy in-memory 1D path)
  forecaster/
    forecaster.py            — training loop, normalization, logging, checkpointing
  models/
    lstm_historic.py         — LSTMHistoric
    lstm_encoder_decoder.py  — LSTMEncoderDecoder
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
| `start` | `str \| None` | Start date, e.g. `'2000-01-01'`; `None` = no lower bound (whole file) |
| `end` | `str \| None` | End date, e.g. `'2014-12-31'`; `None` = no upper bound (whole file) |
| `csv` | `str \| None` | Path to CSV file (1D tabular) |
| `netcdf_1d` | `str \| None` | Path to NetCDF file for 1D variables |
| `netcdf_1d_vars` | `list` | Variable names to read from `netcdf_1d` (required if `netcdf_1d` set) |
| `csv_index_col` | `int` | Column index to use as DataFrame index when reading CSV (default `0`) |
| `binary_cols` | `list` | Columns to skip normalization for; empty = auto-detect from values |
| `nodata_values` | `list` | Sentinel values (e.g. `-999`) treated as missing, across all columns — converted to `NaN` in `_read_1d_df` |

Validation in `__post_init__`: at least one of `csv`/`netcdf_1d` required; `netcdf_1d_vars` required when `netcdf_1d` set.

No `target_col` field — that's owned by `Forecaster` now (see below), not per-source. For a netcdf source, note that `netcdf_1d_vars` must include the target variable's name too if it's meant to be read from that file for training — `_read_1d_df` subsets the netcdf dataset down to exactly `netcdf_1d_vars` before the target lookup ever happens, unlike the CSV path which reads every column unconditionally.

---

## `src/data/normalization.py`

**`_read_1d_df(source)`** — loads a CSV and/or NetCDF 1D file for a `DataSource`, slices by `start`/`end` (skipped when either is `None`, using the file's full range), and returns a merged `DataFrame` (or `None` if no 1D source). Any `source.nodata_values` are replaced with `NaN` before returning.

**`_merge_stats(n_a, mean_a, M2_a, n_b, mean_b, M2_b)`** — Chan's parallel variance algorithm. Merges two `(n, mean, M2)` accumulators into one without storing raw values. Exact — no approximation.

**`compute_norm_stats(sources, historic_cols, future_cols, target_col='y')`** — computes per-variable mean/std using Chan's algorithm: iterates one source at a time, accumulating `(n, mean, M2)` per column, never holding more than one source's data in memory. Target stats are accumulated in the same pass, using `target_col` (not read off `DataSource` — see `Forecaster` below); if `target_col` is `None` or absent from a source's columns, target stats are skipped for that source.

Called on **training sources only**. The resulting stats are then applied to both training and validation datasets.

Binary column handling: a column is skipped (mean=0, std=1) if it appears in `source.binary_cols` OR if auto-detection finds all values in `{0, 1}`. Skipped columns are recorded in `stats['binary_cols']`.

Returns a dict with keys:
- `xh_mean`, `xh_std` — shape `(n_historic_features,)`
- `xf_mean`, `xf_std` — shape `(n_future_features,)`
- `target_mean`, `target_std` — scalars
- `binary_cols` — `set[str]` of column names that were not normalized

---

## `src/data/timeseries_dataset.py` — `TimeSeriesDataset`

Lazy `torch.utils.data.Dataset`. Pre-loads 1D data into memory.

**Constructor** — `sources, seq_len, horizon, historic_cols, future_cols, norm_stats, target_col='y', dtype`

At construction:
1. Loads 1D DataFrames into memory per source.
2. Builds a flat index `[(source_idx, t)]` of all valid windows, skipping any window with NaN in `x_h`/`x_f`.

**`__getitem__(i)`** — returns a `dict`:

| Key | Shape | Condition |
|---|---|---|
| `x_h` | `(seq_len, n_hist)` | always |
| `x_f` | `(horizon, n_fut)` | always |
| `y` | `(horizon, 1)` | only if `target_col is not None` |

All values are normalized using `norm_stats`. `y` NaN values are preserved for loss masking. `target_col=None` is how `Forecaster.predict` builds its dataset — no target column needed on a file used purely for inference. `_load_1d` looks up `df[[target_col]]` with no existence guard, so a wrong or missing column name raises a plain `KeyError` rather than a custom error.

**`_collate_fn(batch)`** — custom collate that stacks tensors for the intersection of keys across all items in the batch. Import from `src.data.timeseries_dataset`.

---

## `src/data/preprocessing.py` — `TimeSeriesDataPreprocessor` (legacy)

Original in-memory preprocessor for 1D CSV data. Kept for internal reference; new code should use `TimeSeriesDataset`.

---

## `src/forecaster/forecaster.py` — `Forecaster`

Training orchestrator. Takes `DataSource` lists, handles normalization, training, validation, checkpointing, and inference.

**Key constructor parameters:**

| Parameter | Description |
|---|---|
| `training_datasets` | `list[DataSource]` — all combined into one `TimeSeriesDataset` |
| `validation_datasets` | `list[DataSource]` — each becomes a separate `TimeSeriesDataset` / DataLoader |
| `target_col` | `str \| None`, default `'y'` — column name for the target across all sources. Lives here, not on `DataSource` (one Forecaster = one target). Saved/restored in the checkpoint. |
| `num_workers` | DataLoader worker processes (default `0`) |
| `prefetch_factor` | Prefetch factor; applied only when `num_workers > 0` (default `2`) |
| `pin_memory` | DataLoader pin memory (default `False`) |
| `drop_last` | Drop last incomplete batch (default `False`) |
| `shuffle` | Shuffle training loader; validation always `False` (default `True`) |
| `norm_stats` | `dict \| None` — reuse pre-computed stats instead of calling `compute_norm_stats` (used internally by `load_model` on resume, so datasets stay consistent with what the restored model/optimizer were trained against) |

**Init flow:**
1. `compute_norm_stats(training sources only)` → `norm_stats`, unless `norm_stats` was passed in directly (train stats applied to val either way)
2. `TimeSeriesDataset(training_datasets, ..., norm_stats)` → training dataset
3. One `TimeSeriesDataset([src], ..., norm_stats)` per validation source → validation datasets
4. Wrap each in `DataLoader` with supplied settings

**Training / eval loops:** batches are dicts. `y = batch['y']`; `X = {k: v for k, v in batch.items() if k != 'y'}`. Both `X` and `y` are moved to `self.device` before the forward pass. NaN filtering on `y` unchanged. The model receives the full `X` dict — models ignore unknown keys.

**Logging:** `self.val_log` / `self.val_log_aggregated['mean'|'max'|'min'|'median']` are **structured numpy arrays** — named fields instead of positional indices, e.g. `val_log_aggregated['mean'][epoch]['val_score']`. Field names come from `_criteria_field_names`: `'loss'` and `'val_score'` are fixed (first two slots), remaining criteria are named by class (`'MAE'`, `'NSE'`, ...), deduplicated with a numeric suffix (`'MAE_2'`) if the same metric type appears more than once. `self._criteria_names` holds the field name list. `self.loss_log` (training loss only, one value per epoch) stays a plain float array.

**Checkpointing:** `fit()` saves a bundled dict (via `self._checkpoint_dict(epoch)`), not a bare state dict — `torch.save({...}, path)` with keys: `state_dict`, `optimizer_state_dict`, `epoch`, `best_logged_criterion`, `norm_stats`, `model_config`, `historic_cols`, `future_cols`, `target_col`, `forecasting_horizon`, `historic_input_sequence_length`, `loss_function_spec`, `validation_score_spec`, `validation_logging_criteria_specs`, `minimize_validation_score`, `save_aggregation_criterion`, `criteria_names`, `loss_log`, `val_log`, `val_log_aggregated`. Metrics are saved as reconstructable spec strings (`metric_to_spec`, e.g. `'DILATE(alpha=0.5, gamma=0.01)'`) rather than pickled objects — a compiled `DILATE` holds a `torch.compile` closure that isn't reliably picklable across processes. `load_weights(path=None)` reloads just `ckpt['state_dict']` into the current in-memory `Forecaster` (same process, e.g. reverting to the best epoch after training).

**`Forecaster.load_model(checkpoint_path, model, resume, training_datasets=None, validation_datasets=None, device='cpu', batch_size=512, num_workers=0, pin_memory=False, use_torch_compile=False, learning_rate=None, **forecaster_kwargs)`** — classmethod, three modes:
- `training_datasets=None` → **inference only**. Builds just the model from `ckpt['model_config']`, loads weights, carries over `norm_stats`/schema. No optimizer, loaders, or logs — built via `cls.__new__(cls)`, bypassing `__init__`, so calling `.fit()` on the result raises `AttributeError` by design.
- `resume=True` (requires `training_datasets`) → **continues the same run**. Restores weights, optimizer state, `norm_stats`, the loss/score/aggregation config, `best_logged_criterion`, and log history — all pulled from the checkpoint automatically, never re-specified, so a resumed run can't silently diverge from the original. `fc._start_epoch = checkpoint['epoch'] + 1`; raises `ValueError` if `number_of_epochs <= start_epoch`. `learning_rate`, if given, overwrites just the restored optimizer's LR (keeps momentum/Adam moment state) — e.g. for decaying LR on a fine-tuning continuation.
- `resume=False` + `training_datasets` given → **warm start**. A fresh `Forecaster` (fresh optimizer, fresh `norm_stats` computed from whatever `training_datasets` are passed now, fresh logs at epoch 0), with only `state_dict` loaded as an initialization. `model_config`/`historic_cols`/etc. must be supplied via `forecaster_kwargs` here, same as a plain `Forecaster(...)` call.

`use_torch_compile` is applied *after* weights are loaded (matching `load_weights`'s existing behavior) — compiling is a runtime choice for wherever you're loading, not a fact saved in the checkpoint.

**`predict(x_test, batch_size=None, denormalize=True, device=None, num_workers=0, pin_memory=False)`** — `x_test` is a `DataSource` or `list[DataSource]`, nothing else (mirrors exactly what `training_datasets`/`validation_datasets` accept — no `TimeSeriesDataset`/`TensorDataset` support, no type validation; anything else just crashes naturally). Builds a `TimeSeriesDataset` internally using `self.historic_cols`/`self.future_cols`/`self.historic_input_sequence_length`/`self.forecasting_horizon`/`self.norm_stats`, always with `target_col=None` — `predict` never reads `y`, so no target column is required on the source, even if it wasn't needed at training time either. `device`/`num_workers`/`pin_memory` are independent of whatever training used (train and inference are commonly on different machines); `device=None` defaults to `self.device` and moves the model there if it differs.

---

## `src/models/lstm_historic.py`, `src/models/lstm_encoder_decoder.py`

Both models receive a single `config` dict. The Forecaster injects `historic_cols`, `future_cols`, `forecasting_horizon`, and `historic_input_sequence_length` into this dict automatically.

`X` is an **open dict** — models read only the keys they need and silently ignore the rest. Existing models use `X['x_h']` and `X['x_f']`.

**`LSTMHistoric`** — encodes only `x_h`; single LSTM → last hidden → linear → `(B, horizon, 1)`.

**`LSTMEncoderDecoder`** — encoder LSTM on `x_h`; hidden downscaled and repeated over horizon, concatenated with `x_f`; decoder LSTM → linear → `(B, horizon, 1)`.

---

## `src/utils/scores_and_losses.py`

**`resolve_metric(metric)`** — string or instance → metric object. String forms: `'mae'` or `'DILATE(alpha=0.5, gamma=0.01)'`.

**`metric_to_spec(metric)`** — inverse of `resolve_metric`: instance → string. Used by `Forecaster` to make metric configuration checkpointable without pickling the live `nn.Module`.

**`assert_differentiable(metric)`** — raises `TypeError` if metric has `differentiable = False`.

**Metric classes:** `MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `DILATE`, `NSE`, `AlphaNSE` — all wrap `nn.Module`. `DILATE` has `eval_on_normalized = True` (soft-DTW overflows on large denormalized values). `NSE` (Nash-Sutcliffe Efficiency) and `AlphaNSE` (variability ratio, `std(pred)/std(obs)` — the α term from KGE) are higher-is-better and set `differentiable = False` so `assert_differentiable` blocks them from being used as `loss_function` (minimizing either directly would push a model the wrong way). Both are computed per-call on whatever tensor is passed in; since `Forecaster`'s evaluation loop calls metrics per batch and averages the results, `validation_score='nse'` reports the mean of per-batch NSE values, not the single NSE of the full validation set computed at once.

---

## `check.ipynb`

Reference notebook demonstrating a full training run using the `DataSource` API:
1. Create `DataSource` descriptors for train/val/test time windows pointing to `data/data.csv`
2. Inspect `TimeSeriesDataset` output shapes
3. Train `LSTMHistoric` (16 hidden, 1 layer) for 5 epochs with MAE loss, RMSE val score, NSE also logged via `validation_logging_criteria=['nse']`. `train_source`/`val_source`/`test_source` all set `nodata_values=[-999]` — the target column has 84 `-999` sentinel rows (all in the 2014 tail of `test_source`), converted to `NaN`.
4. **Evaluation and plotting on `test_source`, for `LSTMHistoric` only** (needs `matplotlib`, added as a project dependency):
   - Predictions via `fc.predict(test_source)` directly. True values come from a small separate `TimeSeriesDataset` built with `target_col=TARGET_COL` — its only purpose is reading off `test_ds._index`'s `(source_idx, t)` pairs for window/date alignment, not running the model; `target_col` doesn't affect which windows get built (only `x_h`/`x_f` NaN does), so this covers the exact same windows `predict()`'s internal dataset does. Produces aligned, denormalized `(y_true, y_pred)` arrays of shape `(n_windows, horizon)`. Dates per lead time are derived from the recovered `t` values, not assumed contiguous — `TimeSeriesDataset` may skip NaN windows.
   - Training diagnostics: train/val loss (normalized), val_score (denormalized), and val NSE vs. epoch — three panels, since loss/val_score are on different scales and NSE is unbounded-above-zero on yet another scale
   - Hydrograph: observed vs. predicted discharge over time, lead times 1/4/7 days overlaid
   - Scatter plot: observed vs. predicted (lead=1d) with a 1:1 reference line
   - Flow duration curve: both series sorted descending vs. exceedance probability, log-y
   - MAE and NSE vs. lead time (1–7 days) — NSE computed via the actual `NSE` class from `scores_and_losses`, not a re-derived formula, with an explicit NaN-mask applied first since the metric classes don't skip NaN on their own the way `np.nanmean` does
5. Train `LSTMEncoderDecoder` (16 hidden, 1 layer, downscale 8) same settings — no evaluation/plotting cells (LSTMHistoric only)
