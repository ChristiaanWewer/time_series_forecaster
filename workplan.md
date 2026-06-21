# Implementation Workplan

Implements everything in `to_do.md`. Steps are ordered by dependency — each step is a self-contained unit that can be reviewed before the next begins.

---

## Step 1 — Add dependencies to `pyproject.toml`

New packages required:

| Package | Why |
|---|---|
| `xarray` | NetCDF and Zarr reading with named coordinate dims |
| `zarr` | Chunked array store; parallel reads for DataLoader workers |
| `netCDF4` | xarray backend for reading `.nc` files |
| `dask` | Lazy evaluation when opening large Zarr/NetCDF arrays via xarray |

`pyproject.toml` `dependencies` currently empty — add all four.

---

## Step 2 — `DataSource` descriptor (`src/data/datasource.py`)

A plain dataclass that describes one dataset (one location, one time window, one or two file sources).

```python
@dataclass
class DataSource:
    start: str                        # e.g. '2000-01-01'
    end: str                          # e.g. '2014-12-31'
    csv: str | None = None            # path to CSV file (1D tabular)
    netcdf_1d: str | None = None      # path to NetCDF file for 1D variables
    netcdf_1d_vars: list = field(default_factory=list)  # variable names to read from netcdf_1d
    zarr: str | None = None           # path to Zarr store (2D grid)
    location: tuple | None = None     # (ix, iy) pixel OR (cx, cy) coord
    location_is_index: bool = True    # True = pixel index, False = coordinate value
    coord_dims: tuple = ('x', 'y')   # dimension names in the Zarr store
    target_col: str = 'target'        # column name in CSV/NetCDF 1D for y
    static_zarr_vars: list = field(default_factory=list)  # Zarr vars with no time dim
```

Validation in `__post_init__`:
- At least one of `csv`, `netcdf_1d`, or `zarr` must be set
- `location` required if `zarr` is set
- `netcdf_1d_vars` required if `netcdf_1d` is set

Exported from `src/data/__init__.py`.

---

## Step 3 — Zarr store manager (`src/data/zarr_store.py`)

Handles opening Zarr stores, coordinate-to-index conversion, and patch extraction. Multiple `DataSource` objects pointing to the same path share one open store.

```python
class ZarrStoreManager:
    _open_stores: dict[str, xr.Dataset] = {}  # class-level cache

    @classmethod
    def open(cls, path: str) -> xr.Dataset: ...

    @staticmethod
    def coord_to_index(ds, location, coord_dims, location_is_index) -> tuple[int, int]:
        # if location_is_index: return as-is
        # else: use xr.Dataset.indexes to find nearest grid point

    @staticmethod
    def read_patch(ds, vars, time_slice, ix, iy, patch_size) -> dict[str, np.ndarray]:
        # returns {var_name: array(time, H, W)} for dynamic vars
        # and {var_name: array(H, W)} for static vars
        # slices: x[ix - patch_size//2 : ix + patch_size//2 + 1, ...]
```

---

## Step 4 — Lazy `torch.utils.data.Dataset` (`src/data/timeseries_dataset.py`)

Replaces the current in-memory `TensorDataset` for grid data. For purely 1D (CSV-only) data, still loads into memory for speed.

```python
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sources: list[DataSource],
        seq_len: int,
        horizon: int,
        historic_cols: list[str],
        future_cols: list[str],
        patch_size: int,
        norm_stats: dict,           # filled in by Forecaster after computing stats
    ):
        # Build index: list of (source_idx, t) for all valid windows across all sources
        # Valid = x_h window fits, x_f window fits, no NaN in x_h/x_f/grid inputs
        self._index: list[tuple[int, int]] = []

    def __len__(self): return len(self._index)

    def __getitem__(self, i) -> dict:
        source_idx, t = self._index[i]
        # Read x_h, x_f from CSV or NetCDF 1D (already in memory)
        # Read x_h_grid, x_f_grid, x_static_grid from Zarr via ZarrStoreManager
        # Apply normalization
        # Return dict with only the keys that are present
        # Keys: x_h, x_f, x_h_grid, x_f_grid, x_static_grid, y
```

NaN policy per key:
- `x_h`, `x_f`: window excluded from index during construction
- `x_h_grid`, `x_f_grid`: window excluded from index during construction
- `y`: included; NaN is handled in the loss

Collation: standard PyTorch default collate works since all tensors from one source have fixed shapes. A custom `collate_fn` is needed if mixing sources with and without grid tensors (missing keys must be padded or the batch filtered to homogeneous type).

---

## Step 5 — Normalization over all sources (`src/data/normalization.py`)

Extracted into its own module so both 1D and 2D stats can be computed before building the datasets.

```python
def compute_norm_stats(
    sources: list[DataSource],
    seq_len: int,
    horizon: int,
    historic_cols: list[str],
    future_cols: list[str],
    patch_size: int,
) -> dict:
    # 1D: iterate over all CSV and NetCDF 1D sources, collect x_h and x_f columns → per-column mean/std
    # 2D: iterate over all Zarr sources, sample patches across time → per-channel mean/std
    # y: collect all non-NaN target values → scalar mean/std
    # Returns dict with keys: xh_mean, xh_std, xf_mean, xf_std,
    #                         grid_mean, grid_std (per-channel),
    #                         target_mean, target_std
```

This replaces the inline normalization block in `Forecaster.__init__`.

---

## Step 6 — NetCDF → Zarr conversion utility (`src/data/utils.py`)

```python
def netcdf_to_zarr(
    input_path: str,
    output_path: str,
    chunk_sizes: dict,   # e.g. {'time': 1, 'x': 256, 'y': 256}
):
    ds = xr.open_dataset(input_path)
    ds.chunk(chunk_sizes).to_zarr(output_path, mode='w')
```

Exported from `src/data/__init__.py`.

---

## Step 7 — Update `Forecaster` (`src/forecaster/forecaster.py`)

Replace current `dfs_training_sets` / `dfs_validation_sets` with `training_datasets` / `validation_datasets` (lists of `DataSource`). Add DataLoader knobs.

**Signature changes:**
```python
def __init__(
    self,
    ...
    training_datasets: list[DataSource],
    validation_datasets: list[DataSource],
    patch_size: int = 64,
    # DataLoader settings:
    batch_size: int = 512,
    num_workers: int = 0,
    prefetch_factor: int = 2,   # only applied when num_workers > 0
    pin_memory: bool = False,
    drop_last: bool = False,
    shuffle: bool = True,       # training loader only; validation always False
    ...
):
```

**Init flow:**
1. Call `compute_norm_stats(training_datasets + validation_datasets, ...)` → `norm_stats`
2. Build `TimeSeriesDataset(training_datasets, ..., norm_stats)` → training dataset
3. Build one `TimeSeriesDataset` per validation source → validation datasets
4. Wrap each in `DataLoader` with the new settings

**Training / eval loop:**
- `x_h`, `x_f`, `x_h_grid`, `x_f_grid`, `x_static_grid` arrive as dict keys (some may be absent)
- Pass full dict `X` to model; existing NaN filtering on `y` unchanged
- NaN filtering for grid already done at index-build time (Step 4)

**`predict`:** accepts either the old `TensorDataset` or a new `TimeSeriesDataset`; duck-typed via the dict batch format.

---

## Step 8 — Update model interface (`src/models/LSTM.py`)

Current models only use `X['x_h']` and `X['x_f']`. No change needed for existing models — they silently ignore unknown keys. New spatial models added later will use `X['x_h_grid']` etc.

No code changes in this step; document in `agents.md` that X is an open dict.

---

## Step 9 — Update exports and `check.ipynb`

- `src/data/__init__.py`: export `DataSource`, `netcdf_to_zarr`
- `check.ipynb`: update to use `DataSource` descriptors with `csv=` and `start=`/`end=` instead of pre-sliced DataFrames

---

## Step 10 — Update `agents.md`

Reflect all new files, classes, and changed Forecaster signature.

---

## File map after completion

```
src/
  data/
    __init__.py              — exports DataSource, netcdf_to_zarr
    datasource.py            — DataSource dataclass          [NEW]
    zarr_store.py            — ZarrStoreManager              [NEW]
    timeseries_dataset.py    — TimeSeriesDataset             [NEW]
    normalization.py         — compute_norm_stats            [NEW]
    utils.py                 — netcdf_to_zarr                [NEW]
    preprocessing.py         — keep for now (used internally for 1D in-memory path)
  forecaster/
    forecaster.py            — updated signature + DataLoader settings
  models/
    LSTM.py                  — unchanged
  utils/
    scores_and_losses.py     — unchanged
    DILATE/                  — unchanged
```
