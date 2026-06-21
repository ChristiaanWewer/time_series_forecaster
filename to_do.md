# Data Loading — Requirements

## Dataset descriptor
Each dataset is described by a single object with the following fields:
- `csv`: path to CSV file (optional)
- `zarr`: path to Zarr store (optional)
- `location`: patch centre as pixel index `(ix, iy)` OR coordinate value `(cx, cy)` — required if `zarr` is given; loader handles coordinate-to-index conversion
- `start`: datetime string
- `end`: datetime string
- At least one of `csv` or `zarr` must be present; both allowed for `{1D, 2D}` combined input

## Data sources
- CSV → reads file, slices by `start`/`end`
- NetCDF 1D variable → reads named variable, slices by `start`/`end`
- Zarr store → lazy patch reads, slices by `start`/`end`
- Multiple dataset descriptors can point to the same Zarr store; store is opened once and shared

## Coordinate handling
- Coordinate dimension names are configurable (lat/lon, x/y, or any)
- No assumption baked in; specified per Zarr store

## Patch extraction
- Patch size NxN configurable (HPO-tunable)
- Patch centred on specified location
- Zarr chunk size independent of patch size (set at conversion time; patch size can be any value ≤ chunk size)

## Tensor structure
- `x_h`: `(batch, seq_len, n_tabular_features)` — present if CSV/NetCDF 1D given
- `x_f`: `(batch, horizon, n_tabular_features)` — present if CSV/NetCDF 1D given
- `x_h_grid`: `(batch, seq_len, C, H, W)` — present if Zarr given
- `x_f_grid`: `(batch, horizon, C, H, W)` — present if Zarr given
- `x_static_grid`: `(batch, C, H, W)` — static 2D layer, no time dimension
- `y`: `(batch, horizon, 1)` — always 1D for now

## Temporal windowing
- Sliding window: `seq_len` historic + `horizon` future
- Applied identically to 1D and 2D sources (same time indices across both)
- 1D and 2D sources must share the same time axis and frequency

## NaN handling
- `x_h`, `x_f`: skip entire sample if any NaN present in the window
- `x_h_grid`, `x_f_grid`: skip entire sample if any NaN present in the patch at any timestep
- `y`: mask in loss (not filtered)

## Normalization
- Per-variable, computed over train + val combined
- 1D: per-column mean/std (current behaviour)
- 2D: per-channel mean/std over all spatial positions and time

## Preprocessing utility
- `netcdf_to_zarr(input_path, output_path, chunk_sizes)` conversion helper
- `chunk_sizes` configurable per dimension (e.g. `{'time': 1, 'x': 256, 'y': 256}`)

## DataLoader settings (all configurable)
- `batch_size`
- `num_workers`
- `prefetch_factor`
- `pin_memory`
- `drop_last`
- `shuffle` (training loader only)

## Forecaster integration
- `training_datasets` and `validation_datasets` replace current `dfs_training_sets` / `dfs_validation_sets`
- Both accept lists of dataset descriptors
- Model type (1D, 2D, mixed) inferred from which fields are present in the descriptors

## agents.md
- update the file agents.md with the new implementation