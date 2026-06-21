import numpy as np
import pandas as pd

from src.data.datasource import DataSource
from src.data.zarr_store import ZarrStoreManager


def _read_1d_df(source: DataSource) -> pd.DataFrame | None:
    dfs = []
    if source.csv:
        df = pd.read_csv(source.csv, index_col=source.csv_index_col, parse_dates=True)
        df = df[(df.index >= source.start) & (df.index <= source.end)]
        dfs.append(df)
    if source.netcdf_1d:
        import xarray as xr
        ds = xr.open_dataset(source.netcdf_1d)
        ds = ds.sel(time=slice(source.start, source.end))
        df_nc = ds[source.netcdf_1d_vars].to_dataframe().reset_index(drop=False)
        df_nc = df_nc.set_index('time') if 'time' in df_nc.columns else df_nc
        dfs.append(df_nc)
    if not dfs:
        return None
    if len(dfs) == 1:
        return dfs[0]
    return dfs[0].join(dfs[1], how='inner', lsuffix='', rsuffix='_nc')


def compute_norm_stats(
    sources: list[DataSource],
    seq_len: int,
    horizon: int,
    historic_cols: list[str],
    future_cols: list[str],
    patch_size: int,
) -> dict:
    stats = {}

    # --- 1D stats ---
    hist_accum = {c: [] for c in historic_cols}
    fut_accum = {c: [] for c in future_cols}
    y_accum = []

    for source in sources:
        df = _read_1d_df(source)
        if df is None:
            continue
        for c in historic_cols:
            if c in df.columns:
                hist_accum[c].append(df[c].values.astype(np.float32))
        for c in future_cols:
            if c in df.columns:
                fut_accum[c].append(df[c].values.astype(np.float32))
        if source.target_col in df.columns:
            vals = df[source.target_col].values.astype(np.float32)
            y_accum.append(vals[~np.isnan(vals)])

    if historic_cols and any(hist_accum[c] for c in historic_cols):
        stats['xh_mean'] = np.array(
            [np.nanmean(np.concatenate(hist_accum[c])) if hist_accum[c] else 0.0 for c in historic_cols],
            dtype=np.float32,
        )
        stats['xh_std'] = np.array(
            [max(float(np.nanstd(np.concatenate(hist_accum[c]))), 1e-8) if hist_accum[c] else 1.0 for c in historic_cols],
            dtype=np.float32,
        )

    if future_cols and any(fut_accum[c] for c in future_cols):
        stats['xf_mean'] = np.array(
            [np.nanmean(np.concatenate(fut_accum[c])) if fut_accum[c] else 0.0 for c in future_cols],
            dtype=np.float32,
        )
        stats['xf_std'] = np.array(
            [max(float(np.nanstd(np.concatenate(fut_accum[c]))), 1e-8) if fut_accum[c] else 1.0 for c in future_cols],
            dtype=np.float32,
        )

    if y_accum:
        all_y = np.concatenate(y_accum)
        stats['target_mean'] = float(np.nanmean(all_y))
        stats['target_std'] = max(float(np.nanstd(all_y)), 1e-8)

    # --- 2D (grid) stats ---
    zarr_sources = [s for s in sources if s.zarr]
    if zarr_sources:
        # Determine variable order from first zarr source
        first_ds = ZarrStoreManager.open(zarr_sources[0].zarr)
        dynamic_vars = [v for v in first_ds.data_vars if v not in zarr_sources[0].static_zarr_vars]
        stats['zarr_vars'] = dynamic_vars

        grid_accum = {v: [] for v in dynamic_vars}

        for source in zarr_sources:
            ds = ZarrStoreManager.open(source.zarr)
            ix, iy = ZarrStoreManager.coord_to_index(
                ds, source.location, source.coord_dims, source.location_is_index
            )
            half = patch_size // 2
            dim_x, dim_y = source.coord_dims

            time_mask = (
                (ds.time.values >= np.datetime64(source.start)) &
                (ds.time.values <= np.datetime64(source.end))
            )
            time_indices = np.where(time_mask)[0].tolist()

            spatial_sel = {
                dim_x: slice(ix - half, ix + half + 1),
                dim_y: slice(iy - half, iy + half + 1),
            }
            src_dynamic = [v for v in ds.data_vars if v not in source.static_zarr_vars]
            for v in dynamic_vars:
                if v in src_dynamic:
                    data = ds[v].isel(time=time_indices, **spatial_sel).values
                    grid_accum[v].append(data.flatten())

        stats['grid_mean'] = np.array(
            [np.nanmean(np.concatenate(grid_accum[v])) if grid_accum[v] else 0.0 for v in dynamic_vars],
            dtype=np.float32,
        )
        stats['grid_std'] = np.array(
            [max(float(np.nanstd(np.concatenate(grid_accum[v]))), 1e-8) if grid_accum[v] else 1.0 for v in dynamic_vars],
            dtype=np.float32,
        )

    return stats
