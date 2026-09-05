import numpy as np
import pandas as pd

from src.data.datasource import DataSource


def _read_1d_df(source: DataSource) -> pd.DataFrame | None:
    dfs = []
    if source.csv:
        df = pd.read_csv(source.csv, index_col=source.csv_index_col, parse_dates=True)
        if source.start is not None:
            df = df[df.index >= source.start]
        if source.end is not None:
            df = df[df.index <= source.end]
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
        df = dfs[0]
    else:
        df = dfs[0].join(dfs[1], how='inner', lsuffix='', rsuffix='_nc')
    if source.nodata_values:
        df = df.replace(source.nodata_values, np.nan)
    return df


def _merge_stats(
    n_a: float, mean_a: float, M2_a: float,
    n_b: float, mean_b: float, M2_b: float,
) -> tuple[float, float, float]:
    """Chan's parallel algorithm: merge two (n, mean, M2) accumulators."""
    n = n_a + n_b
    if n == 0:
        return 0.0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    M2 = M2_a + M2_b + delta ** 2 * (n_a * n_b / n)
    return n, mean, M2


def _is_binary(values: np.ndarray) -> bool:
    return len(values) > 0 and bool(np.all((values == 0) | (values == 1)))


def compute_norm_stats(
    sources: list[DataSource],
    historic_cols: list[str],
    future_cols: list[str],
    target_col: str | None = 'y',
) -> dict:
    stats = {}
    all_cols = list(dict.fromkeys(historic_cols + future_cols))

    accum: dict[str, tuple[float, float, float]] = {c: (0.0, 0.0, 0.0) for c in all_cols}
    target_accum = (0.0, 0.0, 0.0)
    group_accum: dict[str, tuple[float, float, float]] = {}

    binary_cols: set[str] = set()
    for source in sources:
        binary_cols.update(source.binary_cols)

    has_1d = False
    for source in sources:
        df = _read_1d_df(source)
        if df is None:
            continue
        has_1d = True

        for col in all_cols:
            if col not in df.columns:
                continue
            vals = df[col].values.astype(np.float64)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            if col in binary_cols or _is_binary(vals):
                binary_cols.add(col)
                continue
            n_b = float(len(vals))
            mean_b = float(vals.mean())
            M2_b = float(((vals - mean_b) ** 2).sum())
            accum[col] = _merge_stats(*accum[col], n_b, mean_b, M2_b)

        if target_col is not None and target_col in df.columns:
            target_vals = df[target_col].values.astype(np.float64)
            target_vals = target_vals[~np.isnan(target_vals)]
            if len(target_vals) > 0:
                n_b = float(len(target_vals))
                mean_b = float(target_vals.mean())
                M2_b = float(((target_vals - mean_b) ** 2).sum())
                target_accum = _merge_stats(*target_accum, n_b, mean_b, M2_b)
                key = source.group_key
                group_accum[key] = _merge_stats(*group_accum.get(key, (0.0, 0.0, 0.0)), n_b, mean_b, M2_b)

    if has_1d:
        xh_mean, xh_std = [], []
        xf_mean, xf_std = [], []
        for col in historic_cols:
            n, mean, M2 = accum[col]
            if col in binary_cols or n < 2:
                xh_mean.append(0.0)
                xh_std.append(1.0)
            else:
                xh_mean.append(mean)
                xh_std.append(max(float(np.sqrt(M2 / (n - 1))), 1e-8))
        for col in future_cols:
            n, mean, M2 = accum[col]
            if col in binary_cols or n < 2:
                xf_mean.append(0.0)
                xf_std.append(1.0)
            else:
                xf_mean.append(mean)
                xf_std.append(max(float(np.sqrt(M2 / (n - 1))), 1e-8))

        stats['xh_mean'] = np.array(xh_mean, dtype=np.float32)
        stats['xh_std'] = np.array(xh_std, dtype=np.float32)
        stats['xf_mean'] = np.array(xf_mean, dtype=np.float32)
        stats['xf_std'] = np.array(xf_std, dtype=np.float32)

        n, mean, M2 = target_accum
        stats['target_mean'] = float(mean)
        stats['target_std'] = max(float(np.sqrt(M2 / (n - 1))) if n >= 2 else 1.0, 1e-8)

        stats['group_target_std'] = {
            key: max(float(np.sqrt(M2_g / (n_g - 1))) if n_g >= 2 else 1.0, 1e-8)
            for key, (n_g, _, M2_g) in group_accum.items()
        }

    stats['binary_cols'] = binary_cols

    return stats
