import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.datasource import DataSource
from src.data.normalization import _read_1d_df
from src.data.zarr_store import ZarrStoreManager


def _collate_fn(batch: list[dict]) -> dict:
    common_keys = set.intersection(*[set(item.keys()) for item in batch])
    return {k: torch.stack([item[k] for item in batch]) for k in sorted(common_keys)}


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        sources: list[DataSource],
        seq_len: int,
        horizon: int,
        historic_cols: list[str],
        future_cols: list[str],
        patch_size: int,
        norm_stats: dict,
        dtype: torch.dtype = torch.float32,
    ):
        self._sources = sources
        self._seq_len = seq_len
        self._horizon = horizon
        self._historic_cols = historic_cols
        self._future_cols = future_cols
        self._patch_size = patch_size
        self._norm_stats = norm_stats
        self._dtype = dtype
        self._window_len = seq_len + horizon

        # Per-source pre-loaded 1D data: (all_hist, all_fut, all_y) or None
        self._1d_data: list = []
        # Per-source zarr time indices into the zarr store's full time axis
        self._zarr_time_indices: list = []
        # Per-source zarr per-timestep NaN validity mask
        self._zarr_valid: list = []

        self._index: list[tuple[int, int]] = []

        for src_idx, source in enumerate(sources):
            tab = self._load_1d(source)
            self._1d_data.append(tab)

            # Determine n_time for this source
            if tab is not None:
                n_time = tab[3]
            elif source.zarr:
                ds = ZarrStoreManager.open(source.zarr)
                time_mask = (
                    (ds.time.values >= np.datetime64(source.start)) &
                    (ds.time.values <= np.datetime64(source.end))
                )
                n_time = int(time_mask.sum())
            else:
                self._zarr_time_indices.append(None)
                self._zarr_valid.append(None)
                continue

            # Zarr time indices and NaN mask
            if source.zarr:
                ds = ZarrStoreManager.open(source.zarr)
                time_mask = (
                    (ds.time.values >= np.datetime64(source.start)) &
                    (ds.time.values <= np.datetime64(source.end))
                )
                zarr_idx = np.where(time_mask)[0]
                self._zarr_time_indices.append(zarr_idx)
                zarr_valid = self._build_zarr_nan_mask(source, ds, zarr_idx)
                self._zarr_valid.append(zarr_valid)
                if tab is not None and len(zarr_idx) != n_time:
                    raise ValueError(
                        f"Source {src_idx}: 1D data length ({n_time}) does not match "
                        f"zarr time length ({len(zarr_idx)})"
                    )
            else:
                self._zarr_time_indices.append(None)
                self._zarr_valid.append(None)

            # Build valid window index
            for t in range(n_time - self._window_len + 1):
                if tab is not None:
                    all_hist, all_fut, _, _ = tab
                    if all_hist.shape[1] > 0 and np.isnan(all_hist[t:t + seq_len]).any():
                        continue
                    if all_fut.shape[1] > 0 and np.isnan(all_fut[t + seq_len:t + seq_len + horizon]).any():
                        continue
                if source.zarr:
                    zarr_valid = self._zarr_valid[src_idx]
                    if not zarr_valid[t:t + self._window_len].all():
                        continue
                self._index.append((src_idx, t))

    def _load_1d(self, source: DataSource):
        df = _read_1d_df(source)
        if df is None:
            return None
        all_hist = (
            df[self._historic_cols].values.astype(np.float32)
            if self._historic_cols else np.empty((len(df), 0), dtype=np.float32)
        )
        all_fut = (
            df[self._future_cols].values.astype(np.float32)
            if self._future_cols else np.empty((len(df), 0), dtype=np.float32)
        )
        all_y = df[[source.target_col]].values.astype(np.float32)
        return all_hist, all_fut, all_y, len(df)

    def _build_zarr_nan_mask(self, source: DataSource, ds, zarr_idx: np.ndarray) -> np.ndarray:
        half = self._patch_size // 2
        dim_x, dim_y = source.coord_dims
        ix, iy = ZarrStoreManager.coord_to_index(
            ds, source.location, source.coord_dims, source.location_is_index
        )
        spatial_sel = {
            dim_x: slice(ix - half, ix + half + 1),
            dim_y: slice(iy - half, iy + half + 1),
        }
        dynamic_vars = [v for v in ds.data_vars if v not in source.static_zarr_vars]
        has_nan = np.zeros(len(zarr_idx), dtype=bool)
        for v in dynamic_vars:
            data = ds[v].isel(time=zarr_idx.tolist(), **spatial_sel).values  # (n_time, H, W)
            has_nan |= np.isnan(data).any(axis=(-2, -1))
        return ~has_nan

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        src_idx, t = self._index[i]
        source = self._sources[src_idx]
        ns = self._norm_stats
        result = {}

        if self._1d_data[src_idx] is not None:
            all_hist, all_fut, all_y, _ = self._1d_data[src_idx]

            xh = all_hist[t:t + self._seq_len].copy()
            xf = all_fut[t + self._seq_len:t + self._seq_len + self._horizon].copy()
            y = all_y[t + self._seq_len:t + self._seq_len + self._horizon].copy()

            if self._historic_cols and 'xh_mean' in ns:
                xh = (xh - ns['xh_mean']) / ns['xh_std']
            if self._future_cols and 'xf_mean' in ns:
                xf = (xf - ns['xf_mean']) / ns['xf_std']
            if 'target_mean' in ns:
                y = (y - ns['target_mean']) / ns['target_std']

            result['x_h'] = torch.tensor(xh, dtype=self._dtype)
            result['x_f'] = torch.tensor(xf, dtype=self._dtype)
            result['y'] = torch.tensor(y, dtype=self._dtype)

        if source.zarr:
            ds = ZarrStoreManager.open(source.zarr)
            ix, iy = ZarrStoreManager.coord_to_index(
                ds, source.location, source.coord_dims, source.location_is_index
            )
            zarr_idx = self._zarr_time_indices[src_idx]
            hist_time = zarr_idx[t:t + self._seq_len].tolist()
            fut_time = zarr_idx[t + self._seq_len:t + self._seq_len + self._horizon].tolist()

            dynamic_vars = ns.get('zarr_vars') or [v for v in ds.data_vars if v not in source.static_zarr_vars]

            x_h_grid = np.stack(
                list(ZarrStoreManager.read_patch(ds, dynamic_vars, hist_time, ix, iy, self._patch_size, source.coord_dims).values()),
                axis=1,
            )  # (seq_len, C, H, W)
            x_f_grid = np.stack(
                list(ZarrStoreManager.read_patch(ds, dynamic_vars, fut_time, ix, iy, self._patch_size, source.coord_dims).values()),
                axis=1,
            )  # (horizon, C, H, W)

            if 'grid_mean' in ns:
                gm = ns['grid_mean'][None, :, None, None]
                gs = ns['grid_std'][None, :, None, None]
                x_h_grid = (x_h_grid - gm) / gs
                x_f_grid = (x_f_grid - gm) / gs

            result['x_h_grid'] = torch.tensor(x_h_grid, dtype=self._dtype)
            result['x_f_grid'] = torch.tensor(x_f_grid, dtype=self._dtype)

            if source.static_zarr_vars:
                static_patch = ZarrStoreManager.read_static_patch(
                    ds, source.static_zarr_vars, ix, iy, self._patch_size, source.coord_dims
                )
                x_static = np.stack(list(static_patch.values()), axis=0)  # (C_static, H, W)
                result['x_static_grid'] = torch.tensor(x_static, dtype=self._dtype)

        return result
