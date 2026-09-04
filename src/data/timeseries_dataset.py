import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.datasource import DataSource
from src.data.normalization import _read_1d_df


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        sources: list[DataSource],
        seq_len: int,
        horizon: int,
        historic_cols: list[str],
        future_cols: list[str],
        norm_stats: dict,
        target_col: str | None = 'y',
        dtype: torch.dtype = torch.float32,
    ):
        self._sources = sources
        self._seq_len = seq_len
        self._horizon = horizon
        self._historic_cols = historic_cols
        self._future_cols = future_cols
        self._norm_stats = norm_stats
        self._target_col = target_col
        self._dtype = dtype
        self._window_len = seq_len + horizon

        self._1d_data: list = []

        self._index: list[tuple[int, int]] = []

        for src_idx, source in enumerate(sources):
            tab = self._load_1d(source)
            self._1d_data.append(tab)
            n_time = tab[3]

            for t in range(n_time - self._window_len + 1):
                all_hist, all_fut, _, _ = tab
                if all_hist.shape[1] > 0 and np.isnan(all_hist[t:t + seq_len]).any():
                    continue
                if all_fut.shape[1] > 0 and np.isnan(all_fut[t + seq_len:t + seq_len + horizon]).any():
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
        all_y = (
            df[[self._target_col]].values.astype(np.float32)
            if self._target_col is not None else None
        )
        return all_hist, all_fut, all_y, len(df)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        src_idx, t = self._index[i]
        ns = self._norm_stats
        all_hist, all_fut, all_y, _ = self._1d_data[src_idx]

        xh = all_hist[t:t + self._seq_len].copy()
        xf = all_fut[t + self._seq_len:t + self._seq_len + self._horizon].copy()

        binary_cols = ns.get('binary_cols', set())
        if self._historic_cols and 'xh_mean' in ns:
            for col_idx, col in enumerate(self._historic_cols):
                if col not in binary_cols:
                    xh[:, col_idx] = (xh[:, col_idx] - ns['xh_mean'][col_idx]) / ns['xh_std'][col_idx]
        if self._future_cols and 'xf_mean' in ns:
            for col_idx, col in enumerate(self._future_cols):
                if col not in binary_cols:
                    xf[:, col_idx] = (xf[:, col_idx] - ns['xf_mean'][col_idx]) / ns['xf_std'][col_idx]

        result = {
            'x_h': torch.tensor(xh, dtype=self._dtype),
            'x_f': torch.tensor(xf, dtype=self._dtype),
        }
        if all_y is not None:
            y = all_y[t + self._seq_len:t + self._seq_len + self._horizon].copy()
            if 'target_mean' in ns:
                y = (y - ns['target_mean']) / ns['target_std']
            result['y'] = torch.tensor(y, dtype=self._dtype)
        return result
