import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.timeseries_data import TimeSeriesData
from src.data.windows import build_window_index


class TimeSeriesDataset(Dataset):
    """Turns a list of window indices into one batch of normalized tensors
    (x_h, x_f, y, group_idx), reading rows from a TimeSeriesData. 'memory' gathers the
    whole batch at once; 'disk' does the same gather with its reads sorted by position,
    so the disk is read mostly forward. The output is bitwise identical to
    TimeSeriesDatasetOld."""

    def __init__(
        self,
        data: TimeSeriesData,
        seq_len: int,
        horizon: int,
        historic_cols: list[str],
        future_cols: list[str],
        norm_stats: dict,
        target_col: str | None = 'y',
        dtype: torch.dtype = torch.float32,
        source_positions: list[int] | None = None,
    ):
        """Resolve column positions, build the window index for `source_positions` (all
        sources by default) and turn norm_stats into per-column mean/std vectors.
        group_idx is the position within `source_positions`, so a single-source
        validation view always has group_idx 0, as today."""
        self.data = data
        self._seq_len = seq_len
        self._horizon = horizon
        self._window_len = seq_len + horizon
        self._dtype = dtype
        positions = np.arange(len(data)) if source_positions is None else np.asarray(source_positions)
        self._starts = data.offsets[positions]

        historic_idx = data.column_indices(historic_cols)
        future_idx = data.column_indices(future_cols)
        target_idx = data.column_indices([target_col]) if target_col is not None else np.empty(0, np.int64)
        self.window_index = build_window_index(data, positions, seq_len, horizon, historic_idx, future_idx)

        # gather only the columns this run uses, then address them by local position
        self._used_cols = np.array(list(dict.fromkeys([*historic_idx, *future_idx, *target_idx])), dtype=np.int64)
        local = {c: i for i, c in enumerate(self._used_cols)}
        self._h = np.array([local[c] for c in historic_idx], dtype=np.int64)
        self._f = np.array([local[c] for c in future_idx], dtype=np.int64)
        self._y = np.array([local[c] for c in target_idx], dtype=np.int64)
        self._has_target = target_col is not None

        ns = norm_stats
        self._xh_norm = (ns['xh_mean'], ns['xh_std']) if historic_cols and 'xh_mean' in ns else None
        self._xf_norm = (ns['xf_mean'], ns['xf_std']) if future_cols and 'xf_mean' in ns else None
        self._y_norm = (ns['target_mean'], ns['target_std']) if 'target_mean' in ns else None

    def __len__(self) -> int:
        """The number of valid windows."""
        return len(self.window_index)

    def __getitem__(self, i: int) -> dict:
        """One sample, through the batched path with a batch of one. Kept for debugging
        and for the test that proves DataLoader uses __getitems__."""
        return {k: v[0] for k, v in self.__getitems__([i]).items()}

    def __getitems__(self, indices: list[int]) -> dict:
        """A whole batch in one call. PyTorch's DataLoader uses this hook instead of
        calling __getitem__ once per sample; requires an identity collate_fn."""
        windows = self.window_index[np.asarray(indices, dtype=np.int64)]
        return self._assemble(self._gather(windows), windows[:, 0])

    def _gather(self, windows: np.ndarray) -> np.ndarray:
        """Raw rows of the given windows as (batch, seq_len + horizon, n_cols), in one fancy
        index over data.rows. In 'disk' mode the reads are sorted first and the result is
        put back in the requested order, so sorting is invisible to the caller."""
        starts = self._starts[windows[:, 0]] + windows[:, 1]
        offsets = np.arange(self._window_len)
        if self.data.mode != 'disk':
            return self.data.rows[(starts[:, None] + offsets)[:, :, None], self._used_cols]
        order = np.argsort(starts, kind='stable')
        raw_sorted = self.data.rows[(starts[order][:, None] + offsets)[:, :, None], self._used_cols]
        raw = np.empty_like(raw_sorted)
        raw[order] = raw_sorted
        return raw

    def _assemble(self, raw: np.ndarray, group_idx: np.ndarray) -> dict:
        """Split raw windows into x_h, x_f and y, normalize them as whole-array arithmetic
        (binary columns have mean 0 and std 1, so they pass through unchanged), and
        convert them to tensors."""
        s = self._seq_len
        xh = raw[:, :s, self._h]
        xf = raw[:, s:, self._f]
        if self._xh_norm is not None:
            xh = (xh - self._xh_norm[0]) / self._xh_norm[1]
        if self._xf_norm is not None:
            xf = (xf - self._xf_norm[0]) / self._xf_norm[1]

        result = {
            'x_h': torch.from_numpy(xh).to(self._dtype),
            'x_f': torch.from_numpy(xf).to(self._dtype),
            'group_idx': torch.from_numpy(group_idx.astype(np.int64)),
        }
        if self._has_target:
            y = raw[:, s:, self._y]
            if self._y_norm is not None:
                y = (y - self._y_norm[0]) / self._y_norm[1]
            result['y'] = torch.from_numpy(y).to(self._dtype)
        return result
