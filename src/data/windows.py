import numpy as np

from src.data.timeseries_data import TimeSeriesData


def build_window_index(
    data: TimeSeriesData,
    source_positions: list[int],
    seq_len: int,
    horizon: int,
    historic_idx: np.ndarray,
    future_idx: np.ndarray,
) -> np.ndarray:
    """All trainable windows of the given sources, as an int64 array (n_windows, 2) of
    (position in source_positions, start row within that source), ordered by source and
    then by start row. Reads one source at a time, so in 'disk' mode it is one sequential
    pass over those sources."""
    parts = [np.empty((0, 2), dtype=np.int64)]
    for local, src in enumerate(source_positions):
        start, n = data.offsets[src], data.n_rows[src]
        starts = _valid_starts(data.rows[start:start + n], seq_len, horizon, historic_idx, future_idx)
        # load-bearing: in the flat row space a window past its source's end would silently
        # read the next, unrelated source
        assert starts.size == 0 or starts[-1] + seq_len + horizon <= n
        parts.append(np.column_stack([np.full(len(starts), local, dtype=np.int64), starts]))
    return np.concatenate(parts).astype(np.int64)


def _window_has_nan(row_has_nan: np.ndarray, length: int) -> np.ndarray:
    """For every start t, whether any of rows t .. t + length - 1 has a NaN."""
    counts = np.concatenate([[0], np.cumsum(row_has_nan, dtype=np.int64)])
    return counts[length:] - counts[:-length] > 0


def _valid_starts(
    block: np.ndarray, seq_len: int, horizon: int, historic_idx: np.ndarray, future_idx: np.ndarray,
) -> np.ndarray:
    """Start rows within one source's block whose historic span is NaN-free in the
    historic columns and whose future span is NaN-free in the future columns. Targets
    are not checked; NaN targets are masked per batch in the training loop."""
    n_starts = len(block) - (seq_len + horizon) + 1
    if n_starts <= 0:
        return np.empty(0, dtype=np.int64)
    valid = np.ones(n_starts, dtype=bool)
    if len(historic_idx):
        bad = np.isnan(block[:, historic_idx]).any(axis=1)
        valid &= ~_window_has_nan(bad, seq_len)[:n_starts]
    if len(future_idx):
        bad = np.isnan(block[:, future_idx]).any(axis=1)
        valid &= ~_window_has_nan(bad, horizon)[seq_len:seq_len + n_starts]
    return np.flatnonzero(valid).astype(np.int64)
