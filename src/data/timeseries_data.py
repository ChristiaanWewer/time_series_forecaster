import shutil
from pathlib import Path

import numpy as np
from pandas.api.types import is_numeric_dtype

from src.data.bundle import CORPUS_FILE, MANIFEST_FILE, BundleManifest, bundle_key, file_fingerprint
from src.data.datasource import DataSource
from src.data.normalization import _read_1d_df, column_stats

_MODES = ('memory', 'disk')


def _parse_source(source: DataSource) -> tuple[list[str], np.ndarray, dict]:
    """Read one DataSource through _read_1d_df and return its numeric column names, its
    rows as a float32 array (n_rows, n_cols), and its per-column statistics
    (see normalization.column_stats). Shared by both modes, so both store the same rows
    and the same statistics."""
    df = _read_1d_df(source)
    df = df[[c for c in df.columns if is_numeric_dtype(df[c])]]
    values = df.values.astype(np.float32)
    return [str(c) for c in df.columns], values, column_stats(df)


def _offsets(n_rows: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(n_rows)[:-1]]).astype(np.int64)


class TimeSeriesData:
    """The user-facing collection of DataSources, prepared up front.

    All sources are presented as one flat float32 row space `rows` (total_rows, n_cols),
    in which source k occupies `offsets[k] .. offsets[k] + n_rows[k]`, in the order of
    the user's list. `mode='memory'` holds `rows` in RAM; `mode='disk'` holds it in a
    bundle at `path`, read through a memory map. The per-source column statistics are
    computed while reading, so normalization stats never need another pass over the data."""

    def __init__(self, sources: list[DataSource], mode: str = 'memory', path: str | Path | None = None):
        """Validate the mode/path combination, then do all expensive work now: parse
        everything into RAM ('memory'), or compile or reuse the bundle at `path` ('disk')."""
        if isinstance(sources, DataSource):
            sources = [sources]
        if not sources:
            raise ValueError('TimeSeriesData needs at least one DataSource.')
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
        if mode == 'disk' and path is None:
            raise ValueError("mode='disk' requires a path to store the bundle in.")
        if mode == 'memory' and path is not None:
            raise ValueError("path is only used with mode='disk'.")

        self.sources = list(sources)
        self.mode = mode
        self.path = Path(path) if path is not None else None
        self._rows = None

        if mode == 'memory':
            self._load_memory()
        else:
            self._prepare_disk()

    def __len__(self) -> int:
        """The number of sources."""
        return len(self.sources)

    @property
    def rows(self) -> np.ndarray:
        """The (total_rows, n_cols) float32 array: the resident array in 'memory' mode, or
        the memory map in 'disk' mode, opened lazily on first use in the calling process."""
        if self._rows is None:
            self._rows = np.load(self.path / CORPUS_FILE, mmap_mode='r')
        return self._rows

    def column_indices(self, cols: list[str]) -> np.ndarray:
        """Positions of `cols` in `rows`, raising a clear error if a column is missing from
        any source (rather than silently returning all-NaN windows)."""
        for col in cols:
            for source, stats in zip(self.sources, self.column_stats):
                if col not in stats:
                    raise KeyError(f'column {col!r} is missing (or not numeric) in source {source}.')
        position = {c: i for i, c in enumerate(self.columns)}
        return np.array([position[c] for c in cols], dtype=np.int64)

    def _load_memory(self) -> None:
        """'memory' mode: parse every source, take the union of their columns (NaN where a
        source lacks one), and concatenate everything into one resident array."""
        parsed = [_parse_source(s) for s in self.sources]
        self.columns = list(dict.fromkeys(c for names, _, _ in parsed for c in names))
        self.n_rows = np.array([len(values) for _, values, _ in parsed], dtype=np.int64)
        self.offsets = _offsets(self.n_rows)
        self.column_stats = [stats for _, _, stats in parsed]

        position = {c: i for i, c in enumerate(self.columns)}
        rows = np.full((int(self.n_rows.sum()), len(self.columns)), np.nan, dtype=np.float32)
        for (names, values, _), start in zip(parsed, self.offsets):
            rows[start:start + len(values), [position[c] for c in names]] = values
        self._rows = rows

    def _prepare_disk(self) -> None:
        """'disk' mode: reuse the bundle at `path` if it contains every source and none of
        their files changed, compile it if `path` is empty, and raise otherwise."""
        self.path.mkdir(parents=True, exist_ok=True)
        if not (self.path / MANIFEST_FILE).exists():
            if any(self.path.iterdir()):
                raise ValueError(
                    f'{self.path} is not empty but holds no complete bundle (an interrupted '
                    f'compile leaves one behind). Delete the folder or use a new path.'
                )
            self._compile_bundle()

        manifest = BundleManifest.load(self.path)
        try:
            entries = manifest.resolve(self.sources)
        except ValueError as e:
            raise ValueError(
                f'The bundle at {self.path} cannot be reused: {e} '
                f'Delete the folder or use a new path to compile a fresh bundle.'
            ) from None
        self.columns = manifest.columns
        self.offsets = np.array([e['start'] for e in entries], dtype=np.int64)
        self.n_rows = np.array([e['n_rows'] for e in entries], dtype=np.int64)
        self.column_stats = [e['column_stats'] for e in entries]

    def _compile_bundle(self) -> None:
        """Write corpus.npy and manifest.json at `path`, holding at most one source in RAM.
        Pass 1 parses each source into a temporary file and collects columns, row counts
        and statistics; pass 2 copies those into the final corpus at their column slots."""
        tmp = self.path / '.compile'
        tmp.mkdir()
        parts, seen = [], set()
        for source in self.sources:
            key = bundle_key(source)
            if key in seen:
                continue
            seen.add(key)
            fingerprint = file_fingerprint(source)
            names, values, stats = _parse_source(source)
            part_file = tmp / f'{len(parts)}.npy'
            np.save(part_file, values)
            parts.append(dict(key=key, fingerprint=fingerprint, names=names,
                              n_rows=len(values), column_stats=stats, file=part_file))

        columns = list(dict.fromkeys(c for part in parts for c in part['names']))
        position = {c: i for i, c in enumerate(columns)}
        n_rows = np.array([part['n_rows'] for part in parts], dtype=np.int64)
        offsets = _offsets(n_rows)

        corpus = np.lib.format.open_memmap(
            self.path / CORPUS_FILE, mode='w+', dtype=np.float32,
            shape=(int(n_rows.sum()), len(columns)),
        )
        entries = []
        for part, start in zip(parts, offsets):
            block = np.full((part['n_rows'], len(columns)), np.nan, dtype=np.float32)
            block[:, [position[c] for c in part['names']]] = np.load(part['file'])
            corpus[start:start + part['n_rows']] = block
            entries.append(dict(key=part['key'], start=int(start), n_rows=part['n_rows'],
                                fingerprint=part['fingerprint'], column_stats=part['column_stats']))
        corpus.flush()
        del corpus
        shutil.rmtree(tmp)
        # written last, so an interrupted compile never looks like a valid bundle
        BundleManifest(columns=columns, entries=entries).save(self.path)

    def __getstate__(self) -> dict:
        """Pickle without the memory map, so DataLoader workers reopen it themselves
        instead of receiving a full copy of the data."""
        state = self.__dict__.copy()
        if self.mode == 'disk':
            state['_rows'] = None
        return state
