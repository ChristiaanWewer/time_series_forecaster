import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.data.datasource import DataSource

CORPUS_FILE = 'corpus.npy'
MANIFEST_FILE = 'manifest.json'


def _files(source: DataSource) -> list[str]:
    return [str(Path(p).resolve()) for p in (source.csv, source.netcdf_1d) if p]


def bundle_key(source: DataSource) -> str:
    """Identity of a source's rows inside a bundle: every DataSource field that changes
    which rows are read (files, netcdf vars, index column, start, end, nodata values).
    Unlike group_key, two date ranges of one series get different keys."""
    return json.dumps({
        'files': _files(source),
        'netcdf_1d_vars': list(source.netcdf_1d_vars),
        'csv_index_col': source.csv_index_col,
        'start': source.start,
        'end': source.end,
        'nodata_values': list(source.nodata_values),
    }, sort_keys=True, default=str)


def file_fingerprint(source: DataSource) -> dict:
    """Size and modification time of each file the source reads, used to detect that a
    file changed after the bundle was compiled."""
    fingerprint = {}
    for path in _files(source):
        st = os.stat(path)
        fingerprint[path] = [st.st_size, st.st_mtime_ns]
    return fingerprint


@dataclass
class BundleManifest:
    """Describes a compiled bundle on disk (manifest.json next to corpus.npy): the column
    order of the row-major float32 corpus, and for each source its bundle key, starting
    row, row count, file fingerprint and per-column statistics."""
    columns: list[str]
    entries: list[dict]

    def save(self, path: Path) -> None:
        """Write the manifest as manifest.json inside `path`."""
        tmp = Path(path) / (MANIFEST_FILE + '.tmp')
        tmp.write_text(json.dumps({'columns': self.columns, 'entries': self.entries}))
        os.replace(tmp, Path(path) / MANIFEST_FILE)

    @classmethod
    def load(cls, path: Path) -> 'BundleManifest':
        """Read manifest.json from `path`."""
        raw = json.loads((Path(path) / MANIFEST_FILE).read_text())
        for entry in raw['entries']:
            entry['column_stats'] = {
                col: (n, mean, M2, bool(is_binary))
                for col, (n, mean, M2, is_binary) in entry['column_stats'].items()
            }
        return cls(columns=raw['columns'], entries=raw['entries'])

    def resolve(self, sources: list[DataSource]) -> list[dict]:
        """The manifest entry of each source, in the order of `sources`, matched by bundle
        key and never by position. Raises if a source is missing or its files changed."""
        by_key = {entry['key']: entry for entry in self.entries}
        resolved = []
        for source in sources:
            entry = by_key.get(bundle_key(source))
            if entry is None:
                raise ValueError(f'it does not contain source {source}.')
            if entry['fingerprint'] != file_fingerprint(source):
                raise ValueError(f'the files of source {source} changed after it was compiled.')
            resolved.append(entry)
        return resolved
