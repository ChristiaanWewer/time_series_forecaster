import pickle

import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import DataSource, TimeSeriesData
from src.data.bundle import CORPUS_FILE, BundleManifest
from src.data.normalization import compute_norm_stats, norm_stats_from
from src.data.timeseries_dataset import TimeSeriesDataset
from src.data.timeseries_dataset_old import TimeSeriesDatasetOld
from src.forecaster import Forecaster
from src.forecaster.forecaster import _identity_collate, _make_loader
from src.forecaster.prefetcher import _make_prefetcher
from src.models import LSTMHistoric
from src.utils.scores_and_losses import MAE

SEQ_LEN, HORIZON = 10, 3
HISTORIC = ['temp', 'precip', 'flag']
FUTURE = ['fc', 'temp']  # overlaps HISTORIC on purpose


def _write_csv(path, n, seed, extra=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'temp': rng.normal(10, 3, n),
        'precip': rng.exponential(2, n),
        'flag': rng.integers(0, 2, n).astype(float),
        'fc': rng.normal(10, 3, n),
        'y': rng.normal(5, 2, n),
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    df.loc[df.index[rng.choice(n, n // 15, replace=False)], 'temp'] = np.nan
    df.loc[df.index[rng.choice(n, n // 20, replace=False)], 'fc'] = np.nan
    df.loc[df.index[rng.choice(n, n // 10, replace=False)], 'y'] = np.nan
    df.loc[df.index[rng.choice(n, 3, replace=False)], 'precip'] = -999
    if extra:
        df['extra'] = rng.normal(0, 1, n)
    df.to_csv(path)
    return str(path)


@pytest.fixture
def sources(tmp_path):
    a = _write_csv(tmp_path / 'a.csv', 160, 0)
    b = _write_csv(tmp_path / 'b.csv', 70, 1, extra=True)
    c = _write_csv(tmp_path / 'c.csv', 8, 2)  # shorter than one window
    return [
        DataSource(csv=a, end='2020-04-01', nodata_values=[-999]),
        DataSource(csv=b, nodata_values=[-999]),
        DataSource(csv=c),
        DataSource(csv=a, start='2020-04-02', nodata_values=[-999]),  # same series, other range
    ]


def _data(sources, mode, tmp_path, name='bundle'):
    return TimeSeriesData(sources, mode=mode, path=tmp_path / name if mode == 'disk' else None)


def _dataset(data, norm_stats, target_col='y', **kwargs):
    return TimeSeriesDataset(data, SEQ_LEN, HORIZON, HISTORIC, FUTURE, norm_stats, target_col=target_col, **kwargs)


def _bits(t: torch.Tensor) -> np.ndarray:
    """Bit pattern of a tensor, so NaNs compare equal and nothing is compared at tolerance."""
    return t.numpy().view(np.int32 if t.dtype == torch.float32 else np.int64)


def _assert_batches_identical(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        np.testing.assert_array_equal(_bits(a[k]), _bits(b[k]), err_msg=k)


def _all(ds):
    return ds.__getitems__(list(range(len(ds))))


@pytest.mark.parametrize('mode', ['memory', 'disk'])
def test_norm_stats_match_reference(sources, tmp_path, mode):
    reference = compute_norm_stats(sources, HISTORIC, FUTURE, 'y')
    stats = norm_stats_from(_data(sources, mode, tmp_path), HISTORIC, FUTURE, 'y')
    assert stats.keys() == reference.keys()
    for k in reference:
        if isinstance(reference[k], np.ndarray):
            np.testing.assert_array_equal(stats[k], reference[k])
        else:
            assert stats[k] == reference[k], k
    assert 'flag' in stats['binary_cols']


@pytest.mark.parametrize('target_col', ['y', None])
@pytest.mark.parametrize('mode', ['memory', 'disk'])
def test_bitwise_equivalence_with_old_dataset(sources, tmp_path, mode, target_col):
    norm_stats = compute_norm_stats(sources, HISTORIC, FUTURE, 'y')
    old = TimeSeriesDatasetOld(sources, SEQ_LEN, HORIZON, HISTORIC, FUTURE, norm_stats, target_col=target_col)
    new = _dataset(_data(sources, mode, tmp_path), norm_stats, target_col=target_col)

    assert len(new) == len(old) > 0
    np.testing.assert_array_equal(new.window_index, np.array(old._index))
    expected = {k: torch.stack([old[i][k] for i in range(len(old))]) for k in old[0]}
    _assert_batches_identical(_all(new), expected)


def test_batched_path_matches_per_sample(sources, tmp_path):
    ds = _dataset(_data(sources, 'memory', tmp_path), compute_norm_stats(sources, HISTORIC, FUTURE))
    indices = [5, 0, 17, 3, 3]
    per_sample = {k: torch.stack([ds[i][k] for i in indices]) for k in ds[0]}
    _assert_batches_identical(ds.__getitems__(indices), per_sample)


class _NoPerSample(TimeSeriesDataset):
    def __getitem__(self, i):
        raise AssertionError('DataLoader fell back to per-sample __getitem__')


def test_dataloader_takes_the_batched_path(sources, tmp_path):
    data = _data(sources, 'memory', tmp_path)
    norm_stats = norm_stats_from(data, HISTORIC, FUTURE)
    ds = _NoPerSample(data, SEQ_LEN, HORIZON, HISTORIC, FUTURE, norm_stats)
    batch = next(iter(DataLoader(ds, batch_size=7, collate_fn=_identity_collate)))
    assert batch['x_h'].shape == (7, SEQ_LEN, len(HISTORIC))

    # a loader configured the way Forecaster configures it yields the right batch
    plain = _dataset(data, norm_stats)
    loader = _make_loader(plain, 16, shuffle=False, device='cpu')
    _assert_batches_identical(next(iter(loader)), plain.__getitems__(list(range(16))))


def test_sorted_disk_reads_change_nothing(sources, tmp_path):
    norm_stats = compute_norm_stats(sources, HISTORIC, FUTURE)
    memory = _dataset(_data(sources, 'memory', tmp_path), norm_stats)
    disk = _dataset(_data(sources, 'disk', tmp_path), norm_stats)
    indices = np.random.default_rng(0).permutation(len(memory)).tolist()
    _assert_batches_identical(disk.__getitems__(indices), memory.__getitems__(indices))


def test_group_idx_follows_the_users_order_not_the_bundles(sources, tmp_path):
    a, b = sources[0], sources[1]
    _data([a, b], 'disk', tmp_path)  # bundle compiled in order a, b
    disk = _data([b, a], 'disk', tmp_path)  # reused in order b, a
    memory = _data([b, a], 'memory', tmp_path)

    assert [s.group_key for s in disk.sources] == [b.group_key, a.group_key]
    norm_stats = norm_stats_from(memory, HISTORIC, FUTURE)
    assert norm_stats_from(disk, HISTORIC, FUTURE)['group_target_std'] == norm_stats['group_target_std']
    _assert_batches_identical(_all(_dataset(disk, norm_stats)), _all(_dataset(memory, norm_stats)))


def test_windows_skip_nans_and_never_cross_sources(sources, tmp_path):
    data = _data(sources, 'memory', tmp_path)
    ds = _dataset(data, norm_stats_from(data, HISTORIC, FUTURE))
    hist, fut = data.column_indices(HISTORIC), data.column_indices(FUTURE)
    assert 2 not in ds.window_index[:, 0]  # source c is shorter than one window
    for src, start in ds.window_index:
        assert start + SEQ_LEN + HORIZON <= data.n_rows[src]
        rows = data.rows[data.offsets[src] + start:data.offsets[src] + start + SEQ_LEN + HORIZON]
        assert not np.isnan(rows[:SEQ_LEN][:, hist]).any()
        assert not np.isnan(rows[SEQ_LEN:][:, fut]).any()


def test_missing_column_raises(sources, tmp_path):
    data = _data(sources, 'memory', tmp_path)
    with pytest.raises(KeyError, match='extra'):
        data.column_indices(['extra'])  # only source b has it


def test_bundle_is_reused_not_recompiled(sources, tmp_path):
    _data(sources, 'disk', tmp_path)
    corpus = tmp_path / 'bundle' / CORPUS_FILE
    mtime = corpus.stat().st_mtime_ns
    subset = _data([sources[3], sources[1]], 'disk', tmp_path)  # a subset in another order
    assert corpus.stat().st_mtime_ns == mtime
    assert len(subset) == 2


def test_two_date_ranges_of_one_series_get_separate_entries(sources, tmp_path):
    data = _data(sources, 'disk', tmp_path)
    manifest = BundleManifest.load(tmp_path / 'bundle')
    assert len(manifest.entries) == 4
    assert sources[0].group_key == sources[3].group_key
    assert data.offsets[0] != data.offsets[3]


def test_bundle_mismatches_raise(sources, tmp_path):
    _data(sources[:2], 'disk', tmp_path)
    with pytest.raises(ValueError, match='cannot be reused'):
        _data(sources[:3], 'disk', tmp_path)  # source c was never compiled in

    _write_csv(sources[1].csv, 70, 99, extra=True)  # file changes after compile
    with pytest.raises(ValueError, match='changed'):
        _data(sources[:2], 'disk', tmp_path)

    (tmp_path / 'other').mkdir()
    (tmp_path / 'other' / 'notes.txt').write_text('not a bundle')
    with pytest.raises(ValueError, match='not empty'):
        _data(sources, 'disk', tmp_path, name='other')


def test_mode_and_path_validation(sources, tmp_path):
    with pytest.raises(ValueError):
        TimeSeriesData(sources, mode='disk')
    with pytest.raises(ValueError):
        TimeSeriesData(sources, mode='memory', path=tmp_path / 'x')
    with pytest.raises(ValueError):
        TimeSeriesData(sources, mode='ram')


def test_disk_dataset_pickles_without_the_data(sources, tmp_path):
    data = _data(sources, 'disk', tmp_path)
    ds = _dataset(data, norm_stats_from(data, HISTORIC, FUTURE))
    _ = _all(ds)  # opens the memory map in this process
    blob = pickle.dumps(ds)
    assert len(blob) < data.rows.nbytes
    _assert_batches_identical(_all(pickle.loads(blob)), _all(ds))


@pytest.mark.parametrize('mode', ['memory', 'disk'])
def test_workers_return_the_same_batches(sources, tmp_path, mode):
    data = _data(sources, mode, tmp_path)
    ds = _dataset(data, norm_stats_from(data, HISTORIC, FUTURE))
    single = list(_make_loader(ds, 16, shuffle=False, device='cpu'))
    workers = list(_make_loader(ds, 16, shuffle=False, device='cpu', num_workers=2))
    assert len(single) == len(workers)
    for a, b in zip(single, workers):
        _assert_batches_identical(a, b)


def test_cpu_prefetcher_passes_batches_through(sources, tmp_path):
    data = _data(sources, 'memory', tmp_path)
    ds = _dataset(data, norm_stats_from(data, HISTORIC, FUTURE))
    loader = _make_loader(ds, 16, shuffle=False, device='cpu')
    for a, b in zip(_make_prefetcher(loader, 'cpu'), loader):
        _assert_batches_identical(a, b)


def _fit(training_data, validation_data, tmp_path, name, **kwargs):
    val_mae = MAE()
    val_mae.eval_on_normalized = True
    fc = Forecaster(
        model=LSTMHistoric(hidden_size=8, num_layers=1, dropout_rate=0.0), name=name,
        loss=MAE(reweight=True), validation_logging_criteria=[val_mae],
        training_data=training_data, validation_data=validation_data,
        historic_cols=HISTORIC, future_cols=[], target_col='y',
        forecasting_horizon=HORIZON, historic_input_sequence_length=SEQ_LEN,
        number_of_epochs=2, batch_size=16, save_path=str(tmp_path), device=torch.device('cpu'),
        **kwargs,
    )
    fc.fit()
    return fc


def test_end_to_end_both_modes_agree(sources, tmp_path):
    train, val = sources[:2], [sources[3], sources[1]]
    memory = _fit(_data(train, 'memory', tmp_path), _data(val, 'memory', tmp_path), tmp_path, 'memory')
    disk = _fit(_data(train, 'disk', tmp_path, 'train'), _data(val, 'disk', tmp_path, 'val'), tmp_path, 'disk')

    assert np.all(np.isfinite(memory.loss_log))
    assert memory.val_log.shape == (2, 2)  # one row per validation source
    assert np.all(np.isfinite(rfn.structured_to_unstructured(memory.val_log)))
    np.testing.assert_array_equal(disk.loss_log, memory.loss_log)
    np.testing.assert_array_equal(rfn.structured_to_unstructured(disk.val_log),
                                  rfn.structured_to_unstructured(memory.val_log))

    # pinned, pre-existing behaviour: with reweight=True the validation 'loss' is the
    # plain pooled MAE, not the weighted training loss
    np.testing.assert_array_equal(memory.val_log['loss'], memory.val_log['MAE'])

    preds = disk.predict(_data(val, 'memory', tmp_path))
    assert preds.shape[1:] == (HORIZON, 1) and np.all(np.isfinite(preds))


def test_end_to_end_with_workers(sources, tmp_path):
    train, val = sources[:2], sources[3:]
    fc = _fit(_data(train, 'disk', tmp_path), _data(val, 'memory', tmp_path), tmp_path, 'workers',
              num_workers=2, validation_num_workers=1)
    assert np.all(np.isfinite(fc.loss_log))
    assert np.all(np.isfinite(fc.val_log['loss']))
