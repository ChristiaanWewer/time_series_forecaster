# Codebase Overview

## Structure

```
src/
  data/
    __init__.py              — exports DataSource
    datasource.py            — DataSource dataclass
    timeseries_dataset.py    — TimeSeriesDataset + _collate_fn
    normalization.py         — compute_norm_stats, _read_1d_df helper
  forecaster/
    forecaster.py            — training loop, normalization, logging, checkpointing
  models/
    lstm_historic.py         — LSTMHistoric
    lstm_encoder_decoder.py  — LSTMEncoderDecoder
  utils/
    scores_and_losses.py     — metric classes + resolve_metric
    DILATE/
      soft_dtw.py            — Numba-JIT soft-DTW forward/backward DP + SoftDTWBatch autograd.Function
      path_soft_dtw.py       — Numba-JIT Hessian-vector-product DP + PathDTWBatch autograd.Function
      loss_utils.py          — Numba softmin/Hessian-product helpers used by path_soft_dtw.py
      dilate_loss.py         — DILATE loss combining shape + temporal terms
      __init__.py
```

---

## `src/data/datasource.py` — `DataSource`

Plain dataclass describing one dataset (one location, one time window, one or two file sources).

| Field | Type | Description |
|---|---|---|
| `start` | `str \| None` | Start date, e.g. `'2000-01-01'`; `None` = no lower bound (whole file) |
| `end` | `str \| None` | End date, e.g. `'2014-12-31'`; `None` = no upper bound (whole file) |
| `csv` | `str \| None` | Path to CSV file (1D tabular) |
| `netcdf_1d` | `str \| None` | Path to NetCDF file for 1D variables |
| `netcdf_1d_vars` | `list` | Variable names to read from `netcdf_1d` (required if `netcdf_1d` set) |
| `csv_index_col` | `int` | Column index to use as DataFrame index when reading CSV (default `0`) |
| `binary_cols` | `list` | Columns to skip normalization for; empty = auto-detect from values |
| `nodata_values` | `list` | Sentinel values (e.g. `-999`) treated as missing, across all columns — converted to `NaN` in `_read_1d_df` |

Validation in `__post_init__`: at least one of `csv`/`netcdf_1d` required; `netcdf_1d_vars` required when `netcdf_1d` set.

No `target_col` field — that's owned by `Forecaster` now (see below), not per-source. For a netcdf source, note that `netcdf_1d_vars` must include the target variable's name too if it's meant to be read from that file for training — `_read_1d_df` subsets the netcdf dataset down to exactly `netcdf_1d_vars` before the target lookup ever happens, unlike the CSV path which reads every column unconditionally.

---

## `src/data/normalization.py`

**`_read_1d_df(source)`** — loads a CSV and/or NetCDF 1D file for a `DataSource`, slices by `start`/`end` (skipped when either is `None`, using the file's full range), and returns a merged `DataFrame` (or `None` if no 1D source). Any `source.nodata_values` are replaced with `NaN` before returning.

**`_merge_stats(n_a, mean_a, M2_a, n_b, mean_b, M2_b)`** — Chan's parallel variance algorithm. Merges two `(n, mean, M2)` accumulators into one without storing raw values. Exact — no approximation.

**`compute_norm_stats(sources, historic_cols, future_cols, target_col='y')`** — computes per-variable mean/std using Chan's algorithm: iterates one source at a time, accumulating `(n, mean, M2)` per column, never holding more than one source's data in memory. Target stats are accumulated in the same pass, using `target_col` (not read off `DataSource` — see `Forecaster` below); if `target_col` is `None` or absent from a source's columns, target stats are skipped for that source.

Called on **training sources only**. The resulting stats are then applied to both training and validation datasets.

Binary column handling: a column is skipped (mean=0, std=1) if it appears in `source.binary_cols` OR if auto-detection finds all values in `{0, 1}`. Skipped columns are recorded in `stats['binary_cols']`.

Returns a dict with keys:
- `xh_mean`, `xh_std` — shape `(n_historic_features,)`
- `xf_mean`, `xf_std` — shape `(n_future_features,)`
- `target_mean`, `target_std` — scalars
- `binary_cols` — `set[str]` of column names that were not normalized

---

## `src/data/timeseries_dataset.py` — `TimeSeriesDataset`

Lazy `torch.utils.data.Dataset`. Pre-loads 1D data into memory.

**Constructor** — `sources, seq_len, horizon, historic_cols, future_cols, norm_stats, target_col='y', dtype`

At construction:
1. Loads 1D DataFrames into memory per source.
2. Builds a flat index `[(source_idx, t)]` of all valid windows, skipping any window with NaN in `x_h`/`x_f`.

**`__getitem__(i)`** — returns a `dict`:

| Key | Shape | Condition |
|---|---|---|
| `x_h` | `(seq_len, n_hist)` | always |
| `x_f` | `(horizon, n_fut)` | always |
| `y` | `(horizon, 1)` | only if `target_col is not None` |

All values are normalized using `norm_stats`. `y` NaN values are preserved for loss masking. `target_col=None` is how `Forecaster.predict` builds its dataset — no target column needed on a file used purely for inference. `_load_1d` looks up `df[[target_col]]` with no existence guard, so a wrong or missing column name raises a plain `KeyError` rather than a custom error.

**`_collate_fn(batch)`** — custom collate that stacks tensors for the intersection of keys across all items in the batch. Import from `src.data.timeseries_dataset`.

---

## `src/forecaster/forecaster.py` — `Forecaster`

Training orchestrator. Takes `DataSource` lists, handles normalization, training, validation, checkpointing, and inference.

**Key constructor parameters:**

| Parameter | Description |
|---|---|
| `training_datasets` | `list[DataSource]` — all combined into one `TimeSeriesDataset` |
| `validation_datasets` | `list[DataSource]` — each becomes a separate `TimeSeriesDataset` / DataLoader |
| `target_col` | `str \| None`, default `'y'` — column name for the target across all sources. Lives here, not on `DataSource` (one Forecaster = one target). Saved/restored in the checkpoint. |
| `num_workers` | DataLoader worker processes (default `0`) |
| `prefetch_factor` | Prefetch factor; applied only when `num_workers > 0` (default `2`) |
| `pin_memory` | DataLoader pin memory (default `False`) |
| `drop_last` | Drop last incomplete batch (default `False`) |
| `shuffle` | Shuffle training loader; validation always `False` (default `True`) |
| `norm_stats` | `dict \| None` — reuse pre-computed stats instead of calling `compute_norm_stats` (used internally by `load_model` on resume, so datasets stay consistent with what the restored model/optimizer were trained against) |
| `optimizer` | `str`, default `'adam'` — spec string resolved by `_resolve_optimizer` via `scores_and_losses.parse_spec` (same `'name'` / `'name(kw=val, ...)'` grammar `resolve_metric` uses for losses, e.g. `'DILATE(alpha=0.5, gamma=0.01)'`). Looked up case-insensitively against every `torch.optim.Optimizer` subclass (`_OPTIMIZER_REGISTRY`, built from `vars(torch.optim)`) — not `getattr(torch.optim, optimizer.capitalize())` (the old approach silently broke for any name that isn't capitalize()-shaped, e.g. `'sgd'.capitalize()` → `'Sgd'`, but the real class is `SGD`). Any optimizer kwarg goes in the string — `'adamw(weight_decay=0.01)'`, `'sgd(momentum=0.9, weight_decay=1e-4)'`. A `lr` kwarg inside the string overrides the separate constructor's resolved `learning_rate`. |
| `l1_lambda` | `float`, default `0.0` — L1 penalty coefficient added to the training loss (no PyTorch optimizer supports L1 natively, unlike L2/`weight_decay`). `0.0` is a true no-op — no parameter traversal happens. |

**Weight decay / L1 apply to weights only, never biases.** At construction, `self.model.parameters()` is split once into `decay_params` (`p.dim() > 1` — every `nn.Linear.weight`, `nn.LSTM`'s `weight_ih_*`/`weight_hh_*`) and `no_decay_params` (`p.dim() <= 1` — every `*.bias`). The optimizer is built from two `param_groups`, the bias group pinned to `weight_decay=0.0` regardless of what the `optimizer` spec string sets for the other group. `self._l1_params` (= `decay_params`) is reused for the L1 term in `__training_step`, so both regularizers exclude biases via the same split. This changes `self.optimizer.param_groups` from 1 group to 2 relative to checkpoints saved before this — see `load_model`'s resume behavior below.

`__training_step` adds `l1_lambda * sum(p.abs().sum() for p in self._l1_params)` to the loss before `backward()` when `l1_lambda` is truthy, so `self.loss_log` reflects the regularized objective. `__evaluate_model` is a fully separate code path (never calls `__training_step`) and never adds this term — every validation metric, including `validation_logging_criteria[0]` (the bare loss function), always reflects the un-regularized value.

**Init flow:**
1. `compute_norm_stats(training sources only)` → `norm_stats`, unless `norm_stats` was passed in directly (train stats applied to val either way)
2. `TimeSeriesDataset(training_datasets, ..., norm_stats)` → training dataset
3. One `TimeSeriesDataset([src], ..., norm_stats)` per validation source → validation datasets
4. Wrap each in `DataLoader` with supplied settings

**Training / eval loops:** batches are dicts. `y = batch['y']`; `X = {k: v for k, v in batch.items() if k != 'y'}`. Both `X` and `y` are moved to `self.device` before the forward pass. NaN filtering on `y` unchanged. The model receives the full `X` dict — models ignore unknown keys.

`__evaluate_model` computes each of `validation_logging_criteria` one of two ways per validation set (see "Pooled (exact) validation metrics" above): poolable metrics (`getattr(m, 'poolable', False)`) accumulate exact statistics across all of that validation set's batches via `new_accumulators()`/`accumulate()`, finalized once via `pooled_value()`; everything else keeps the original per-batch-average (`log[idx] += m(...)`, divided by batch count at the end). Accumulators are built fresh per validation set per epoch — pooling happens *within* one `DataSource`'s own batches only, never mixing two validation sets together; cross-validation-set combination still happens afterward via `val_log_aggregated`'s mean/max/min/median across validation sets, unchanged. Index 0 of `validation_logging_criteria` (the loss function) always sees normalized values (matching `__training_step`); the rest see denormalized values unless `eval_on_normalized`. `__evaluation_step` no longer exists — this logic is inline in `__evaluate_model` now, since accumulation is a side effect and doesn't fit a "return a stacked tensor" shape.

**Logging:** `self.val_log` / `self.val_log_aggregated['mean'|'max'|'min'|'median']` are **structured numpy arrays** — named fields instead of positional indices, e.g. `val_log_aggregated['mean'][epoch]['val_score']`. Field names come from `_criteria_field_names`: `'loss'` and `'val_score'` are fixed (first two slots), remaining criteria are named by class (`'MAE'`, `'NSE'`, ...), deduplicated with a numeric suffix (`'MAE_2'`) if the same metric type appears more than once. `self._criteria_names` holds the field name list. `self.loss_log` (training loss only, one value per epoch) stays a plain float array.

**Checkpointing:** `fit()` saves a bundled dict (via `self._checkpoint_dict(epoch)`), not a bare state dict — `torch.save({...}, path)` with keys: `state_dict`, `optimizer_state_dict`, `epoch`, `best_logged_criterion`, `norm_stats`, `model_config`, `historic_cols`, `future_cols`, `target_col`, `forecasting_horizon`, `historic_input_sequence_length`, `loss_function_spec`, `validation_score_spec`, `validation_logging_criteria_specs`, `optimizer_spec`, `l1_lambda`, `minimize_validation_score`, `save_aggregation_criterion`, `criteria_names`, `loss_log`, `val_log`, `val_log_aggregated`. Metrics are saved as reconstructable spec strings (`metric_to_spec`, e.g. `'DILATE(alpha=0.5, gamma=0.01)'`) rather than pickled objects, so metric configuration survives across processes/machines without pickling `nn.Module`s. `load_weights(path=None)` reloads just `ckpt['state_dict']` into the current in-memory `Forecaster` (same process, e.g. reverting to the best epoch after training).

`optimizer_spec`/`l1_lambda` follow the same "auto-restored on resume" philosophy as the loss/score specs — but unlike `weight_decay` (an optimizer hyperparameter, so it rides along inside `optimizer_state_dict`'s `param_groups` for free once the optimizer *type* is right), the optimizer's *type* itself has no such free lunch: `load_state_dict` needs matching internal state structure (e.g. Adam's `exp_avg`/`exp_avg_sq` vs SGD's `momentum_buffer`), so `optimizer_spec` is what lets resume rebuild the correct class before that call. No backward compatibility with checkpoints saved before these fields existed — `resume=True` on one of those raises `KeyError` (and separately, would raise on `optimizer.load_state_dict` regardless, since the weight/bias param-group split also changed the group count from 1 to 2).

**`Forecaster.load_model(checkpoint_path, model, resume, training_datasets=None, validation_datasets=None, device='cpu', batch_size=512, num_workers=0, pin_memory=False, use_torch_compile=False, learning_rate=None, **forecaster_kwargs)`** — classmethod, three modes:
- `training_datasets=None` → **inference only**. Builds just the model from `ckpt['model_config']`, loads weights, carries over `norm_stats`/schema. No optimizer, loaders, or logs — built via `cls.__new__(cls)`, bypassing `__init__`, so calling `.fit()` on the result raises `AttributeError` by design.
- `resume=True` (requires `training_datasets`) → **continues the same run**. Restores weights, optimizer state, `norm_stats`, the loss/score/aggregation config, `optimizer_spec`/`l1_lambda`, `best_logged_criterion`, and log history — all pulled from the checkpoint automatically, never re-specified, so a resumed run can't silently diverge from the original. `fc._start_epoch = checkpoint['epoch'] + 1`; raises `ValueError` if `number_of_epochs <= start_epoch`. `learning_rate`, if given, overwrites just the restored optimizer's LR (keeps momentum/Adam moment state) — e.g. for decaying LR on a fine-tuning continuation.
- `resume=False` + `training_datasets` given → **warm start**. A fresh `Forecaster` (fresh optimizer, fresh `norm_stats` computed from whatever `training_datasets` are passed now, fresh logs at epoch 0), with only `state_dict` loaded as an initialization. `model_config`/`historic_cols`/etc. must be supplied via `forecaster_kwargs` here, same as a plain `Forecaster(...)` call.

`use_torch_compile` is applied *after* weights are loaded (matching `load_weights`'s existing behavior) — compiling is a runtime choice for wherever you're loading, not a fact saved in the checkpoint.

**`predict(x_test, batch_size=None, denormalize=True, device=None, num_workers=0, pin_memory=False)`** — `x_test` is a `DataSource` or `list[DataSource]`, nothing else (mirrors exactly what `training_datasets`/`validation_datasets` accept — no `TimeSeriesDataset`/`TensorDataset` support, no type validation; anything else just crashes naturally). Builds a `TimeSeriesDataset` internally using `self.historic_cols`/`self.future_cols`/`self.historic_input_sequence_length`/`self.forecasting_horizon`/`self.norm_stats`, always with `target_col=None` — `predict` never reads `y`, so no target column is required on the source, even if it wasn't needed at training time either. `device`/`num_workers`/`pin_memory` are independent of whatever training used (train and inference are commonly on different machines); `device=None` defaults to `self.device` and moves the model there if it differs.

---

## `src/models/lstm_historic.py`, `src/models/lstm_encoder_decoder.py`

Both models receive a single `config` dict. The Forecaster injects `historic_cols`, `future_cols`, `forecasting_horizon`, and `historic_input_sequence_length` into this dict automatically.

`X` is an **open dict** — models read only the keys they need and silently ignore the rest. Existing models use `X['x_h']` and `X['x_f']`.

**`LSTMHistoric`** — encodes only `x_h`; single LSTM → last hidden → linear → `(B, horizon, 1)`.

**`LSTMEncoderDecoder`** — encoder LSTM on `x_h`; hidden downscaled and repeated over horizon, concatenated with `x_f`; decoder LSTM → linear → `(B, horizon, 1)`.

---

## `src/utils/scores_and_losses.py`

**`parse_spec(spec)`** — parses `'name'` / `'name(kw=val, ...)'` into `(name.lower(), kwargs)`, kwargs values via `ast.literal_eval`. The shared grammar underneath `resolve_metric`; also imported by `Forecaster`'s `_resolve_optimizer` (see below) so the `optimizer='adamw(weight_decay=0.01)'` spec string uses the exact same parsing, not a second copy of it.

**`resolve_metric(metric)`** — string or instance → metric object. String forms: `'mae'` or `'DILATE(alpha=0.5, gamma=0.01)'`.

**`metric_to_spec(metric)`** — inverse of `resolve_metric`: instance → string. Used by `Forecaster` to make metric configuration checkpointable without pickling the live `nn.Module`.

**`assert_differentiable(metric)`** — raises `TypeError` if metric has `differentiable = False`.

**Metric classes:** `MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `DILATE`, `NSE`, `AlphaNSE`, `BetaNSE` — all wrap `nn.Module`. `DILATE` has `eval_on_normalized = True` (soft-DTW overflows on large denormalized values). `NSE` (Nash-Sutcliffe Efficiency), `AlphaNSE` (variability ratio, the α term from the Gupta et al. 2009 NSE/KGE decomposition — matches `neuralhydrology`'s `alpha_nse`, `sim.std()/obs.std()`, verified against its source) and `BetaNSE` (bias term, same decomposition — matches `neuralhydrology`'s `beta_nse`, `(mean(sim)-mean(obs))/std(obs)`) are not lower-is-better losses and set `differentiable = False` so `assert_differentiable` blocks them from being used as `loss_function`. `AlphaNSE`/`BetaNSE` use population variance (`unbiased=False`, ddof=0 — matching numpy/xarray's default, and hence the reference) rather than `torch.std()`'s default ddof=1; for `AlphaNSE` this is a ratio of two same-`n` variances so the correction would cancel either way, but for `BetaNSE` (no matching denominator term) the convention actually changes the numeric result.

`DILATE(alpha=0.5, gamma=0.01)` — no `use_torch_compile` parameter; always eager (see below).

**Pooled (exact) validation metrics.** `forward(y_pred, y_true)` on every metric class still computes on whatever tensor you pass it (unchanged) — used directly by things like `check.ipynb`'s "NSE vs. lead time" cell, which calls it once on the full concatenated test-set array. But naively averaging *per-batch* `forward()` values, which is what `Forecaster`'s evaluation loop used to do, is not the same as computing a metric once over the whole pooled validation set for anything that isn't a plain per-sample mean: NSE/AlphaNSE/BetaNSE are ratios whose denominator depends on the batch's own local mean/variance (a low-variance batch — e.g. a baseflow-only window — can swing its per-batch NSE wildly, then get averaged in with equal weight regardless of size); RMSE's `sqrt` doesn't commute with averaging; and even MAE/MAPE/SMAPE (plain means) are only exactly right under batch-averaging if every batch is the same size, which the last batch usually isn't under `drop_last=False`.

Fixed via an opt-in pooled API on `_Metric`: `poolable: bool = False` (default), and for poolable metrics, `new_accumulators()` (fresh named `ChanAccumulator`s), `accumulate(accs, y_pred, y_true)` (update them with one batch), `pooled_value(accs)` (the exact epoch-level value). `ChanAccumulator` is a small `(n, mean, M2)` streaming accumulator built on `_merge_stats` (imported from `src/data/normalization.py` — the same Chan's-parallel-algorithm merge already used there for feature normalization stats, reused rather than reimplemented). `MAE`/`MSE`/`RMSE`/`MAPE`/`SMAPE` each accumulate one stream (their own elementwise per-sample value; `RMSE` takes `sqrt` of the pooled mean at the end). `NSE` accumulates two (`residual_sq`, `obs`); `AlphaNSE`/`BetaNSE` accumulate two each (`pred`, `obs`). No cross-metric accumulator sharing — each metric owns its own, computed independently even if two requested metrics need the same underlying stream (negligible cost next to the model forward pass). `DILATE` stays `poolable = False` (not a reduction of a simple per-sample stream).

---

## `src/utils/DILATE/`

Ported from `marcdemers/batch-DILATE` (a batched fork of the original `vincent-leguen/DILATE` reference implementation), replacing an earlier pure-PyTorch version that had two problems: (1) it looped over the DP grid cell-by-cell in Python (~49 sequential ops per batch for `horizon=7`), and (2) it computed the temporal-alignment term via `.detach()`ed tensors under `@torch.no_grad()`, so `loss_temporal` never actually contributed gradients — DILATE was silently training on the shape term alone. Both are fixed by this port; correctness was verified against the real upstream package (loss values and gradients matched to ~3e-5).

- **`loss_utils.py`** — `numba_min`/`numba_max_hessian_product`/etc., `@njit` helpers for the softmin and Hessian-vector-product math. `numba_apply_along_axis` deliberately omits `cache=True` — it takes a function (`np.max`/`np.sum`) as an argument, which Numba can't pickle for its on-disk cache on this Numba version; everything downstream of it keeps `cache=True`.
- **`soft_dtw.py`** — `compute_softdtw_batch_channel`/`compute_softdtw_backward_batch_channel` (`@njit`, vectorized over a `(batch, channel)` leading pair — one Numba call per forward/backward, not one per sample) and `SoftDTWBatch(torch.autograd.Function)`, whose `forward`/`backward` round-trip through `.cpu().numpy()` to call them and hold the shape (soft-DTW) loss.
- **`path_soft_dtw.py`** — `dtw_grad`/`dtw_hessian_prod` (`@njit`) and `PathDTWBatch(torch.autograd.Function)`, which produces the temporal-alignment path via a hand-derived Hessian-vector-product backward — this is what makes `loss_temporal` actually differentiable.
- **`dilate_loss.py`** — glue: computes `D` (our own direct squared-difference formula, not the reference's norm-expansion/`.double()` variant), adds a dummy `channel=1` axis to match the ported kernels' `(batch, channel, N, N)` shape, calls `SoftDTWBatch.apply`/`PathDTWBatch.apply`, returns `(loss, loss_shape, loss_temporal)` — same signature as before the port.

**No `torch.compile` support, intentionally.** `torch.compile` cannot safely trace into `SoftDTWBatch`/`PathDTWBatch` — Dynamo tries to trace their `.cpu().numpy()`/Numba internals and crashes (a `CppCompileError` under the old pure-PyTorch implementation's giant unrolled loop; a Dynamo fake-tensor dtype error under this Numba-based one). `torch._dynamo.disable`/`allow_in_graph` were tried and don't fix it either (the former doesn't intercept when the disabled function is itself the direct `torch.compile()` target; the latter's own docs warn it's the wrong tool for black-boxing a function through Inductor — a proper fix would mean rewriting `SoftDTWBatch`/`PathDTWBatch` as `torch.library.custom_op`s, out of scope). Since Numba JIT already makes DILATE fast in eager mode, `use_torch_compile` was removed entirely from `DILATE.__init__` and from `Forecaster.__init__`'s compile-sharing block — `Forecaster(use_torch_compile=True)` now only ever compiles the model, never the loss, so it no longer crashes when combined with a DILATE loss.

---

## `check.ipynb`

Reference notebook demonstrating a full training run using the `DataSource` API. Trains and compares **four models** — the two architectures (`LSTMHistoric`, `LSTMEncoderDecoder`; 16 hidden, 1 layer, downscale 8 for the latter) each with two loss functions (`'mae'`, `'dilate'`) — all sharing the same `historic_cols`/`target_col`/horizon/seq_len/batch_size(256)/epochs(50):

| Var | Model | Loss | Checkpoint name |
|---|---|---|---|
| `fc`  | `LSTMHistoric`        | `mae`    | `LSTMHistoric_test` |
| `fc2` | `LSTMEncoderDecoder`  | `mae`    | `LSTMEncoderDecoder_test` |
| `fc3` | `LSTMHistoric`        | `dilate` | `LSTMHistoric_dilate_test` |
| `fc4` | `LSTMEncoderDecoder`  | `dilate` | `LSTMEncoderDecoder_dilate_test` |

1. Create `DataSource` descriptors for train/val/test time windows pointing to `data/data.csv`. All three set `nodata_values=[-999]` — the target column has 84 `-999` sentinel rows (all in the 2014 tail of `test_source`), converted to `NaN`.
2. Train each of the four `Forecaster`s in turn (`validation_score='nse'`, `validation_logging_criteria=['rmse']`, `minimize_validation_score=False`) — `loss_function='dilate'` resolves to `DILATE(alpha=0.5, gamma=0.01)` via `resolve_metric` (string form also accepts kwargs, e.g. `'dilate(alpha=0.3, gamma=0.05)'`). `Forecaster(use_torch_compile=False)` here compiles nothing either way — DILATE never supports compilation (see `src/utils/DILATE/` above), so the flag only ever affects the model.
3. **Evaluation on `test_source`, for all four models** (needs `matplotlib`): after each `fc.fit()`, predictions come from `fc.predict(test_source)` directly. True values/dates come from a small separate `TimeSeriesDataset` per model, built with `target_col=TARGET_COL` purely to read off `test_ds._index`'s `(source_idx, t)` pairs for window/date alignment — not to run the model (`target_col` doesn't affect which windows get built, only `x_h`/`x_f` NaN does, so this covers the exact same windows `predict()`'s internal dataset does). Since all four models share `historic_cols`/`future_cols`/`target_col`, their four `t_values`/`y_true` arrays end up identical in practice — kept separate per model for consistency/clarity rather than deduplicated.
4. **Plotting, all four models overlaid** in every panel:
   - Training diagnostics: four panels — train/val loss for the two `mae`-loss models, train/val loss for the two `dilate`-loss models (split into separate panels since the two loss functions aren't on comparable scales), val_score (NSE) for all four, val RMSE for all four.
   - Hydrograph: observed vs. predicted, lead times 1/4/7 days — color encodes lead time, linestyle encodes model.
   - Scatter plot: observed vs. predicted per lead (1/4/7d) with a 1:1 reference line, one color per model.
   - Flow duration curve: both series sorted descending vs. exceedance probability, log-y, one color per model.
   - MAE and NSE vs. lead time (1–7 days) — NSE via the actual `NSE` class from `scores_and_losses`, with an explicit NaN-mask applied first since metric classes don't skip NaN the way `np.nanmean` does.
