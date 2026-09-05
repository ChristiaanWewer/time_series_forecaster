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
    heads.py                 — PooledHead, SequenceHead (output projections, shared across models)
  utils/
    scores_and_losses.py     — metric classes + resolve_metric
    regularization.py        — pluggable weight regularizers (L1, RLN) + resolve_regularization
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
| `name` | `str \| None` | Stable identity for this series (e.g. a station/catchment id), used to key per-series statistics — see `group_key` below. `None` = fall back to the file path. |
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

**`group_key` property** — `self.name or self.csv or self.netcdf_1d`. The identity used to key `compute_norm_stats`' `group_target_std` and `Forecaster`'s per-series lookup for group-reweighted losses (see both below). Falling back to the already-required file path when `name` is unset means multiple `DataSource` entries for the same underlying series (e.g. separate train/val date-range windows on the same file) share one key without needing `name` set explicitly, and existing single-series usage is unaffected. Deliberately *not* keyed by list position — a checkpoint resume that passes `training_datasets` in a different order still lines samples up with the right series.

---

## `src/data/normalization.py`

**`_read_1d_df(source)`** — loads a CSV and/or NetCDF 1D file for a `DataSource`, slices by `start`/`end` (skipped when either is `None`, using the file's full range), and returns a merged `DataFrame` (or `None` if no 1D source). Any `source.nodata_values` are replaced with `NaN` before returning.

**`_merge_stats(n_a, mean_a, M2_a, n_b, mean_b, M2_b)`** — Chan's parallel variance algorithm. Merges two `(n, mean, M2)` accumulators into one without storing raw values. Exact — no approximation.

**`compute_norm_stats(sources, historic_cols, future_cols, target_col='y')`** — computes per-variable mean/std using Chan's algorithm: iterates one source at a time, accumulating `(n, mean, M2)` per column, never holding more than one source's data in memory. Target stats are accumulated in the same pass, using `target_col` (not read off `DataSource` — see `Forecaster` below); if `target_col` is `None` or absent from a source's columns, target stats are skipped for that source. Alongside the single pooled `target_accum` (across every source), a second `group_accum: dict[group_key, (n, mean, M2)]` is merged per `source.group_key` in the same loop iteration — same `_merge_stats` call, just keyed by series identity instead of pooled into one bucket, so multiple sources sharing a `group_key` (e.g. separate date-range windows of the same series) correctly combine into one entry.

Called on **training sources only**. The resulting stats are then applied to both training and validation datasets.

Binary column handling: a column is skipped (mean=0, std=1) if it appears in `source.binary_cols` OR if auto-detection finds all values in `{0, 1}`. Skipped columns are recorded in `stats['binary_cols']`.

Returns a dict with keys:
- `xh_mean`, `xh_std` — shape `(n_historic_features,)`
- `xf_mean`, `xf_std` — shape `(n_future_features,)`
- `target_mean`, `target_std` — scalars (pooled across every source)
- `group_target_std` — `dict[group_key, float]`, one raw (pre-normalization) target std per distinct series identity — used by `Forecaster` to bind group-reweighted losses (`MAE`/`MSE`/`QuantileLoss`/`ExpectileLoss` with `reweight=True`, see `scores_and_losses.py` below); fixes multi-series training being dominated by whichever series has the largest-scale target, without needing stratified/balanced batches — see that section for the mechanism.
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
| `group_idx` | scalar, `long` | always — the sample's position (`source_idx`) in the `sources` list this `TimeSeriesDataset` was built from |
| `y` | `(horizon, 1)` | only if `target_col is not None` |

All values are normalized using `norm_stats`. `y` NaN values are preserved for loss masking. `target_col=None` is how `Forecaster.predict` builds its dataset — no target column needed on a file used purely for inference. `_load_1d` looks up `df[[target_col]]` with no existence guard, so a wrong or missing column name raises a plain `KeyError` rather than a custom error.

`group_idx` is unconditional (present even when `reweight` is never used — harmless extra key `Forecaster` excludes before the model sees `X`) and is meaningful only *within* one `TimeSeriesDataset`/`Forecaster` instance: since `Forecaster` always builds `TimeSeriesDataset(sources=training_datasets, ...)` from the exact same list it holds, position *i* here matches `training_datasets[i]` for that run. It is not itself persisted across checkpoint resumes — `Forecaster` re-derives the array group-reweighted losses actually bind against fresh, every run, by looking up each of *this* run's `training_datasets` by their stable `group_key` (see `DataSource.group_key` above) in the checkpoint's saved `group_target_std`, so a reordered `training_datasets` on resume still lines up correctly.

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
| `regularization` | `str \| None`, default `None` — spec string resolved by `regularization.resolve_regularization` (same `parse_spec` grammar as `optimizer`/losses), e.g. `'l1(lambda_=0.01)'` or `'rln(theta=-4.0, lr=0.01)'`. See `src/utils/regularization.py` below. |

**Output head sizing.** Right after resolving `loss_function`, `Forecaster` reads `output_kind`/`output_levels`/`n_outputs` straight off it (`getattr(..., default)`, so plain losses like `'mae'` are unaffected — `output_kind='point'`, `n_outputs=1`) and stores them as `self.output_kind`/`self.output_levels`/`self.n_outputs`. All three get injected into `self.model_config` (next to `historic_cols`, `forecasting_horizon`, etc.) before `self.model = model(self.model_config)` — this is what lets `loss_function='quantile_loss(q=[...])'` alone drive the model's output head width (see `heads.py` above). It's also how every other resolved criterion gets bound: right after `self.validation_logging_criteria` is assembled, `Forecaster` calls `m.bind_levels(self.output_kind, self.output_levels)` on each entry, unconditionally (see the probabilistic-scores section above).

**Weight decay / regularization apply to weights only, never biases or the output head.** At construction, `self.model.named_parameters()` is split once into three groups: `output_head.*` (by name — the probabilistic output head, always exempt), `weights` (`p.dim() > 1` — every `nn.Linear.weight`, `nn.LSTM`'s `weight_ih_*`/`weight_hh_*`), and `biases` (`p.dim() <= 1` — every `*.bias`). The optimizer is built from three `param_groups`, with the bias and head groups both pinned to `weight_decay=0.0` regardless of what the `optimizer` spec string sets for the weights group. `self._regularized_weights` (= `weights`, excluding both biases and the head) is passed to `self.regularizer` for the same reason. This changes `self.optimizer.param_groups` from 2 groups to 3 relative to checkpoints saved before this was introduced — `load_model(resume=True)` on one of those now raises (no backward compatibility is maintained for pre-this-change checkpoints, same as the earlier 1→2 group change when `regularization=` replaced `l1_lambda`).

`__training_step` calls `self.regularizer.step(self._regularized_weights)` right after `loss.backward()` (so it sees the plain data gradient in `p.grad`) and before `optimizer.step()` — regularizers mutate `p.grad` in place rather than adding anything to the loss graph, so `self.loss_log` is purely the data loss regardless of `regularization`. `__evaluate_model` is a fully separate code path (never calls `__training_step`) and never touches the regularizer either way — every validation metric, including `validation_logging_criteria[0]` (the bare loss function), always reflects the un-regularized value.

**Init flow:**
1. `compute_norm_stats(training sources only)` → `norm_stats`, unless `norm_stats` was passed in directly (train stats applied to val either way)
2. `TimeSeriesDataset(training_datasets, ..., norm_stats)` → training dataset
3. One `TimeSeriesDataset([src], ..., norm_stats)` per validation source → validation datasets
4. Wrap each in `DataLoader` with supplied settings

**Training / eval loops:** batches are dicts. `y = batch['y']`; `X = {k: v for k, v in batch.items() if k not in ('y', 'group_idx')}`. Both `X` and `y` are moved to `self.device` before the forward pass. NaN filtering on `y` unchanged. The model receives the full `X` dict — models ignore unknown keys. `group_idx` is excluded from `X` (in `__train_model`, `__evaluate_model`, and `predict`) since it identifies a series, not a model input feature.

**Group-reweighted losses (`reweight=True` on `MAE`/`MSE`/`QuantileLoss`/`ExpectileLoss`, see `scores_and_losses.py` below).** Right after `norm_stats` is resolved in `__init__`, if `getattr(loss_function, 'needs_group_idx', False)`, `Forecaster` builds `group_std = torch.tensor([norm_stats['group_target_std'][ds.group_key] for ds in training_datasets], dtype=dtype)` — one entry per training source, in the same order `TimeSeriesDataset` assigns `group_idx` positions — calls `loss_function.bind(group_std)`, then `loss_function.to(self.device)`. A `training_datasets` entry whose `group_key` isn't in the checkpoint's `norm_stats['group_target_std']` (e.g. resuming with an unrecognized series) raises a plain `KeyError` — deliberately uncaught, since silently misaligning a series with the wrong std would be worse. Entirely inert for the default (`reweight=False`) losses — `needs_group_idx` is unset (falls back to `False`) so none of this runs.

In `__train_model`, `batch['group_idx']` is sliced by the same NaN `valid` mask as `y` and threaded through `__training_step`, which calls `self.loss_function(y_pred, y, group_idx)` when `needs_group_idx` is set, else the plain 2-arg call — every other loss is unaffected. `__evaluate_model` needs no equivalent wiring: `MAE`/`MSE` (and `QuantileLoss`/`ExpectileLoss`) are all `poolable=True`, so their contribution to `val_log`/validation logging goes through `accumulate()`/`pooled_value()` (unweighted, unchanged), never through `forward()` — group-reweighting only ever affects the training objective, not validation scoring (which is already handled correctly per-series, see "Pooled (exact) validation metrics" below).

`__evaluate_model` computes each of `validation_logging_criteria` one of two ways per validation set (see "Pooled (exact) validation metrics" above): poolable metrics (`getattr(m, 'poolable', False)`) accumulate exact statistics across all of that validation set's batches via `new_accumulators()`/`accumulate()`, finalized once via `pooled_value()`; everything else keeps the original per-batch-average (`log[idx] += m(...)`, divided by batch count at the end). Accumulators are built fresh per validation set per epoch — pooling happens *within* one `DataSource`'s own batches only, never mixing two validation sets together; cross-validation-set combination still happens afterward via `val_log_aggregated`'s mean/max/min/median across validation sets, unchanged. Index 0 of `validation_logging_criteria` (the loss function) always sees normalized values (matching `__training_step`); the rest see denormalized values unless `eval_on_normalized`. `__evaluation_step` no longer exists — this logic is inline in `__evaluate_model` now, since accumulation is a side effect and doesn't fit a "return a stacked tensor" shape.

**Logging:** `self.val_log` / `self.val_log_aggregated['mean'|'max'|'min'|'median']` are **structured numpy arrays** — named fields instead of positional indices, e.g. `val_log_aggregated['mean'][epoch]['val_score']`. Field names come from `_criteria_field_names`: `'loss'` and `'val_score'` are fixed (first two slots), remaining criteria are named by class (`'MAE'`, `'NSE'`, ...), deduplicated with a numeric suffix (`'MAE_2'`) if the same metric type appears more than once. `self._criteria_names` holds the field name list. `self.loss_log` (training loss only, one value per epoch) stays a plain float array.

**Checkpointing:** `fit()` saves a bundled dict (via `self._checkpoint_dict(epoch)`), not a bare state dict — `torch.save({...}, path)` with keys: `state_dict`, `optimizer_state_dict`, `epoch`, `best_logged_criterion`, `norm_stats`, `model_config`, `historic_cols`, `future_cols`, `target_col`, `forecasting_horizon`, `historic_input_sequence_length`, `loss_function_spec`, `validation_score_spec`, `validation_logging_criteria_specs`, `optimizer_spec`, `regularization_spec`, `regularizer_state`, `minimize_validation_score`, `save_aggregation_criterion`, `criteria_names`, `loss_log`, `val_log`, `val_log_aggregated`. Metrics are saved as reconstructable spec strings (`metric_to_spec`, e.g. `'DILATE(alpha=0.5, gamma=0.01)'`) rather than pickled objects, so metric configuration survives across processes/machines without pickling `nn.Module`s. `load_weights(path=None)` reloads just `ckpt['state_dict']` into the current in-memory `Forecaster` (same process, e.g. reverting to the best epoch after training).

`optimizer_spec`/`regularization_spec` follow the same "auto-restored on resume" philosophy as the loss/score specs — but unlike `weight_decay` (an optimizer hyperparameter, so it rides along inside `optimizer_state_dict`'s `param_groups` for free once the optimizer *type* is right), the optimizer's *type* itself has no such free lunch: `load_state_dict` needs matching internal state structure (e.g. Adam's `exp_avg`/`exp_avg_sq` vs SGD's `momentum_buffer`), so `optimizer_spec` is what lets resume rebuild the correct class before that call. `regularizer_state` is a further layer beyond `regularization_spec`: reconstructing `Forecaster` from `regularization_spec` alone only rebuilds a *fresh* regularizer (e.g. RLN's `λ` re-initialized to `theta`) — `regularizer_state` (from `self.regularizer.get_state()`) is what restores the actually-*learned* state (RLN's trained per-weight `λ`), applied via `fc.regularizer.load_state(...)` right after `Forecaster.__init__` completes. No backward compatibility with checkpoints saved before these fields existed — `resume=True` on one of those raises `KeyError` (and separately, would raise on `optimizer.load_state_dict` regardless, since the weight/bias param-group split also changed the group count from 1 to 2).

**`Forecaster.load_model(checkpoint_path, model, resume, training_datasets=None, validation_datasets=None, device='cpu', batch_size=512, num_workers=0, pin_memory=False, use_torch_compile=False, learning_rate=None, **forecaster_kwargs)`** — classmethod, three modes:
- `training_datasets=None` → **inference only**. Builds just the model from `ckpt['model_config']`, loads weights, carries over `norm_stats`/schema (including `output_kind`/`output_levels`, read off `ckpt['model_config']` for symmetry — enough to construct/bind a standalone evaluation metric by hand against this Forecaster's predictions in a notebook). No optimizer, loaders, or logs — built via `cls.__new__(cls)`, bypassing `__init__`, so calling `.fit()` on the result raises `AttributeError` by design.
- `resume=True` (requires `training_datasets`) → **continues the same run**. Restores weights, optimizer state, `norm_stats`, the loss/score/aggregation config, `optimizer_spec`/`regularization_spec`/`regularizer_state`, `best_logged_criterion`, and log history — all pulled from the checkpoint automatically, never re-specified, so a resumed run can't silently diverge from the original. `fc._start_epoch = checkpoint['epoch'] + 1`; raises `ValueError` if `number_of_epochs <= start_epoch`. `learning_rate`, if given, overwrites just the restored optimizer's LR (keeps momentum/Adam moment state) — e.g. for decaying LR on a fine-tuning continuation.
- `resume=False` + `training_datasets` given → **warm start**. A fresh `Forecaster` (fresh optimizer, fresh `norm_stats` computed from whatever `training_datasets` are passed now, fresh logs at epoch 0), with only `state_dict` loaded as an initialization. `model_config`/`historic_cols`/etc. must be supplied via `forecaster_kwargs` here, same as a plain `Forecaster(...)` call.

`use_torch_compile` is applied *after* weights are loaded (matching `load_weights`'s existing behavior) — compiling is a runtime choice for wherever you're loading, not a fact saved in the checkpoint.

**`predict(x_test, batch_size=None, denormalize=True, device=None, num_workers=0, pin_memory=False, epoch=None)`** — `x_test` is a `DataSource` or `list[DataSource]`, nothing else (mirrors exactly what `training_datasets`/`validation_datasets` accept — no `TimeSeriesDataset`/`TensorDataset` support, no type validation; anything else just crashes naturally). Builds a `TimeSeriesDataset` internally using `self.historic_cols`/`self.future_cols`/`self.historic_input_sequence_length`/`self.forecasting_horizon`/`self.norm_stats`, always with `target_col=None` — `predict` never reads `y`, so no target column is required on the source, even if it wasn't needed at training time either. `device`/`num_workers`/`pin_memory` are independent of whatever training used (train and inference are commonly on different machines); `device=None` defaults to `self.device` and moves the model there if it differs. `epoch`, if given, calls `self.load_weights(path=self.save_path_epoch.format(epoch=epoch))` before predicting — the same epoch-numbered checkpoint `fit()` saves every `save_weights_every_n_epochs` epochs (1-indexed, matching its own printed `Epoch N/M` and `model_..._epoch_{N}.pt` filenames). Like `load_weights`, this is a **permanent** weight swap into `self.model`, not scoped to the one `predict()` call — a later `predict()` without `epoch=` keeps using whatever was last loaded. No existence check on the checkpoint file or `self.save_path_epoch` (unset if `Forecaster` was built with no `save_path`) — a missing epoch crashes naturally (`FileNotFoundError`/`AttributeError`), by design.

---

## `src/models/lstm_historic.py`, `src/models/lstm_encoder_decoder.py`, `src/models/heads.py`

Both models receive a single `config` dict. The Forecaster injects `historic_cols`, `future_cols`, `forecasting_horizon`, `historic_input_sequence_length`, and (see below) `n_outputs`/`output_kind`/`output_levels` into this dict automatically.

`X` is an **open dict** — models read only the keys they need and silently ignore the rest. Existing models use `X['x_h']` and `X['x_f']`.

Neither model owns its final projection directly anymore — each builds its output head via `heads.build_output_head(representation, in_features, horizon, n_outputs, output_kind, output_levels)`, always assigned to the attribute `self.output_head` (`Forecaster` keys off this exact name to exclude its params from regularization — see below). `n_outputs` defaults to `1`/`output_kind` to `'point'` via `config.get(...)`, so old configs keep working unchanged. Two backbone shapes (passed as `representation`), two raw head classes:

- **`PooledHead(in_features, horizon, n_outputs)`** (`representation='pooled'`) — for a backbone that produces one pooled vector per sample, `(B, hidden)`. `nn.Linear(hidden, horizon * n_outputs)`, reshaped to `(B, horizon, n_outputs)`.
- **`SequenceHead(in_features, n_outputs)`** (`representation='sequence'`) — for a backbone that already produces a per-timestep representation, `(B, horizon, hidden)`. Plain `nn.Linear(hidden, n_outputs)` (broadcasts over the middle dim for free).

**`LSTMHistoric`** — encodes only `x_h`; single LSTM → last hidden → output head (`'pooled'`) → `(B, horizon, n_outputs)` (its last hidden state is mapped straight to every horizon step at once — there's no per-step decoder representation to project from, hence `'pooled'` rather than `'sequence'`).

**`LSTMEncoderDecoder`** — encoder LSTM on `x_h`; hidden downscaled and repeated over horizon, concatenated with `x_f`; decoder LSTM → output head (`'sequence'`) → `(B, horizon, n_outputs)`.

**`MonotonicHead(raw_head, center_idx)`** — the standard (always-on, not opt-in) wrapper `build_output_head` applies around the raw head whenever `output_kind in ('quantile', 'expectile')`: guarantees non-crossing levels by construction instead of relying on the loss alone to discourage crossing (a known failure mode of independent per-quantile heads; cf. Cannon 2018's monotone composite quantile regression). The raw head's channel at `center_idx` (the level-`0.5` channel's position within `output_levels`, sorted ascending — `sorted(output_levels).index(0.5)`, a plain `ValueError` if `0.5` isn't among the levels, deliberately uncaught) is used directly and unconstrained as the center value — the median for a quantile head, the mean for an expectile head (see `ExpectileLoss`). Every channel above center becomes a strictly positive offset via `softplus`, cumulatively summed outward (`softplus(above).cumsum(-1)`, added to center) so each successive higher level is `>=` the one before it; channels below center are built the same way in reverse (`softplus(below.flip(-1)).cumsum(-1).flip(-1)`, subtracted from center) so each successive lower level is `<=` the one before it. `raw_head` is nested as a submodule (`self.raw_head`), so its params surface as `output_head.raw_head.*` — still matched by the `output_head.` prefix check that excludes it from regularization. A plain point head (`output_kind='point'`, `n_outputs=1`) is returned unwrapped — unaffected by any of this.

For `n_outputs=1` (the default, and every plain point loss like `'mae'`/`'dilate'`), both heads are exactly equivalent to the old hard-coded `Linear(hidden, horizon)`/`Linear(hidden, 1)` output layers. `n_outputs > 1` is how the probabilistic losses below (`quantile_loss`/`expectile_loss`) get a multi-channel output — see `Forecaster`'s model-config injection.

---

## `src/utils/scores_and_losses.py`

**`parse_spec(spec)`** — parses `'name'` / `'name(kw=val, ...)'` into `(name.lower(), kwargs)`, kwargs values via `ast.literal_eval`. The shared grammar underneath `resolve_metric`; also imported by `Forecaster`'s `_resolve_optimizer` (see below) so the `optimizer='adamw(weight_decay=0.01)'` spec string uses the exact same parsing, not a second copy of it.

**`resolve_metric(metric)`** — string or instance → metric object. String forms: `'mae'` or `'DILATE(alpha=0.5, gamma=0.01)'`.

**`metric_to_spec(metric)`** — inverse of `resolve_metric`: instance → string. Used by `Forecaster` to make metric configuration checkpointable without pickling the live `nn.Module`.

**`assert_differentiable(metric)`** — raises `TypeError` if metric has `differentiable = False`.

**Metric classes:** `MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `DILATE`, `NSE`, `AlphaNSE`, `BetaNSE` — all wrap `nn.Module`. `DILATE` has `eval_on_normalized = True` (soft-DTW overflows on large denormalized values). `NSE` (Nash-Sutcliffe Efficiency), `AlphaNSE` (variability ratio, the α term from the Gupta et al. 2009 NSE/KGE decomposition — matches `neuralhydrology`'s `alpha_nse`, `sim.std()/obs.std()`, verified against its source) and `BetaNSE` (bias term, same decomposition — matches `neuralhydrology`'s `beta_nse`, `(mean(sim)-mean(obs))/std(obs)`) are not lower-is-better losses and set `differentiable = False` so `assert_differentiable` blocks them from being used as `loss_function`. `AlphaNSE`/`BetaNSE` use population variance (`unbiased=False`, ddof=0 — matching numpy/xarray's default, and hence the reference) rather than `torch.std()`'s default ddof=1; for `AlphaNSE` this is a ratio of two same-`n` variances so the correction would cancel either way, but for `BetaNSE` (no matching denominator term) the convention actually changes the numeric result.

`DILATE(alpha=0.5, gamma=0.01)` — no `use_torch_compile` parameter; always eager (see below).

**Group-reweighting (`reweight=True` on `MAE`, `MSE`, `QuantileLoss`, `ExpectileLoss`).** Fixes multi-series training being dominated by whichever series has the largest-scale target (pooled normalization uses one global mean/std across all series, so a large-scale series' error is numerically larger in that shared space) — without needing stratified/balanced batches, so it scales to any number of series; batches stay randomly shuffled and mixed-series. Mechanically this is a per-sample error reweighting by that sample's own series' scale, not a change to how `y` is normalized/denormalized (both untouched). Verified to match `neuralhydrology`'s `MaskedNSELoss` formula exactly for the `MSE` case (same reference-implementation-checking as `AlphaNSE`/`BetaNSE` above); the `MAE`/`QuantileLoss`/`ExpectileLoss` cases are this codebase's own dimensionally-consistent extension of the same idea, not independently verified against a reference implementation.

- **`_Metric` additions:** `_reweight_power: int | None = None` (class attribute — `None` means "this metric doesn't support `reweight=True`"), `reweight`/`eps` constructor kwargs (default `False`/`0.1`), `needs_group_idx: bool = False` (instance attribute, flips to `True` only when `reweight=True`), `bind(group_std)` (registers the per-series std as a buffer — same "attach data-dependent state after construction" pattern as `RLN.bind`/`bind_levels`, since a tensor can't be expressed as a literal spec-string kwarg), and `_error(y_pred, y_true)` (abstract — the metric's own unweighted elementwise error, in its own output shape; point metrics call `_select_point` inside their `_error`, multi-channel losses like `QuantileLoss`/`ExpectileLoss` don't).
- **Dispatch is picked once, at construction, not branched on every call.** `__init__` checks whether the concrete class overrides `_error` (`type(self)._error is not _Metric._error`); if so, it assigns `self.forward = self._reweighted_forward` or `self._plain_forward` (an instance attribute shadows the class's `forward` method, so every subsequent call goes straight to the right implementation with no per-batch conditional). Classes that don't override `_error` (`RMSE`, `MAPE`, `SMAPE`, `NSE`-family, `DILATE`, `WinklerScore`, `PICP`, `CoverageGap`, `CRPS`, `SCRPS`, `RSCRPS`) are completely untouched by this — their own class-level `forward()` keeps running unmodified — and passing `reweight=True` to one of them raises `ValueError` at construction (`_reweight_power` stays `None`).
- **Exponent is tied to the error's own units, not independently configurable:** `MAE`/`QuantileLoss` (`_reweight_power=1`, weight `1/(group_std+eps)`, i.e. std) — pinball loss is piecewise-linear in the residual, same `[y]¹` character as absolute error. `MSE`/`ExpectileLoss` (`_reweight_power=2`, weight `1/(group_std+eps)²`, i.e. variance) — the expectile term is quadratic in the residual, same `[y]²` character as squared error. Mismatching the exponent to the error's power would leave the reweighted term scale-dependent again, so it's deliberately not offered as a separate parameter.
- **`CRPS` deliberately does not get this.** `differentiable=False` blocks it from ever being `loss_function`, so a `reweight` option on it would be inert — group-reweighting only matters for what's actually trained on, and validation is already handled correctly per-series without it (see "Pooled (exact) validation metrics" below). Revisit only if `CRPS` is separately promoted to a trainable loss.
- **`group_std` is looked up by `Forecaster`**, not passed through the spec string — see `Forecaster`'s "Group-reweighted losses" paragraph above. `_spec_kwargs()` on each of the four classes includes `{'reweight': True, 'eps': self.eps}` only when `self.reweight` is set, so a plain `'mae'`/`'mse'`/etc. spec (the common case) is unaffected by this addition.
- **Not scale-invariant across series if applied to a validation/scoring metric** — this mechanism fixes the *training* imbalance specifically; it isn't a general "make any metric comparable across series of different scale" tool (see the `SCRPS`/`RSCRPS` discussion below for that distinct problem, which this mechanism does not address).

**Pooled (exact) validation metrics.** `forward(y_pred, y_true)` on every metric class still computes on whatever tensor you pass it (unchanged) — used directly by things like `example_model_and_loss_comparison.ipynb`'s "NSE vs. lead time" cell, which calls it once on the full concatenated test-set array. But naively averaging *per-batch* `forward()` values, which is what `Forecaster`'s evaluation loop used to do, is not the same as computing a metric once over the whole pooled validation set for anything that isn't a plain per-sample mean: NSE/AlphaNSE/BetaNSE are ratios whose denominator depends on the batch's own local mean/variance (a low-variance batch — e.g. a baseflow-only window — can swing its per-batch NSE wildly, then get averaged in with equal weight regardless of size); RMSE's `sqrt` doesn't commute with averaging; and even MAE/MAPE/SMAPE (plain means) are only exactly right under batch-averaging if every batch is the same size, which the last batch usually isn't under `drop_last=False`.

Fixed via an opt-in pooled API on `_Metric`: `poolable: bool = False` (default), and for poolable metrics, `new_accumulators()` (fresh named `ChanAccumulator`s), `accumulate(accs, y_pred, y_true)` (update them with one batch), `pooled_value(accs)` (the exact epoch-level value). `ChanAccumulator` is a small `(n, mean, M2)` streaming accumulator built on `_merge_stats` (imported from `src/data/normalization.py` — the same Chan's-parallel-algorithm merge already used there for feature normalization stats, reused rather than reimplemented). `MAE`/`MSE`/`RMSE`/`MAPE`/`SMAPE` each accumulate one stream (their own elementwise per-sample value; `RMSE` takes `sqrt` of the pooled mean at the end). `NSE` accumulates two (`residual_sq`, `obs`); `AlphaNSE`/`BetaNSE` accumulate two each (`pred`, `obs`). No cross-metric accumulator sharing — each metric owns its own, computed independently even if two requested metrics need the same underlying stream (negligible cost next to the model forward pass). `DILATE` stays `poolable = False` (not a reduction of a simple per-sample stream).

**Probabilistic output heads (`QuantileLoss`, `ExpectileLoss`) and quantile-aware evaluation scores (`WinklerScore`, `CRPS`, `PICP`, `PINAW`).** A loss function can declare three extra `_Metric` class fields — `output_kind` (`'point'` default), `output_levels` (`None` default), `n_outputs` (`1` default) — which `Forecaster` reads straight off the *resolved loss function* to size the model's output head (see `heads.py` above) via `model_config['n_outputs']`. Every other resolved criterion (`validation_score`, `validation_logging_criteria`) then gets `bind_levels(kind, levels)` called on it once, unconditionally, right after — a no-op for metrics that never reference `self._level_index`, and how quantile-specific scores find the right channel(s).

- **`QuantileLoss(q=(0.1, 0.5, 0.9), reweight=False, eps=0.1)`** — pinball loss, vectorized over the last (quantile) dim; sets `output_kind='quantile'`, `output_levels=sorted(q)`, `n_outputs=len(q)`. Poolable (one `ChanAccumulator` per level, averaged). `reweight=True` — see "Group-reweighting" above (`_reweight_power=1`).
- **`ExpectileLoss(e=(0.1, 0.5, 0.9), reweight=False, eps=0.1)`** — asymmetric squared loss (`L_e(u) = |e - 1{u<0}| * u^2`, `u = y_true - y_pred`), same shape as `QuantileLoss` but `output_kind='expectile'`. Its level-0.5 channel is the **mean**, not the median (expectiles ≠ quantiles) — this is what drives the approximation note below. `reweight=True` — see "Group-reweighting" above (`_reweight_power=2`).
- **`WinklerScore(alpha)`**, **`PICP(alpha)`**, **`PINAW(alpha)`** — `differentiable=False` (eval-only, like `NSE`), `poolable=True`. Standard `(1-alpha)` prediction-interval convention: lower/upper bounds are read from the channels at levels `alpha/2` and `1-alpha/2` via `self._level_index` (e.g. `alpha=0.05` pairs with a `q=(0.025, 0.5, 0.975)`-trained head for a 95% PI). `WinklerScore` is the interval score (width, plus a `2/alpha`-scaled penalty for falling outside the interval); `PICP` is the fraction of observations covered (ideally ≈ `1-alpha`); `PINAW` is mean interval width normalized by the pooled range (`max-min`) of `y_true` over the whole evaluation set (plain running min/max alongside a `ChanAccumulator` for width — `ChanAccumulator` itself has no min/max).
- **`CoverageGap(alpha)`** — a thin `PICP` subclass: `PICP - (1-alpha)`, i.e. 0 = perfectly calibrated, positive = over-covered, negative = under-covered, instead of comparing raw `PICP` against a separately-plotted nominal-coverage line. Same lower/upper channel requirements as `PICP` (inherits `__init__`/`_spec_kwargs`, overrides only `spec_name` and `_elementwise`). The `(1-alpha)` shift is applied inside `_elementwise` rather than after pooling — since mean is linear, `mean(x - c) == mean(x) - c` exactly, so shifting the elementwise value before it feeds `ChanAccumulator` doesn't bias the pooled epoch-level value the way shifting an already-pooled per-batch average would have.
- **`CRPS()`** — no params. Standard quantile-decomposition approximation (Gneiting & Raftery, 2007): `2 * mean_q(pinball_q)` over *every* channel the head provides (all of `self._level_index`, not just a chosen pair).
- **`SCRPS()`** — no params, `poolable=True`. Scale-invariant CRPS (Bolin & Wallin, *"Local Scale Invariance and Robustness of Proper Scoring Rules"*, arXiv:1912.05642 / Statistical Science 38(1), 2023): `-E|X-y|/E|X-X'| - 0.5*log(E|X-X'|)` for `X, X' ~ F` independent, a proper *scoring rule* (higher is better, unlike every loss-style metric above). Both expectations reuse `CRPS`'s own quantile-decomposition trick: `E|X-X'|` (the Gini mean difference of `F`) is `4 * mean_q((level_q - 0.5) * Q(level_q))` — the standard quantile-function identity for the Gini mean difference, discretized the same way `CRPS` discretizes its own integral; `E|X-y|` then falls out of `CRPS`'s own definition, `CRPS = E|X-y| - 0.5*E|X-X'|`. Nonlinear in these expectations, so (unlike `CRPS`) it pools both terms over the whole evaluation set *before* combining them, rather than averaging already-combined per-sample values — same reasoning `NSE`'s poolable path uses. **This is the one to use for comparing/averaging scores across series of different scale** (e.g. `val_log_aggregated`'s mean-across-validation-sets step) — the paper's own Table 3 proves `SCRPS` is locally scale invariant while plain `CRPS` is not; averaging raw `CRPS` across series of very different magnitude is dominated by whichever series has the largest scale, the same failure mode "Group-reweighting" above fixes for training.
- **`RSCRPS(c)`** — no `poolable`/pooled-value beyond the standard `num`/`denom` accumulators; "robust" `SCRPS`, replacing the plain `|x-x'|` kernel with a clipped kernel `g_c(x,x') = min(|x-x'|, c)` (same Bolin & Wallin paper, their Corollary 1 robust kernel applied via the standardized/`-0.5*log` construction). **Not locally scale invariant** — the paper's own Table 3 lists `rSCRPS` (this class) as robust but *not* scale invariant, and its Proposition 3 proves why: `c` is one fixed constant in `y`'s absolute units, so across series of varying scale it either barely clips a large-scale series (no robustness benefit) or over-clips a small-scale one; the paper calls building a scoring rule that is both robust *and* properly scale-invariant "an open question," and their own suggested mitigation (scaling `c` by some reference distribution) is explicitly flagged as "still problematic," not something adopted here. Do not use `RSCRPS` to compare/average across series of very different scale — use `SCRPS` for that; `RSCRPS` is for outlier-robustness within a single, roughly fixed-scale evaluation.
- **`_Metric.bind_levels(kind, levels)`** builds `self._level_index = {level: channel_index}`, populated for **both** `kind='quantile'` and `kind='expectile'` (empty for `'point'`). This is a deliberate, accepted approximation: the interval/coverage/width/CRPS formulas above are only exactly correct for true quantiles, but channel lookup by level doesn't care which kind produced the level, and reusing them against an expectile head's levels is treated as good enough rather than gated off.
- **`_Metric._select_point(y_pred)`** — used by every ordinary point-style metric (`MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `NSE`, `AlphaNSE`, `BetaNSE`, `DILATE`) as the first line of `forward()`/`accumulate()`: passes `y_pred` through unchanged if it already has one channel, otherwise selects the level-`0.5` channel (median under a quantile head, mean under an expectile head). This is what lets `'mae'` etc. keep working unmodified as a `validation_logging_criteria` entry even when `loss_function='quantile_loss(...)'`/`'expectile_loss(...)'` — a plain `KeyError` (no `0.5` level bound) is the deliberate crash if there isn't one. `DILATE` calls it *after* its own `dim==2 → unsqueeze` normalization, so its legacy 2-D calling convention (no explicit channel dim) is unaffected.
- **`spec_name`/`_spec_kwargs()`** — each of these 6 classes sets `spec_name` (its `REGISTRY` key, e.g. `'quantile_loss'`) and overrides `_spec_kwargs()` (e.g. `{'q': self.output_levels}`) so `metric_to_spec` can round-trip them through checkpoints; `metric_to_spec` consults these when present, leaving its pre-existing `DILATE`/bare-name branches untouched.
- Registered in `REGISTRY` as `quantile_loss`, `expectile_loss`, `winkler_score`, `crps`, `scrps`, `rscrps`, `picp`, `pinaw`, `coverage_gap`.

---

## `src/utils/regularization.py`

Pluggable weight regularizers for `Forecaster`, selected via the `regularization=` spec string (same `parse_spec` grammar as `optimizer`/losses, e.g. `'l1(lambda_=0.01)'`, `'rln(theta=-4.0, lr=0.01)'`). `resolve_regularization(spec)`: `None → None`, string → instance via `REGISTRY = {'l1': L1, 'rln': RLN}`.

**`_Regularizer`** — base class. `bind(params, lr)` runs once at `Forecaster` construction with the weight-only parameter list (`self._regularized_weights` — biases excluded, same split used for `weight_decay`) and the Forecaster's `learning_rate`. `step(params)` runs every training step, right after `loss.backward()` (so `p.grad` already holds the plain data gradient) and before `optimizer.step()` — may mutate `p.grad` in place. Regularizers never touch the loss graph itself, so `self.loss_log` is always the plain data loss regardless of `regularization`. `get_state()`/`load_state(state)` handle any checkpointable state beyond the constructor's own spec kwargs (empty for stateless regularizers).

**`L1(lambda_=0.01)`** — one shared coefficient for every weight. `step()`: `p.grad += lambda_ * sign(p.data)` for every weight tensor. The trivial case of the same interface `RLN` uses (one shared `λ` instead of one per weight).

**`RLN(theta=-4.0, lr=None)`** — Regularization Learning Networks (Shavitt & Segal, *"Regularization Learning Networks"*, arXiv:1805.06440, NeurIPS 2018): one L1 coefficient learned **per individual weight**, jointly with the weights, instead of a single shared `λ`. The paper's derivation, and what each piece becomes here:

| Theory | Code |
|---|---|
| `L† = L(Z,W) + Σᵢ exp(λᵢ)·\|wᵢ\|` — one `λ` per weight scalar | `self._lambdas`: a list of tensors shaped like each tensor in `params`, `torch.full_like(p, theta)` at `bind()` |
| `w_{t+1,i} = w_{t,i} − η·(g_{t,i}+r_{t,i})`, `g=∂L/∂w`, `r_{t,i}=exp(λ_{t,i})·sign(w_{t,i})` | `r` computed in closed form (no autograd) and added directly onto `p.grad` (which already holds the plain `g_t` from `loss.backward()`) — `r` is deliberately kept separate from the loss graph so `g_t` stays available on its own |
| Counterfactual Loss `L_CF = L(Z_{t+1}, W_{t+1})`; its gradient `∂L_CF/∂λ_{t,i} = −η·g_{t+1,i}·r_{t,i}` needs *next* batch's plain gradient — which the next `step()` call computes anyway, for free | `self._prev_r` stores `r_t`; at the *next* `step()` call, `p.grad` already *is* `g_{t+1}` — combined with the stored `r_t` first, before that step computes its own `r` |
| `λ_{t+1,i} = λ_{t,i} + ν·η·g_{t+1,i}·r_{t,i}` | `self._nu` (`lr`, defaults to the Forecaster's `learning_rate` if not given) and `self._eta` (that `learning_rate`), captured at `bind()` |
| Simplex projection: `λ_{t+1,i} = λ̃_{t+1,i} + (θ − mean_j λ̃_{t+1,j})`, mean over **every weight in the whole network** — without it every `λ` drifts to `−∞` (regularization collapses network-wide); pinning the global mean to `θ` makes it a zero-sum redistribution, and `θ` becomes the *one* tunable hyperparameter | After the raw update, `shift = theta − mean(concat(all λ tensors))`, added to every tensor in `self._lambdas` |

`get_state()`/`load_state()` save/restore `self._lambdas` (the actually-*learned* per-weight coefficients) for checkpointing — `self._prev_r` is deliberately *not* persisted (one-step lookback state, not learned state; the first post-resume step just skips one `λ` update).

**Caveat:** Theorem 1 above is derived for vanilla SGD's literal `w − η(g+r)` update. `Forecaster`'s `optimizer=` can select Adam/AdamW/etc., which apply momentum/adaptive scaling on top of whatever gradient they're handed — the paper asserts (footnote) the analysis "holds for any derivative-based optimization method" without re-deriving it. Implemented exactly as described; treated as a literature-consistent approximation for non-SGD optimizers, not a re-derivation.

Only the L1-norm variant is implemented (the paper's own best-performing variant) — no generalized-norm option, and none of the paper's feature-importance/sparsity analysis tooling (its Section 6).

---

## `src/utils/DILATE/`

Ported from `marcdemers/batch-DILATE` (a batched fork of the original `vincent-leguen/DILATE` reference implementation), replacing an earlier pure-PyTorch version that had two problems: (1) it looped over the DP grid cell-by-cell in Python (~49 sequential ops per batch for `horizon=7`), and (2) it computed the temporal-alignment term via `.detach()`ed tensors under `@torch.no_grad()`, so `loss_temporal` never actually contributed gradients — DILATE was silently training on the shape term alone. Both are fixed by this port; correctness was verified against the real upstream package (loss values and gradients matched to ~3e-5).

- **`loss_utils.py`** — `numba_min`/`numba_max_hessian_product`/etc., `@njit` helpers for the softmin and Hessian-vector-product math. `numba_apply_along_axis` deliberately omits `cache=True` — it takes a function (`np.max`/`np.sum`) as an argument, which Numba can't pickle for its on-disk cache on this Numba version; everything downstream of it keeps `cache=True`.
- **`soft_dtw.py`** — `compute_softdtw_batch_channel`/`compute_softdtw_backward_batch_channel` (`@njit`, vectorized over a `(batch, channel)` leading pair — one Numba call per forward/backward, not one per sample) and `SoftDTWBatch(torch.autograd.Function)`, whose `forward`/`backward` round-trip through `.cpu().numpy()` to call them and hold the shape (soft-DTW) loss.
- **`path_soft_dtw.py`** — `dtw_grad`/`dtw_hessian_prod` (`@njit`) and `PathDTWBatch(torch.autograd.Function)`, which produces the temporal-alignment path via a hand-derived Hessian-vector-product backward — this is what makes `loss_temporal` actually differentiable.
- **`dilate_loss.py`** — glue: computes `D` (our own direct squared-difference formula, not the reference's norm-expansion/`.double()` variant), adds a dummy `channel=1` axis to match the ported kernels' `(batch, channel, N, N)` shape, calls `SoftDTWBatch.apply`/`PathDTWBatch.apply`, returns `(loss, loss_shape, loss_temporal)` — same signature as before the port.

**No `torch.compile` support, intentionally.** `torch.compile` cannot safely trace into `SoftDTWBatch`/`PathDTWBatch` — Dynamo tries to trace their `.cpu().numpy()`/Numba internals and crashes (a `CppCompileError` under the old pure-PyTorch implementation's giant unrolled loop; a Dynamo fake-tensor dtype error under this Numba-based one). `torch._dynamo.disable`/`allow_in_graph` were tried and don't fix it either (the former doesn't intercept when the disabled function is itself the direct `torch.compile()` target; the latter's own docs warn it's the wrong tool for black-boxing a function through Inductor — a proper fix would mean rewriting `SoftDTWBatch`/`PathDTWBatch` as `torch.library.custom_op`s, out of scope). Since Numba JIT already makes DILATE fast in eager mode, `use_torch_compile` was removed entirely from `DILATE.__init__` and from `Forecaster.__init__`'s compile-sharing block — `Forecaster(use_torch_compile=True)` now only ever compiles the model, never the loss, so it no longer crashes when combined with a DILATE loss.

---

## `examples/example_model_and_loss_comparison.ipynb`

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

---

## `examples/example_regularization_rln.ipynb`

Second reference notebook, same `DataSource`/`HISTORIC_COLS`/`TARGET_COL`/`HORIZON`/`SEQ_LEN` setup as `example_model_and_loss_comparison.ipynb`, demonstrating `regularization='rln(...)'` specifically. Trains two identical `LSTMHistoric` models (`mae` loss, `LSTM_hidden_size=28`, 50 epochs) differing only in `regularization`: `fc_baseline` (`None`) vs. `fc_rln` (`'rln(theta=-8.0)'`). `theta` is dataset/architecture-dependent (see `RLN`'s docstring) — `-8.0` was found by sweeping `[-4, -6, -7, -8, -9, -10]` on this exact setup; `-4.0` (a naive first guess) collapsed the model entirely (99.8% of weights crushed to ~0, NSE went negative and never recovered — regularization overwhelmed the data gradient from the very first step, since every weight starts at `exp(theta)` uniformly). `-8.0` gives a real demonstration of the paper's claim without sacrificing much accuracy: RLN reaches best NSE 0.79 vs. the baseline's 0.81, while driving ~71% of weights below `1e-3` vs. the baseline's ~0.6%.

`plot_weight_magnitudes(models: dict[str, Forecaster], threshold=1e-3)` — defined inline in the notebook (not extracted to `src/`, matching `example_model_and_loss_comparison.ipynb`'s convention that plotting stays notebook-local): flattens every tensor in `fc._regularized_weights` per model into `|weight|`, then two panels — sorted-descending `|weight|` vs. rank (log-y) on the left, and a density histogram of `|weight|` (log-y) on the right, both overlaid across models — plus prints the fraction below `threshold` per model. The histogram makes RLN's effect visible a second way: a sharp density spike near 0 next to the baseline's much flatter, wider spread. Everything after that (training diagnostics, hydrograph, scatter, NSE-vs-lead-time) reuses `example_model_and_loss_comparison.ipynb`'s cell patterns, simplified from four models down to two.

---

## `examples/example_probabilistic_forecasting.ipynb`

Third reference notebook, same `DataSource`/`HISTORIC_COLS`/`TARGET_COL`/`HORIZON`/`SEQ_LEN` setup as `example_model_and_loss_comparison.ipynb`/`example_regularization_rln.ipynb`, demonstrating the probabilistic output head (`quantile_loss`/`expectile_loss`, `src/models/heads.py`) and the new evaluation scores (`winkler_score`, `crps`, `scrps`, `coverage_gap`, `pinaw`). `CoverageGap` is displayed multiplied by 100 everywhere in this notebook (training diagnostics, coverage-gap-vs-lead-time, axis labels all say "(%)") since it's a coverage-probability difference and reads more naturally as a percentage than a fraction. Trains two `LSTMHistoric` models (`LSTM_hidden_size=28`, 30 epochs), identical except for the loss:

| Var | Loss | `output_kind` |
|---|---|---|
| `fc_qr` | `quantile_loss(q=LEVELS)` | `'quantile'` |
| `fc_er` | `expectile_loss(e=LEVELS)` | `'expectile'` |

`LEVELS = [round(0.025 * i, 6) for i in range(1, 40)]` — every PI width from 95% down to 5% in 5% steps (each contributing its `alpha/2`/`1-alpha/2` quantile pair — e.g. the 95% PI's 0.025/0.975, the 90% PI's 0.05/0.95, ...) lands on this same step-0.025 grid, 39 levels total, plus the center (0.5: median for QR, mean for ER) — all in one 39-channel output head, built as a `MonotonicHead` (see `heads.py` above) so all 39 channels are guaranteed non-crossing by construction (`LO95 <= LO75 <= CENTER <= HI75 <= HI95`, and everything in between), which the fan chart's shaded bands rely on. `LO95`/`HI95`/`LO75`/`HI75`/`CENTER` (defined right next to `LEVELS`, via `LEVELS.index(...)`) are the channel indices for the 95%/75% PI bounds and center within this 39-wide axis. `validation_score='crps'` for both; `validation_logging_criteria=['mae', 'scrps', 'winkler_score(alpha=0.05)', 'coverage_gap(alpha=0.05)', 'pinaw(alpha=0.05)', 'winkler_score(alpha=0.25)', 'coverage_gap(alpha=0.25)', 'pinaw(alpha=0.25)']` — scored at just the 95%/75% PIs (`alpha=0.05`/`0.25`) regardless of the denser training grid; `mae` demonstrates `_select_point`'s center-channel auto-select against a genuinely multi-channel output, working unmodified for both the QR (median) and ER (mean) models since `bind_levels` accepts both `output_kind`s.

1. **Training diagnostics** — loss (QR and ER on separate panels, different scales, same convention as `example_model_and_loss_comparison.ipynb`'s MAE-vs-DILATE split), CRPS validation score, sCRPS (a scoring rule — higher is better, unlike every other panel here), Winkler score, `CoverageGap` (straight from `val_log_aggregated`, not computed by hand, shown as a percentage), and PINAW (each at both PI widths) — all vs. epoch, both models overlaid.
2. **Evaluation on `test_source` and `val_source`** — `evaluate(source)` (predict + `TimeSeriesDataset`/`_read_1d_df` for window/date alignment, same pattern as `example_model_and_loss_comparison.ipynb`/`example_regularization_rln.ipynb`) called on both, so later cells can compare in- vs. out-of-validation-sample behavior. `fc_qr.output_levels == fc_er.output_levels == LEVELS` asserted so the channel order (`sorted(LEVELS)`) the earlier `LO95`/etc. indices assume is actually what the trained heads produced.
3. **Fan chart** — observed vs. predicted at `lead=1` on `test_source`, QR and ER side by side (shared y-axis): the 95% PI as a light shaded band, the 75% PI as a darker one, plus the center line.
4. **Coverage gap / PINAW vs. lead time** — same NaN-masking-then-metric-class pattern as `example_model_and_loss_comparison.ipynb`'s "NSE vs. lead time" cell, computed per lead via `coverage_gap`/`pinaw` metric instances directly (`bind_levels` once per model, reused across leads) at both PI widths, on both `test_source` (solid) and `val_source` (dashed) — a test/val gap here would flag overfitting to validation. Coverage gap values are multiplied by 100 for display.
5. **Quantitative summary** — all of `VAL_CRITERIA` plus `crps`, computed directly via the metric classes on the full pooled (NaN-masked) test-set array, printed per model with uppercase abbreviation labels (`MAE`, `CRPS`, `SCRPS`, `PICP`, `PINAW`, `Winkler Score`) via a small `METRIC_LABELS` dict.
