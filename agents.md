# Codebase Overview

## Environment

Dependency management is `uv`-based (`pyproject.toml` + `uv.lock`, dev tools like `pytest` under `[dependency-groups] dev = [...]`). Always run Python/pytest/tooling via `uv run <cmd>` (e.g. `uv run pytest`, `uv run python -c "..."`) — never a bare `python`/`pytest` call or manual venv activation. `uv run` targets the project's `.venv` on its own, even if an unrelated environment (e.g. a conda env) happens to be active in the shell — that's expected and correct, not something to work around. Add new dev dependencies via `uv add --group dev <pkg>` (or edit `pyproject.toml` + `uv lock`) rather than ad-hoc `pip install`.

## Structure

```
src/
  data/
    __init__.py              — exports DataSource
    datasource.py             — DataSource dataclass
    timeseries_dataset.py     — TimeSeriesDataset + _collate_fn
    normalization.py          — compute_norm_stats, _read_1d_df helper
  forecaster/
    forecaster.py              — training loop, embeddings/model binding, logging, checkpointing
  models/
    base.py                    — BaseForecastModel (settings-then-bind scaffolding shared by every backbone)
    embeddings.py               — EmbeddingNetwork (settings-then-bind per-group embedding, owned by Forecaster)
    lstm_historic.py            — LSTMHistoric
    lstm_encoder_decoder.py      — LSTMEncoderDecoder
    heads.py                    — PooledHead, SequenceHead, MonotonicHead, build_output_head
  optim.py                       — settings-then-bind wrappers around torch.optim (Adam/AdamW/SGD)
  utils/
    scores_and_losses.py         — metric classes (constructed directly, no string registry)
    regularization.py             — pluggable weight regularizers (L1Regularizer, L2Regularizer, RLN)
    DILATE/
      soft_dtw.py                — Numba-JIT soft-DTW forward/backward DP + SoftDTWBatch autograd.Function
      path_soft_dtw.py            — Numba-JIT Hessian-vector-product DP + PathDTWBatch autograd.Function
      loss_utils.py                — Numba softmin/Hessian-product helpers used by path_soft_dtw.py
      dilate_loss.py                — DILATE loss combining shape + temporal terms
      __init__.py
tests/                             — pytest suite (test_embeddings.py, test_models.py,
                                      test_optim_regularization.py, test_forecaster_end_to_end.py)
```

---

## Design: "settings, then bind"

Every pluggable piece of this codebase is a real, directly-constructed Python object — never a string spec, never a config dict. Two categories:

- **Fully real at construction** — losses/evaluation metrics (`QuantileLoss(q=[0.1, 0.5, 0.9])`, `MAE()`). Nothing about them depends on information that doesn't exist yet, so `loss(y_pred, y_true)` works the instant you construct `loss`.
- **Settings now, `.bind(...)` later** — anything that needs something only `Forecaster` can supply once it exists: `EmbeddingNetwork` needs the actual column list, a model (`LSTMHistoric`/`LSTMEncoderDecoder`) needs the embedding-derived input size and the loss-derived output shape, a regularizer/optimizer needs real parameter tensors. Construction stores settings only — no real `nn.Linear`/`nn.LSTM` layers exist yet — and `.bind(...)`, called once by `Forecaster`, is what actually allocates them. This is one pattern applied uniformly everywhere it's needed, not a special case per component.

Nothing is ever passed as a dict or a string. A metric/regularizer/optimizer/embedding/model class's `__init__` signature is the complete, IDE-discoverable list of what it accepts.

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

**`_merge_stats(n_a, mean_a, M2_a, n_b, mean_b, M2_b)`** — Chan's parallel variance algorithm. Merges two `(n, mean, M2)` accumulators into one without storing raw values. Exact — no approximation. Reused by `ChanAccumulator` in `scores_and_losses.py`.

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

Lazy `torch.utils.data.Dataset`. Pre-loads 1D data into memory. Always returns *raw, unembedded* `x_h`/`x_f` — embedding is entirely a `Forecaster`-level concern now (see below), so this class is unaffected by whether any embeddings are configured.

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

`group_idx` is unconditional (present even when `reweight` is never used — harmless extra key `Forecaster` excludes before the model sees anything) and is meaningful only *within* one `TimeSeriesDataset`/`Forecaster` instance: since `Forecaster` always builds `TimeSeriesDataset(sources=training_datasets, ...)` from the exact same list it holds, position *i* here matches `training_datasets[i]` for that run. It is not itself persisted across checkpoint resumes — `Forecaster` re-derives the array group-reweighted losses actually bind against fresh, every run, by looking up each of *this* run's `training_datasets` by their stable `group_key` (see `DataSource.group_key` above) in the checkpoint's saved `group_target_std`, so a reordered `training_datasets` on resume still lines up correctly.

**`_collate_fn(batch)`** — custom collate that stacks tensors for the intersection of keys across all items in the batch. Import from `src.data.timeseries_dataset`.

---

## `src/models/embeddings.py` — `EmbeddingNetwork`

Per-timestep input embedding for one named group of raw columns. Owned entirely by `Forecaster` — a model never sees embeddings or raw column names at all. Follows the "settings, then bind" pattern:

**`EmbeddingNetwork(name, variables, embedding_dim, historic, future, num_layers=1, hidden_size=None, output_activation=None)`** — construction stores settings only, no real layers yet:

| Arg | Meaning |
|---|---|
| `name` | required, non-empty, unique across whatever `embeddings=` list is passed to `Forecaster` — used as this embedding's checkpoint key |
| `variables` | raw column names this group claims |
| `embedding_dim` | output width |
| `historic`, `future` | at least one must be `True` — which side(s) this embedding applies to |
| `num_layers` | `1` → a plain `nn.Linear` (broadcasts over every leading dim for free — no explicit time loop needed); `>1` → an MLP with `GELU` between hidden layers |
| `hidden_size` | hidden width for `num_layers > 1`; defaults to `embedding_dim` |
| `output_activation` | `None` / `'gelu'` / `'softmax'` |

**`.bind(historic_cols, future_cols)`** — called once by `Forecaster`, once it knows the real column layout. Validates: if `historic=True`, every `variables` entry must be present in `historic_cols` (else `ValueError` listing what's missing), symmetrically for `future=True`/`future_cols`. Builds the real `nn.Sequential`/`nn.Linear` (`in_features=len(variables)`), and records `self._historic_idx`/`self._future_idx` — the column index list `Forecaster` uses to slice the raw `x_h`/`x_f` tensor before calling this network. Raises `RuntimeError` if called twice.

A group with **both** `historic=True` and `future=True` is a single instance — one set of weights — applied to both sides by `Forecaster` (once per side, using the corresponding index list): weight sharing is automatic whenever a group spans both sides, not a separate opt-in flag. Independent (unshared) per-side networks over the same columns are expressed as **two** separate `EmbeddingNetwork`s instead — one `historic`-only, one `future`-only.

**`forward(x)`** — `self.net(x)` on whatever tensor slice it's handed; raises `RuntimeError` if called before `.bind()`. This class has no awareness of which side (historic/future) a call is for, nor of any other embedding — that combination logic lives in `Forecaster`, not here (see below): `Forecaster` owns slicing each bound network's claimed columns out of the raw `x_h`/`x_f`, applying it, and concatenating every network's output with whatever raw columns no embedding claimed (passthrough, untouched, original order). Zero embeddings given (`embeddings=[]`, the default) makes the whole pipeline pure identity passthrough.

---

## `src/models/base.py` — `BaseForecastModel`

Shared scaffolding every forecasting model builds on, regardless of backbone architecture. Also follows "settings, then bind":

**`__init__(self, hidden_size, num_layers, dropout_rate)`** — every subclass calls `super().__init__(...)` first with just these three, generic, backbone-agnostic hyperparameters. No columns, no embeddings, no output shape — none of that is knowable yet, so no real layers are allocated here.

**`.bind(historic_input_size, future_input_size, forecasting_horizon, n_outputs, output_kind, output_levels)`** — called once by `Forecaster`, after it has bound and combined the embeddings (so `historic_input_size`/`future_input_size` are known numbers) and resolved the loss (so `n_outputs`/`output_kind`/`output_levels` are known). Stores these on `self`, then calls `self._build_backbone()` — the subclass hook that actually allocates the LSTM(s) and, via `self.build_output_head(...)`, the output head. Raises `RuntimeError` if called twice. `forward()` on every subclass starts with `self._check_bound()`, raising a clear `RuntimeError` rather than an obscure `AttributeError` if called before `.bind()`.

**`build_output_head(representation, in_features, horizon=None)`** — thin wrapper around `heads.build_output_head`, filling in `self.n_outputs`/`self.output_kind`/`self.output_levels` (set by `.bind()`) so a subclass only supplies what's actually architecture-specific.

**`parameter_breakdown()`** — returns `[(label, param_count), ...]`: a backbone row (labeled `type(self).__name__`, counting everything not under `output_head.`, computed by exclusion so it needs no per-architecture bookkeeping), a head row (labeled `type(self.output_head).__name__`), and a `('total', ...)` row. No embedding rows here — embeddings live on `Forecaster`, which prepends its own rows (one per bound embedding) before these when it calls `print_parameter_breakdown()`.

---

## `src/models/lstm_historic.py`, `src/models/lstm_encoder_decoder.py`, `src/models/heads.py`

Both models subclass `BaseForecastModel` (above). Their constructors take only their own architecture hyperparameters — `LSTMHistoric(hidden_size, num_layers, dropout_rate)`, `LSTMEncoderDecoder(hidden_size, num_layers, dropout_rate, encoder_downscale_layer_size)` — and implement `_build_backbone(self)`, the hook `.bind()` calls once `self.historic_input_size`/`self.future_input_size`/`self.forecasting_horizon`/`self.n_outputs`/`self.output_kind`/`self.output_levels` are all set.

**`forward(self, x_h, x_f)`** — plain already-embedded tensors, not a dict — `Forecaster` applies embeddings before ever calling the model. `LSTMHistoric` simply ignores `x_f` (it only ever encoded the historic side, even before this refactor).

Neither model owns its final projection directly — each builds its output head via `self.build_output_head(representation, in_features, horizon=None)` (the `BaseForecastModel` wrapper around `heads.build_output_head`), always assigned to the attribute `self.output_head` (`Forecaster` keys off this exact name to exclude its params from weight_decay/regularization — see below). Two backbone shapes (passed as `representation`), two raw head classes:

- **`PooledHead(in_features, horizon, n_outputs)`** (`representation='pooled'`) — for a backbone that produces one pooled vector per sample, `(B, hidden)`. `nn.Linear(hidden, horizon * n_outputs)`, reshaped to `(B, horizon, n_outputs)`.
- **`SequenceHead(in_features, n_outputs)`** (`representation='sequence'`) — for a backbone that already produces a per-timestep representation, `(B, horizon, hidden)`. Plain `nn.Linear(hidden, n_outputs)` (broadcasts over the middle dim for free).

**`LSTMHistoric`** — encodes only `x_h`; single LSTM → last hidden → output head (`'pooled'`) → `(B, horizon, n_outputs)` (its last hidden state is mapped straight to every horizon step at once — there's no per-step decoder representation to project from, hence `'pooled'` rather than `'sequence'`).

**`LSTMEncoderDecoder`** — encoder LSTM on `x_h`; hidden downscaled and repeated over horizon, concatenated with `x_f`; decoder LSTM → output head (`'sequence'`) → `(B, horizon, n_outputs)`.

**`MonotonicHead(raw_head, center_idx)`** — the standard (always-on, not opt-in) wrapper `build_output_head` applies around the raw head whenever `output_kind in ('quantile', 'expectile')`: guarantees non-crossing levels by construction instead of relying on the loss alone to discourage crossing (a known failure mode of independent per-quantile heads; cf. Cannon 2018's monotone composite quantile regression). The raw head's channel at `center_idx` (the level-`0.5` channel's position within `output_levels`, sorted ascending — `sorted(output_levels).index(0.5)`, a plain `ValueError` if `0.5` isn't among the levels, deliberately uncaught) is used directly and unconstrained as the center value — the median for a quantile head, the mean for an expectile head (see `ExpectileLoss`). Every channel above center becomes a strictly positive offset via `softplus`, cumulatively summed outward (`softplus(above).cumsum(-1)`, added to center) so each successive higher level is `>=` the one before it; channels below center are built the same way in reverse (`softplus(below.flip(-1)).cumsum(-1).flip(-1)`, subtracted from center) so each successive lower level is `<=` the one before it. `raw_head` is nested as a submodule (`self.raw_head`), so its params surface as `output_head.raw_head.*` — still matched by the `output_head.` prefix check that excludes it from regularization. A plain point head (`output_kind='point'`, `n_outputs=1`) is returned unwrapped — unaffected by any of this.

For `n_outputs=1` (the default, and every plain point loss like `MAE()`/`DILATE()`), both heads are exactly equivalent to a hard-coded `Linear(hidden, horizon)`/`Linear(hidden, 1)` output layer. `n_outputs > 1` is how the probabilistic losses below (`QuantileLoss`/`ExpectileLoss`) get a multi-channel output — `Forecaster` reads `n_outputs`/`output_kind`/`output_levels` straight off the resolved `loss` object and passes them into `model.bind(...)`.

---

## `src/utils/scores_and_losses.py`

Every metric class below is a plain, directly-constructible `nn.Module` — `MAE()`, `DILATE(alpha=0.5, gamma=0.01)`, `QuantileLoss(q=[0.1, 0.5, 0.9])`. There is no string-spec registry, no `resolve_metric`, no `parse_spec` — construct the class you want directly and hand the instance to `Forecaster`.

**`assert_differentiable(metric)`** — raises `TypeError` if metric has `differentiable = False`. Called by `Forecaster` on whatever `loss` it's given.

**Metric classes:** `MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `DILATE`, `NSE`, `AlphaNSE`, `BetaNSE` — all wrap `nn.Module`. `DILATE` has `eval_on_normalized = True` (soft-DTW overflows on large denormalized values). `NSE` (Nash-Sutcliffe Efficiency), `AlphaNSE` (variability ratio, the α term from the Gupta et al. 2009 NSE/KGE decomposition — matches `neuralhydrology`'s `alpha_nse`, `sim.std()/obs.std()`, verified against its source) and `BetaNSE` (bias term, same decomposition — matches `neuralhydrology`'s `beta_nse`, `(mean(sim)-mean(obs))/std(obs)`) are not lower-is-better losses and set `differentiable = False` so `assert_differentiable` blocks them from being used as `loss`. `AlphaNSE`/`BetaNSE` use population variance (`unbiased=False`, ddof=0 — matching numpy/xarray's default, and hence the reference) rather than `torch.std()`'s default ddof=1; for `AlphaNSE` this is a ratio of two same-`n` variances so the correction would cancel either way, but for `BetaNSE` (no matching denominator term) the convention actually changes the numeric result.

`DILATE(alpha=0.5, gamma=0.01)` — no `use_torch_compile` parameter; always eager (see below).

**Group-reweighting (`reweight=True` on `MAE`, `MSE`, `QuantileLoss`, `ExpectileLoss`).** Fixes multi-series training being dominated by whichever series has the largest-scale target (pooled normalization uses one global mean/std across all series, so a large-scale series' error is numerically larger in that shared space) — without needing stratified/balanced batches, so it scales to any number of series; batches stay randomly shuffled and mixed-series. Mechanically this is a per-sample error reweighting by that sample's own series' scale, not a change to how `y` is normalized/denormalized (both untouched). Verified to match `neuralhydrology`'s `MaskedNSELoss` formula exactly for the `MSE` case (same reference-implementation-checking as `AlphaNSE`/`BetaNSE` above); the `MAE`/`QuantileLoss`/`ExpectileLoss` cases are this codebase's own dimensionally-consistent extension of the same idea, not independently verified against a reference implementation.

- **`_Metric` additions:** `_reweight_power: int | None = None` (class attribute — `None` means "this metric doesn't support `reweight=True`"), `reweight`/`eps` constructor kwargs (default `False`/`0.1`), `needs_group_idx: bool = False` (instance attribute, flips to `True` only when `reweight=True`), `bind(group_std)` (registers the per-series std as a buffer — attached by `Forecaster` after construction, the same "attach data-dependent state once it exists" idea `RLN.bind` and `bind_levels` also follow), and `_error(y_pred, y_true)` (abstract — the metric's own unweighted elementwise error, in its own output shape; point metrics call `_select_point` inside their `_error`, multi-channel losses like `QuantileLoss`/`ExpectileLoss` don't).
- **Dispatch is picked once, at construction, not branched on every call.** `__init__` checks whether the concrete class overrides `_error` (`type(self)._error is not _Metric._error`); if so, it assigns `self.forward = self._reweighted_forward` or `self._plain_forward` (an instance attribute shadows the class's `forward` method, so every subsequent call goes straight to the right implementation with no per-batch conditional). Classes that don't override `_error` (`RMSE`, `MAPE`, `SMAPE`, `NSE`-family, `DILATE`, `WinklerScore`, `PICP`, `CoverageGap`, `CRPS`, `SCRPS`, `RSCRPS`) are completely untouched by this — their own class-level `forward()` keeps running unmodified — and passing `reweight=True` to one of them raises `ValueError` at construction (`_reweight_power` stays `None`).
- **Exponent is tied to the error's own units, not independently configurable:** `MAE`/`QuantileLoss` (`_reweight_power=1`, weight `1/(group_std+eps)`, i.e. std) — pinball loss is piecewise-linear in the residual, same `[y]¹` character as absolute error. `MSE`/`ExpectileLoss` (`_reweight_power=2`, weight `1/(group_std+eps)²`, i.e. variance) — the expectile term is quadratic in the residual, same `[y]²` character as squared error. Mismatching the exponent to the error's power would leave the reweighted term scale-dependent again, so it's deliberately not offered as a separate parameter.
- **`CRPS` deliberately does not get this.** `differentiable=False` blocks it from ever being `loss`, so a `reweight` option on it would be inert — group-reweighting only matters for what's actually trained on, and validation is already handled correctly per-series without it (see "Pooled (exact) validation metrics" below). Revisit only if `CRPS` is separately promoted to a trainable loss.
- **`group_std` is bound by `Forecaster`** — see `Forecaster`'s group-reweighting wiring below.
- **Not scale-invariant across series if applied to a validation/scoring metric** — this mechanism fixes the *training* imbalance specifically; it isn't a general "make any metric comparable across series of different scale" tool (see the `SCRPS`/`RSCRPS` discussion below for that distinct problem, which this mechanism does not address).

**Pooled (exact) validation metrics.** `forward(y_pred, y_true)` on every metric class still computes on whatever tensor you pass it (unchanged) — used directly for e.g. computing a metric once over a full concatenated test-set array in a notebook. But naively averaging *per-batch* `forward()` values, which is what `Forecaster`'s evaluation loop would do naively, is not the same as computing a metric once over the whole pooled validation set for anything that isn't a plain per-sample mean: NSE/AlphaNSE/BetaNSE are ratios whose denominator depends on the batch's own local mean/variance (a low-variance batch — e.g. a baseflow-only window — can swing its per-batch NSE wildly, then get averaged in with equal weight regardless of size); RMSE's `sqrt` doesn't commute with averaging; and even MAE/MAPE/SMAPE (plain means) are only exactly right under batch-averaging if every batch is the same size, which the last batch usually isn't under `drop_last=False`.

Fixed via an opt-in pooled API on `_Metric`: `poolable: bool = False` (default), and for poolable metrics, `new_accumulators()` (fresh named `ChanAccumulator`s), `accumulate(accs, y_pred, y_true)` (update them with one batch), `pooled_value(accs)` (the exact epoch-level value). `ChanAccumulator` is a small `(n, mean, M2)` streaming accumulator built on `_merge_stats` (imported from `src/data/normalization.py` — the same Chan's-parallel-algorithm merge already used there for feature normalization stats, reused rather than reimplemented). `MAE`/`MSE`/`RMSE`/`MAPE`/`SMAPE` each accumulate one stream (their own elementwise per-sample value; `RMSE` takes `sqrt` of the pooled mean at the end). `NSE` accumulates two (`residual_sq`, `obs`); `AlphaNSE`/`BetaNSE` accumulate two each (`pred`, `obs`). No cross-metric accumulator sharing — each metric owns its own, computed independently even if two requested metrics need the same underlying stream (negligible cost next to the model forward pass). `DILATE` stays `poolable = False` (not a reduction of a simple per-sample stream).

**Probabilistic output heads (`QuantileLoss`, `ExpectileLoss`) and quantile-aware evaluation scores (`WinklerScore`, `CRPS`, `PICP`, `PINAW`).** A loss can declare three extra `_Metric` class fields — `output_kind` (`'point'` default), `output_levels` (`None` default), `n_outputs` (`1` default) — which `Forecaster` reads straight off the *resolved loss object* to size the model's output head (see the model `.bind()` mechanism above) via `model.bind(..., n_outputs=..., output_kind=..., output_levels=...)`. Every other criterion (`validation_score`, `validation_logging_criteria`) then gets `bind_levels(kind, levels)` called on it once, unconditionally, right after — a no-op for metrics that never reference `self._level_index`, and how quantile-specific scores find the right channel(s).

- **`QuantileLoss(q=(0.1, 0.5, 0.9), reweight=False, eps=0.1)`** — pinball loss, vectorized over the last (quantile) dim; sets `output_kind='quantile'`, `output_levels=sorted(q)`, `n_outputs=len(q)`. Poolable (one `ChanAccumulator` per level, averaged). `reweight=True` — see "Group-reweighting" above (`_reweight_power=1`).
- **`ExpectileLoss(e=(0.1, 0.5, 0.9), reweight=False, eps=0.1)`** — asymmetric squared loss (`L_e(u) = |e - 1{u<0}| * u^2`, `u = y_true - y_pred`), same shape as `QuantileLoss` but `output_kind='expectile'`. Its level-0.5 channel is the **mean**, not the median (expectiles ≠ quantiles) — this is what drives the approximation note below. `reweight=True` — see "Group-reweighting" above (`_reweight_power=2`).
- **`WinklerScore(alpha)`**, **`PICP(alpha)`**, **`PINAW(alpha)`** — `differentiable=False` (eval-only, like `NSE`), `poolable=True`. Standard `(1-alpha)` prediction-interval convention: lower/upper bounds are read from the channels at levels `alpha/2` and `1-alpha/2` via `self._level_index` (e.g. `alpha=0.05` pairs with a `q=(0.025, 0.5, 0.975)`-trained head for a 95% PI). `WinklerScore` is the interval score (width, plus a `2/alpha`-scaled penalty for falling outside the interval); `PICP` is the fraction of observations covered (ideally ≈ `1-alpha`); `PINAW` is mean interval width normalized by the pooled range (`max-min`) of `y_true` over the whole evaluation set (plain running min/max alongside a `ChanAccumulator` for width — `ChanAccumulator` itself has no min/max).
- **`CoverageGap(alpha)`** — a thin `PICP` subclass: `PICP - (1-alpha)`, i.e. 0 = perfectly calibrated, positive = over-covered, negative = under-covered, instead of comparing raw `PICP` against a separately-plotted nominal-coverage line. Same lower/upper channel requirements as `PICP` (inherits `__init__`/`bind_levels` behavior, overrides only `_elementwise`). The `(1-alpha)` shift is applied inside `_elementwise` rather than after pooling — since mean is linear, `mean(x - c) == mean(x) - c` exactly, so shifting the elementwise value before it feeds `ChanAccumulator` doesn't bias the pooled epoch-level value the way shifting an already-pooled per-batch average would have.
- **`CRPS()`** — no params. Standard quantile-decomposition approximation (Gneiting & Raftery, 2007): `2 * mean_q(pinball_q)` over *every* channel the head provides (all of `self._level_index`, not just a chosen pair).
- **`SCRPS()`** — no params, `poolable=True`. Scale-invariant CRPS (Bolin & Wallin, *"Local Scale Invariance and Robustness of Proper Scoring Rules"*, arXiv:1912.05642 / Statistical Science 38(1), 2023): `-E|X-y|/E|X-X'| - 0.5*log(E|X-X'|)` for `X, X' ~ F` independent, a proper *scoring rule* (higher is better, unlike every loss-style metric above). Both expectations reuse `CRPS`'s own quantile-decomposition trick: `E|X-X'|` (the Gini mean difference of `F`) is `4 * mean_q((level_q - 0.5) * Q(level_q))` — the standard quantile-function identity for the Gini mean difference, discretized the same way `CRPS` discretizes its own integral; `E|X-y|` then falls out of `CRPS`'s own definition, `CRPS = E|X-y| - 0.5*E|X-X'|`. Nonlinear in these expectations, so (unlike `CRPS`) it pools both terms over the whole evaluation set *before* combining them, rather than averaging already-combined per-sample values — same reasoning `NSE`'s poolable path uses. **This is the one to use for comparing/averaging scores across series of different scale** — the paper's own Table 3 proves `SCRPS` is locally scale invariant while plain `CRPS` is not; averaging raw `CRPS` across series of very different magnitude is dominated by whichever series has the largest scale, the same failure mode "Group-reweighting" above fixes for training.
- **`RSCRPS(c)`** — no `poolable`/pooled-value beyond the standard `num`/`denom` accumulators; "robust" `SCRPS`, replacing the plain `|x-x'|` kernel with a clipped kernel `g_c(x,x') = min(|x-x'|, c)` (same Bolin & Wallin paper, their Corollary 1 robust kernel applied via the standardized/`-0.5*log` construction). **Not locally scale invariant** — the paper's own Table 3 lists `rSCRPS` (this class) as robust but *not* scale invariant, and its Proposition 3 proves why: `c` is one fixed constant in `y`'s absolute units, so across series of varying scale it either barely clips a large-scale series (no robustness benefit) or over-clips a small-scale one; the paper calls building a scoring rule that is both robust *and* properly scale-invariant "an open question," and their own suggested mitigation (scaling `c` by some reference distribution) is explicitly flagged as "still problematic," not something adopted here. Do not use `RSCRPS` to compare/average across series of very different scale — use `SCRPS` for that; `RSCRPS` is for outlier-robustness within a single, roughly fixed-scale evaluation.
- **`_Metric.bind_levels(kind, levels)`** builds `self._level_index = {level: channel_index}`, populated for **both** `kind='quantile'` and `kind='expectile'` (empty for `'point'`). This is a deliberate, accepted approximation: the interval/coverage/width/CRPS formulas above are only exactly correct for true quantiles, but channel lookup by level doesn't care which kind produced the level, and reusing them against an expectile head's levels is treated as good enough rather than gated off.
- **`_Metric._select_point(y_pred)`** — used by every ordinary point-style metric (`MAE`, `MSE`, `RMSE`, `MAPE`, `SMAPE`, `NSE`, `AlphaNSE`, `BetaNSE`, `DILATE`) as the first line of `forward()`/`accumulate()`: passes `y_pred` through unchanged if it already has one channel, otherwise selects the level-`0.5` channel (median under a quantile head, mean under an expectile head). This is what lets `MAE()` etc. keep working unmodified as a `validation_logging_criteria` entry even when `loss=QuantileLoss(...)`/`ExpectileLoss(...)` — a plain `KeyError` (no `0.5` level bound) is the deliberate crash if there isn't one. `DILATE` calls it *after* its own `dim==2 → unsqueeze` normalization, so its legacy 2-D calling convention (no explicit channel dim) is unaffected.

---

## `src/utils/regularization.py`

Pluggable weight regularizers, following the "settings, then bind" pattern (this is the pattern every other settings-then-bind class in the codebase was generalized from).

**`_Regularizer`** — base class. `bind(params, lr)` runs once, called by `Forecaster` right after the relevant module (embeddings or model) has been bound and its real parameters exist — with a weight-only parameter list and a learning rate (so a regularizer's own rate, if unset, can default to it). `step(params)` runs every training step, right after `loss.backward()` (so `p.grad` already holds the plain data gradient) and before `optimizer.step()` — may mutate `p.grad` in place. Regularizers never touch the loss graph itself, so `Forecaster`'s `loss_log` is always the plain data loss regardless of regularization. `get_state()`/`load_state(state)` handle any checkpointable state beyond the constructor's own settings (empty for stateless regularizers).

`Forecaster` binds **two independent regularizer slots**: `embedding_regularization` (applies to embedding-network weights only) and `model_regularizer` (applies to the model backbone's weights only, excluding the output head). They're independent because embeddings and the model backbone are two separately-owned pieces on `Forecaster` — see `Forecaster`'s parameter-pool construction below. Either or both may be `None`. The output head is never regularized by either slot.

**`L1Regularizer(lambda_=0.01)`** — one shared coefficient for every weight in its pool. `step()`: `p.grad += lambda_ * sign(p.data)`.

**`L2Regularizer(gamma=0.01)`** — one shared coefficient, same shape as `L1Regularizer`: `step()`: `p.grad += gamma * p.data`. Deliberately independent of (not a replacement for) an optimizer's own `weight_decay` kwarg — the two may be combined; it's on the caller to avoid double-penalizing if that's not intended.

**`RLN(theta=-4.0, lr=None)`** — Regularization Learning Networks (Shavitt & Segal, *"Regularization Learning Networks"*, arXiv:1805.06440, NeurIPS 2018): one L1 coefficient learned **per individual weight in its pool**, jointly with the weights, instead of a single shared `λ`. The paper's derivation, and what each piece becomes here:

| Theory | Code |
|---|---|
| `L† = L(Z,W) + Σᵢ exp(λᵢ)·\|wᵢ\|` — one `λ` per weight scalar | `self._lambdas`: a list of tensors shaped like each tensor in `params`, `torch.full_like(p, theta)` at `bind()` |
| `w_{t+1,i} = w_{t,i} − η·(g_{t,i}+r_{t,i})`, `g=∂L/∂w`, `r_{t,i}=exp(λ_{t,i})·sign(w_{t,i})` | `r` computed in closed form (no autograd) and added directly onto `p.grad` (which already holds the plain `g_t` from `loss.backward()`) — `r` is deliberately kept separate from the loss graph so `g_t` stays available on its own |
| Counterfactual Loss `L_CF = L(Z_{t+1}, W_{t+1})`; its gradient `∂L_CF/∂λ_{t,i} = −η·g_{t+1,i}·r_{t,i}` needs *next* batch's plain gradient — which the next `step()` call computes anyway, for free | `self._prev_r` stores `r_t`; at the *next* `step()` call, `p.grad` already *is* `g_{t+1}` — combined with the stored `r_t` first, before that step computes its own `r` |
| `λ_{t+1,i} = λ_{t,i} + ν·η·g_{t+1,i}·r_{t,i}` | `self._nu` (`lr`, defaults to whatever `Forecaster` passes into `bind()` if not given) and `self._eta` (that rate), captured at `bind()` |
| Simplex projection: `λ_{t+1,i} = λ̃_{t+1,i} + (θ − mean_j λ̃_{t+1,j})`, mean over **every weight in this regularizer's own pool** — without it every `λ` drifts to `−∞` (regularization collapses within that pool); pinning the mean to `θ` makes it a zero-sum redistribution, and `θ` becomes the *one* tunable hyperparameter | After the raw update, `shift = theta − mean(concat(all λ tensors))`, added to every tensor in `self._lambdas` |

`get_state()`/`load_state()` save/restore `self._lambdas` (the actually-*learned* per-weight coefficients) for checkpointing — `self._prev_r` is deliberately *not* persisted (one-step lookback state, not learned state; the first post-resume step just skips one `λ` update).

**Caveat:** Theorem 1 above is derived for vanilla SGD's literal `w − η(g+r)` update. `Forecaster`'s `optimizer=` can select Adam/AdamW/etc. (see `src/optim.py` below), which apply momentum/adaptive scaling on top of whatever gradient they're handed — the paper asserts (footnote) the analysis "holds for any derivative-based optimization method" without re-deriving it. Implemented exactly as described; treated as a literature-consistent approximation for non-SGD optimizers, not a re-derivation.

Only the L1-norm variant is implemented (the paper's own best-performing variant) — no generalized-norm option, and none of the paper's feature-importance/sparsity analysis tooling (its Section 6).

---

## `src/optim.py`

Settings-then-bind wrappers around `torch.optim`, one per supported optimizer class (currently `Adam`, `AdamW`, `SGD`). A real `torch.optim.Optimizer` can't be constructed without `params` (required, no default in every subclass), and `params` don't exist until `Forecaster` has bound the model/embeddings — so these wrappers hold just the optimizer's own hyperparameters, typed against that optimizer's real signature (`AdamW(lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`), and `.bind(params)` is what constructs and returns the real thing.

Unlike a regularizer's `bind()`, this needs no extra setup step of its own — it's a plain, one-shot construction call (`self._torch_cls(params, **vars(self))`), not stateful. A typed class (rather than e.g. `functools.partial(torch.optim.AdamW, lr=0.001)`) is what lets `Forecaster` save "which optimizer, which settings" via plain attribute access, and gives autocomplete against the real optimizer's kwargs rather than an opaque bag of `**kwargs`.

`Forecaster` reads `self.learning_rate = optimizer.lr` off the bound settings object — there's no separate `learning_rate=` argument on `Forecaster` itself; regularizers that need a rate and weren't given one explicitly (e.g. `RLN(theta=-8.0)` with no `lr`) fall back to this.

`optimizer.bind(param_groups)` is called with a list of per-group dicts (`Forecaster` builds these — see below), the same mechanism `torch.optim.Optimizer` already supports for per-group overrides: a group's own dict can override the settings object's defaults (e.g. pinning `weight_decay=0.0` on the bias/head groups regardless of what `weight_decay` the optimizer settings object itself specifies).

---

## `src/utils/DILATE/`

Ported from `marcdemers/batch-DILATE` (a batched fork of the original `vincent-leguen/DILATE` reference implementation), replacing an earlier pure-PyTorch version that had two problems: (1) it looped over the DP grid cell-by-cell in Python (~49 sequential ops per batch for `horizon=7`), and (2) it computed the temporal-alignment term via `.detach()`ed tensors under `@torch.no_grad()`, so `loss_temporal` never actually contributed gradients — DILATE was silently training on the shape term alone. Both are fixed by this port; correctness was verified against the real upstream package (loss values and gradients matched to ~3e-5).

- **`loss_utils.py`** — `numba_min`/`numba_max_hessian_product`/etc., `@njit` helpers for the softmin and Hessian-vector-product math. `numba_apply_along_axis` deliberately omits `cache=True` — it takes a function (`np.max`/`np.sum`) as an argument, which Numba can't pickle for its on-disk cache on this Numba version; everything downstream of it keeps `cache=True`.
- **`soft_dtw.py`** — `compute_softdtw_batch_channel`/`compute_softdtw_backward_batch_channel` (`@njit`, vectorized over a `(batch, channel)` leading pair — one Numba call per forward/backward, not one per sample) and `SoftDTWBatch(torch.autograd.Function)`, whose `forward`/`backward` round-trip through `.cpu().numpy()` to call them and hold the shape (soft-DTW) loss.
- **`path_soft_dtw.py`** — `dtw_grad`/`dtw_hessian_prod` (`@njit`) and `PathDTWBatch(torch.autograd.Function)`, which produces the temporal-alignment path via a hand-derived Hessian-vector-product backward — this is what makes `loss_temporal` actually differentiable.
- **`dilate_loss.py`** — glue: computes `D` (our own direct squared-difference formula, not the reference's norm-expansion/`.double()` variant), adds a dummy `channel=1` axis to match the ported kernels' `(batch, channel, N, N)` shape, calls `SoftDTWBatch.apply`/`PathDTWBatch.apply`, returns `(loss, loss_shape, loss_temporal)` — same signature as before the port.

**No `torch.compile` support, intentionally.** `torch.compile` cannot safely trace into `SoftDTWBatch`/`PathDTWBatch` — Dynamo tries to trace their `.cpu().numpy()`/Numba internals and crashes. Since Numba JIT already makes DILATE fast in eager mode, `DILATE` takes no `use_torch_compile` parameter and `Forecaster`'s `use_torch_compile=True` only ever compiles the model, never the loss.

---

## `src/forecaster/forecaster.py` — `Forecaster`

Training orchestrator, and the single place every settings-then-bind component actually gets bound. Takes `DataSource` lists, an unbound model, a list of unbound embeddings, a real loss/score/criteria, and settings-only optimizer/regularizer objects; handles normalization, binding, training, validation, checkpointing, and inference.

**Key constructor parameters:**

| Parameter | Description |
|---|---|
| `model` | An **unbound** model instance (`LSTMHistoric(hidden_size=..., num_layers=..., dropout_rate=...)`, etc.) — constructed with only its own architecture hyperparameters. `Forecaster` calls `.bind(...)` on it internally. |
| `embeddings` | `list[EmbeddingNetwork]`, default `[]` — unbound. `Forecaster` binds each one to the actual `historic_cols`/`future_cols`, then owns applying them to every batch (see below). Never passed to the model. |
| `loss`, `validation_score`, `validation_logging_criteria` | Real metric instances (`QuantileLoss(q=[...])`, `MAE()`, ...), not strings. `validation_score` defaults to `MAE()`, `validation_logging_criteria` defaults to `[MAE()]` if not given. |
| `optimizer` | A settings object from `src/optim.py` (`AdamW(lr=0.001, weight_decay=0.01)`), default `Adam(lr=0.001)` if not given. `self.learning_rate` is read off it (`optimizer.lr`) — there's no separate `learning_rate=` argument. |
| `embedding_regularization`, `model_regularizer` | Two independent regularizer slots (`L1Regularizer`/`L2Regularizer`/`RLN` instances, or `None`) — see `regularization.py` above. |
| `target_col` | `str \| None`, default `'y'` — column name for the target across all sources. Lives here, not on `DataSource` (one Forecaster = one target). Saved/restored in the checkpoint. |
| `num_workers` | DataLoader worker processes (default `0`) |
| `prefetch_factor` | Prefetch factor; applied only when `num_workers > 0` (default `2`) |
| `pin_memory` | DataLoader pin memory (default `False`) |
| `drop_last` | Drop last incomplete batch (default `False`) |
| `shuffle` | Shuffle training loader; validation always `False` (default `True`) |
| `norm_stats` | `dict \| None` — reuse pre-computed stats instead of calling `compute_norm_stats` (used internally by `load_model` on resume, so datasets stay consistent with what the restored model/optimizer were trained against) |

**Init flow (`__init__`):**
1. `assert_differentiable(loss)`; read `output_kind`/`output_levels`/`n_outputs = getattr(loss, ..., default)` off the resolved `loss` object — plain losses like `MAE()` are unaffected (`output_kind='point'`, `n_outputs=1`).
2. `compute_norm_stats(training sources only)` → `norm_stats`, unless passed in directly.
3. Group-reweighted loss wiring — if `getattr(loss, 'needs_group_idx', False)`: builds `group_std` from `norm_stats['group_target_std']` keyed by each `training_datasets` entry's `group_key`, in list order (matching `TimeSeriesDataset`'s `group_idx` assignment); `loss.bind(group_std)`.
4. `TimeSeriesDataset(training_datasets, ...)` → training dataset; one per validation source → validation datasets; wrapped in `DataLoader`s.
5. Every entry in `embeddings` gets `.bind(historic_cols, future_cols)` called (raises `ValueError` first if any two share a `name`).
6. `Forecaster` combines the bound embeddings into its own historic/future feature pipeline: for each side, `_prepare_side(embeddings, cols, idx_attr)` collects the nets applicable to that side plus the passthrough column indices (columns claimed by no embedding), and computes the resulting feature width (`sum(embedding_dim) + len(passthrough)`). `_embed(x_h, x_f)` — called on every batch, in training/eval/predict — slices, applies each net, and concatenates with passthrough at call time (`_embed_side`). This logic lives directly in `Forecaster`, not as an exported class.
7. `model.bind(historic_input_size=..., future_input_size=..., forecasting_horizon=..., n_outputs=..., output_kind=..., output_levels=...)` — the model's real layers get built now.
8. `model` and every embedding net move to `device`/`dtype`. `self._model_forward = torch.compile(self.model, dynamic=True) if use_torch_compile else self.model` — compile scope stays model-only; embeddings always run eager, and `DILATE` never compiles either way.
9. **Parameter pools, five groups:** `embedding_weights`/`embedding_biases` (from every bound embedding's `named_parameters()`, split by `dim() > 1`); `backbone_weights`/`backbone_biases`/`head_params` (from `model.named_parameters()`, split first by the `output_head.` name prefix, then by `dim()`). `self._embedding_weights`/`self._backbone_weights` are kept for the regularizer step; biases and the head are always pinned to `weight_decay=0.0` in their `param_groups` dict regardless of what the optimizer settings object specifies for the weight groups.
10. `self.optimizer = optimizer.bind(param_groups)`; `self.learning_rate = optimizer.lr`.
11. `self.embedding_regularizer`/`self.model_regularizer` — if given, each `.bind(their own weight pool, self.learning_rate)`. Head is never regularized (excluded from both pools by construction).
12. `self.validation_logging_criteria = [loss, validation_score] + validation_logging_criteria`; every entry gets `bind_levels(output_kind, output_levels)` called on it, unconditionally.
13. Structured log arrays (`loss_log`, `val_log`, `val_log_aggregated`) — same structured-numpy-array design as before (see "Logging" below).
14. `save_path` handling — `os.makedirs`, `save_path_best`/`save_path_epoch` templates.

**`__training_step`/`__train_model`/`__evaluate_model`/`predict`** all follow the same batch shape: pull `x_h`/`x_f` (and `y`, `group_idx` where relevant) off the raw batch dict, apply the NaN `valid` mask, then `x_h, x_f = self._embed(x_h, x_f)` before ever calling `self._model_forward(x_h, x_f)`. `__training_step` calls `self.embedding_regularizer.step(self._embedding_weights)`/`self.model_regularizer.step(self._backbone_weights)` right after `loss.backward()` (so they see the plain data gradient) and before `optimizer.step()` — regularizers mutate `p.grad` in place rather than adding anything to the loss graph, so `self.loss_log` is purely the data loss regardless of regularization. `__evaluate_model` never touches either regularizer — every validation metric always reflects the un-regularized value.

`__evaluate_model` computes each of `validation_logging_criteria` one of two ways per validation set: poolable metrics (`getattr(m, 'poolable', False)`) accumulate exact statistics across all of that validation set's batches via `new_accumulators()`/`accumulate()`, finalized once via `pooled_value()`; everything else keeps a per-batch average. Index 0 of `validation_logging_criteria` (the loss) always sees normalized values (matching `__training_step`); the rest see denormalized values unless `eval_on_normalized`. Accumulators are built fresh per validation set per epoch — pooling happens *within* one `DataSource`'s own batches only, never mixing two validation sets together; cross-validation-set combination happens afterward via `val_log_aggregated`'s mean/max/min/median across validation sets.

**Logging:** `self.val_log` / `self.val_log_aggregated['mean'|'max'|'min'|'median']` are **structured numpy arrays** — named fields instead of positional indices, e.g. `val_log_aggregated['mean'][epoch]['val_score']`. Field names come from `_criteria_field_names`: `'loss'` and `'val_score'` are fixed (first two slots), remaining criteria are named by class (`'MAE'`, `'NSE'`, ...), deduplicated with a numeric suffix (`'MAE_2'`) if the same metric type appears more than once. `self._criteria_names` holds the field name list. `self.loss_log` (training loss only, one value per epoch) stays a plain float array.

**`compute_number_of_parameters()`** sums parameters across every bound embedding **and** the model (embeddings no longer live inside the model). **`print_parameter_breakdown()`** builds one row per bound embedding (labeled by `net.name`), then appends `self.model.parameter_breakdown()`'s backbone/head rows, then a combined total row — printed as an aligned table, same rendering as before.

**Checkpointing.** `fit()` saves a bundled dict via `self._checkpoint_dict(epoch)`:

```
{
  'model': {'backbone': <state dict, output_head.* keys excluded>, 'head': <state dict, output_head. prefix stripped>},
  'embeddings': {net.name: net.state_dict() for net in self.embeddings},
  'norm_stats', 'historic_cols', 'future_cols', 'target_col',
  'forecasting_horizon', 'historic_input_sequence_length',
  'n_outputs', 'output_kind', 'output_levels',
  'optimizer_state_dict', 'embedding_regularizer_state', 'model_regularizer_state',
  'best_logged_criterion', 'epoch', 'minimize_validation_score', 'save_aggregation_criterion',
  'criteria_names', 'loss_log', 'val_log', 'val_log_aggregated',
}
```

The model's backbone/head split is done by filtering `model.state_dict()` on the `output_head.` name prefix — no separate submodule needed. Checkpoints hold **learned state only** — no reconstructable hyperparameters, no spec strings of any kind. `load_weights(path=None)` reloads just `ckpt['model']`/`ckpt['embeddings']` into the current in-memory `Forecaster` (same process, e.g. reverting to the best epoch after training).

**`Forecaster.load_model(checkpoint_path, model, embeddings=None, resume=False, loss=None, validation_score=None, validation_logging_criteria=None, optimizer=None, embedding_regularization=None, model_regularizer=None, training_datasets=None, validation_datasets=None, device='cpu', batch_size=512, num_workers=0, pin_memory=False, use_torch_compile=False, learning_rate=None, **forecaster_kwargs)`** — classmethod, three modes. Since checkpoints hold no hyperparameters, every mode requires the caller to supply the same *unbound* component objects they'd hand a fresh `Forecaster(...)` call — resuming looks exactly like constructing a new `Forecaster`, plus `resume=True`.

- `training_datasets=None` → **inference only**. Binds `model`/`embeddings` using the schema and output shape saved in the checkpoint (`ckpt['historic_cols']`/`ckpt['n_outputs']`/etc.), loads weights. No optimizer, loaders, or regularizers — built via `cls.__new__(cls)`, bypassing `__init__`, so calling `.fit()` on the result raises `AttributeError` by design.
- `resume=True` (requires `training_datasets` and every other component) → **continues the same run**. A full `Forecaster(...)` construction (using the checkpoint's saved schema fields), then weights/optimizer state/regularizer state/log history are loaded on top from the checkpoint. `fc._start_epoch = checkpoint['epoch'] + 1`; raises `ValueError` if `number_of_epochs <= start_epoch`. `learning_rate`, if given, overwrites just the restored optimizer's LR (keeps momentum/Adam moment state).
- `resume=False` + `training_datasets` given → **warm start**. A fresh `Forecaster(...)` (fresh optimizer, fresh `norm_stats` computed from whatever `training_datasets` are passed now, fresh logs at epoch 0), with only weights loaded as an initialization. `historic_cols`/etc. must be supplied via `forecaster_kwargs` here, same as a plain `Forecaster(...)` call.

`use_torch_compile` is applied *after* weights are loaded (matching `load_weights`'s existing behavior) — compiling is a runtime choice for wherever you're loading, not a fact saved in the checkpoint.

**`predict(x_test, batch_size=None, denormalize=True, device=None, num_workers=0, pin_memory=False, epoch=None)`** — `x_test` is a `DataSource` or `list[DataSource]`. Builds a `TimeSeriesDataset` internally using `self.historic_cols`/`self.future_cols`/`self.historic_input_sequence_length`/`self.forecasting_horizon`/`self.norm_stats`, always with `target_col=None`. `device`/`num_workers`/`pin_memory` are independent of whatever training used. `epoch`, if given, calls `self.load_weights(path=self.save_path_epoch.format(epoch=epoch))` before predicting — a **permanent** weight swap into `self.model`/`self.embeddings`, not scoped to the one `predict()` call. No existence check on the checkpoint file — a missing epoch crashes naturally, by design.

---

## `examples/`

Four reference notebooks demonstrate the API end to end against `data/data.csv`: `example_model_and_loss_comparison.ipynb` (four `Forecaster`s comparing `LSTMHistoric`/`LSTMEncoderDecoder` × `MAE()`/`DILATE()`), `example_regularization_rln.ipynb` (`RLN` vs. no regularizer via `model_regularizer=`), `example_probabilistic_forecasting.ipynb` (`QuantileLoss`/`ExpectileLoss` and the quantile-aware scores), and `example_embeddings.ipynb` (the `EmbeddingNetwork` mechanism, including a shared historic+future group). All four construct every component directly — model, embeddings, loss, optimizer, regularizers — and pass real instances into `Forecaster(...)`, following the patterns documented above. See each notebook for the exact construction calls.
