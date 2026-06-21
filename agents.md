# Codebase Overview

## Structure

```
src/
  data/
    preprocessing.py        — windowing + NaN filtering, produces TensorDatasets
  forecaster/
    forecaster.py           — training loop, normalization, logging, checkpointing
  models/
    LSTM.py                 — LSTMHistoric and LSTMEncoderDecoder
  utils/
    scores_and_losses.py    — metric classes + resolve_metric
    DILATE/
      soft_dtw.py           — pure-PyTorch soft-DTW forward/backward DP
      dilate_loss.py        — DILATE loss combining shape + temporal terms
      __init__.py
```

---

## `src/data/preprocessing.py` — `TimeSeriesDataPreprocessor`

Converts a list of DataFrames into a `TensorDataset(x_h, x_f, y)` using a sliding window.

**Constructor args:** `historic_input_sequence_length`, `forecasting_horizon`, `historic_cols`, `future_cols`, `target_col`, `dtype`

**`input_dfs_to_tensordataset(df_list)`**
- Selects relevant columns from each DataFrame
- Builds windows via `torch.unfold` + `permute`:
  - `x_h`: `(samples, seq_len, n_historic_features)`
  - `x_f`: `(samples, horizon, n_future_features)`
  - `y`:   `(samples, horizon, 1)`
- Filters out samples where `x_h` or `x_f` contain NaN (y NaN is allowed — masked in loss)
- Concatenates all DataFrames and returns one `TensorDataset`

---

## `src/forecaster/forecaster.py` — `Forecaster`

The main training orchestrator. Takes raw DataFrames, handles normalization, training, validation, checkpointing, and inference.

**Key constructor parameters:**
| Parameter | Description |
|---|---|
| `model` | Model class (not instance) |
| `model_config` | Dict of architecture hyperparameters; Forecaster injects `historic_cols`, `future_cols`, `forecasting_horizon`, `historic_input_sequence_length` |
| `dfs_training_sets` | List of DataFrames for training |
| `dfs_validation_sets` | List of DataFrames for validation (one loader each) |
| `historic_cols` / `future_cols` | Column names for x_h / x_f |
| `target_col` | Target column name |
| `loss_function` | String or metric instance used as training loss (must be differentiable) |
| `validation_score` | String or metric instance used for best-model selection |
| `validation_logging_criteria` | Additional metrics to log per epoch |
| `minimize_validation_score` | Whether lower = better for val score |
| `save_aggregation_criterion` | How to aggregate across validation sets: `'mean'`, `'max'`, `'min'`, `'median'` |
| `save_weights_every_n_epochs` | Save checkpoint every N epochs |
| `use_torch_compile` | Wrap model and DILATE with `torch.compile(dynamic=True)` |
| `optimizer` | String name of `torch.optim` optimizer |

**Normalization:** Per-column mean/std computed over ALL datasets (train + val combined) so stats cover the full value range. Target mean/std computed ignoring NaN. Applied before creating DataLoaders; denormalization applied in eval for non-loss metrics.

**Training loop (`fit`):**
- Each epoch: `__train_model` → `__evaluate_model` → save best / checkpoint → print
- NaN filtering per batch: samples where any y timestep is NaN are skipped; batch count tracks only contributing batches
- `loss_log`: `(n_epochs,)` — mean training loss per epoch
- `val_log`: `(n_epochs, n_val_sets, n_criteria)` — per-set, per-metric
- `val_log_aggregated`: dict of `{mean, max, min, median}` over validation sets

**Evaluation:** `metrics[0]` (loss) computed on normalized data; `metrics[1:]` computed on denormalized data, except metrics with `eval_on_normalized=True` (currently DILATE) which stay normalized to avoid soft-DTW overflow on large values.

**Checkpointing:**
- Best model saved to `save_path/name/model_{name}_best.pt`
- Epoch checkpoints saved to `save_path/name/model_{name}_epoch_{n}.pt`
- `load_weights(path=None)`: loads best checkpoint by default; strips `_orig_mod.` prefix from compiled-model state dicts

**`predict(x_test, batch_size, denormalize)`:** Runs inference over a TensorDataset; denormalizes output by default.

---

## `src/models/LSTM.py`

Both models receive a single `config` dict. The Forecaster injects `historic_cols`, `future_cols`, `forecasting_horizon`, and `historic_input_sequence_length` into this dict automatically.

**`LSTMHistoric`**
- Encodes only `x_h` (historic features)
- Single LSTM → takes last hidden state → linear layer → `(B, horizon, 1)`
- No future features used

**`LSTMEncoderDecoder`**
- Encoder LSTM processes `x_h` → hidden state
- Hidden state downscaled via a linear layer (`encoder_downscale_layer_size`)
- Downscaled representation repeated over horizon and concatenated with `x_f`
- Decoder LSTM processes the combined sequence → linear output → `(B, horizon, 1)`

Both models receive `X = {'x_h': ..., 'x_f': ...}` as input dict.

---

## `src/utils/scores_and_losses.py`

**`resolve_metric(metric)`**
Accepts a metric instance or a string. Two string forms:
- `'mae'` — no-arg instantiation via registry
- `'DILATE(alpha=0.5, gamma=0.01)'` — parsed safely with `ast`; kwargs forwarded to constructor

**`assert_differentiable(metric)`**
Raises `TypeError` if metric has `differentiable = False` (blocks use as training loss).

**Base class `_Metric(nn.Module)`**
- `dim`: reduction dimension (`None` = scalar mean over all dims)
- `_reduce(x)`: calls `x.mean()` or `x.mean(dim=self.dim)`

**Metric classes:**
| Class | Formula | `differentiable` | `eval_on_normalized` |
|---|---|---|---|
| `MAE` | mean absolute error | True | — |
| `MSE` | mean squared error | True | — |
| `RMSE` | root mean squared error | True | — |
| `MAPE` | mean absolute percentage error | True | — |
| `SMAPE` | symmetric MAPE | True | — |
| `DILATE` | alpha × shape_loss + (1-alpha) × temporal_loss | True | True |

**`REGISTRY`:** `{'mae': MAE, 'mse': MSE, 'rmse': RMSE, 'mape': MAPE, 'smape': SMAPE, 'dilate': DILATE}`

---

## `src/utils/DILATE/`

Implementation of DILATE (Le Guen & Thome, NeurIPS 2019) in pure PyTorch.

**`soft_dtw.py`**
- `soft_dtw_value(D, gamma)`: forward DP over `(B, N, M)` distance matrix; fully differentiable via autograd; returns soft-DTW value and the DP table `R`
- `soft_dtw_alignment(D, R, gamma)`: backward DP (no_grad); returns the soft alignment path `E` of shape `(B, N, M)`

**`dilate_loss.py` — `dilate_loss(outputs, targets, alpha, gamma, device)`**
- Inputs: `(B, N_output, 1)`
- Computes pairwise squared distance matrix `D`
- `loss_shape` = mean soft-DTW value (measures sequence shape similarity)
- `loss_temporal` = alignment path weighted by temporal displacement matrix `Omega`
- Returns `(loss, loss_shape, loss_temporal)`

**`DILATE` class** in `scores_and_losses.py` wraps `dilate_loss`; optionally wraps it with `torch.compile`. Has `eval_on_normalized = True` because soft-DTW overflows numerically on large denormalized values (squared distances become ~1e8, causing underflow in `exp(-D/gamma)` with small gamma).

---

## `check.ipynb`

Reference notebook demonstrating a full training run:
1. Load `data/data.csv`, replace `-999` with `NaN`, split into train/val/test by date
2. Inspect preprocessor output shapes
3. Train `LSTMHistoric` (16 hidden, 1 layer) for 5 epochs with MAE loss, RMSE val score
4. Train `LSTMEncoderDecoder` (16 hidden, 1 layer, downscale 8) same settings
