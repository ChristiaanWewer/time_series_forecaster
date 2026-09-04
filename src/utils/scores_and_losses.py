import ast
import torch
import torch.nn as nn

from src.data.normalization import _merge_stats


REGISTRY: dict = {}  # populated after class definitions


class ChanAccumulator:
    """Streaming (n, mean, M2) accumulator via Chan's parallel algorithm — merges
    per-batch statistics into an exact running estimate without holding all data in
    memory. Reduces to a plain (sample-weighted) running mean when only `.mean` is used;
    `.variance`/`.M2` additionally give the exact pooled sum-of-squares. Used by
    poolable metrics (see `_Metric.poolable`) so Forecaster's evaluation loop can compute
    metrics like NSE/RMSE exactly over a whole validation set instead of averaging
    already-reduced per-batch values.
    """
    def __init__(self):
        self.n = 0.0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.reshape(-1)
        n_b = float(values.numel())
        if n_b == 0:
            return
        mean_b = values.mean().item()
        M2_b = ((values - mean_b) ** 2).sum().item()
        self.n, self.mean, self.M2 = _merge_stats(self.n, self.mean, self.M2, n_b, mean_b, M2_b)

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else float('nan')


def parse_spec(spec: str) -> tuple[str, dict]:
    """Parse 'name' or 'name(kw=val, ...)' into (name.lower(), kwargs), kwargs values via
    ast.literal_eval. Shared grammar for resolve_metric and Forecaster's optimizer spec.
    """
    spec = spec.strip()
    if '(' in spec:
        call = ast.parse(spec, mode='eval').body
        return call.func.id.lower(), {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return spec.lower(), {}


def resolve_metric(metric):
    """Accept a metric instance or a name string and return an instance.

    Supported string forms:
        'mae'                           — no-arg instantiation
        'DILATE(alpha=0.5, gamma=0.01)' — kwargs parsed safely via ast
    """
    if not isinstance(metric, str):
        return metric
    name, kwargs = parse_spec(metric)
    if name not in REGISTRY:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def metric_to_spec(metric) -> str:
    """Inverse of resolve_metric: a metric instance -> a string resolve_metric can parse
    back into an equivalent instance. Used to make a metric's configuration checkpointable
    without pickling the live nn.Module — a compiled DILATE holds a torch.compile closure
    that isn't reliably picklable across processes/machines.
    """
    if isinstance(metric, DILATE):
        return f'DILATE(alpha={metric.alpha}, gamma={metric.gamma})'
    return type(metric).__name__.lower()


def assert_differentiable(metric):
    """Raise TypeError if metric cannot be used as a training loss."""
    if not getattr(metric, 'differentiable', True):
        raise TypeError(
            f"{type(metric).__name__} is not differentiable and cannot be used as a training loss."
        )


class _Metric(nn.Module):
    """
    Base class for all metrics.

    Args:
        dim: Dimension to reduce over. None reduces all dims (scalar output).
             E.g. dim=0 gives per-timestep result, dim=1 gives per-sample result.

    Subclasses set differentiable=False when they have no valid gradient
    (assert_differentiable will then block their use as a training loss).

    Subclasses set poolable=True and override new_accumulators/accumulate/pooled_value
    to support exact epoch-level computation across batches (see ChanAccumulator) instead
    of Forecaster's default of averaging each batch's forward() value. forward() itself is
    unaffected either way — poolable is purely an additional, opt-in code path used only by
    Forecaster's evaluation loop.
    """
    differentiable: bool = True
    poolable: bool = False

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def _reduce(self, x):
        return x.mean() if self.dim is None else x.mean(dim=self.dim)

    def new_accumulators(self) -> dict:
        """Fresh named ChanAccumulator(s) this metric needs for pooled computation."""
        return {}

    def accumulate(self, accs: dict, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update this metric's accumulators with one batch's values."""
        pass

    def pooled_value(self, accs: dict) -> float:
        """Compute the final epoch-level value from this metric's accumulated statistics."""
        raise NotImplementedError


class MAE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        return self._reduce(torch.abs(y_pred - y_true))

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update(torch.abs(y_pred - y_true))

    def pooled_value(self, accs):
        return accs['val'].mean


class MSE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        return self._reduce((y_pred - y_true).pow(2))

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update((y_pred - y_true).pow(2))

    def pooled_value(self, accs):
        return accs['val'].mean


class RMSE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        sq = (y_pred - y_true).pow(2)
        if self.dim is None:
            return sq.mean().sqrt()
        return sq.mean(dim=self.dim).sqrt()

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update((y_pred - y_true).pow(2))

    def pooled_value(self, accs):
        return accs['val'].mean ** 0.5


class MAPE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        return self._reduce(
            100.0 * torch.abs(y_pred - y_true) / torch.abs(y_true).clamp(min=1e-8)
        )

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update(100.0 * torch.abs(y_pred - y_true) / torch.abs(y_true).clamp(min=1e-8))

    def pooled_value(self, accs):
        return accs['val'].mean


class SMAPE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        denom = (torch.abs(y_pred) + torch.abs(y_true)).clamp(min=1e-8)
        return self._reduce(200.0 * torch.abs(y_pred - y_true) / denom)

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        denom = (torch.abs(y_pred) + torch.abs(y_true)).clamp(min=1e-8)
        accs['val'].update(200.0 * torch.abs(y_pred - y_true) / denom)

    def pooled_value(self, accs):
        return accs['val'].mean


class DILATE(_Metric):
    """
    DILATE loss: alpha * shape_loss + (1 - alpha) * temporal_loss.
    Le Guen & Thome, "Shape and Time Distortion Loss", NeurIPS 2019.

    Args:
        alpha: Weight between shape [1.0] and temporal [0.0] loss.
        gamma: Soft-DTW smoothing parameter (smaller = closer to hard DTW).
    """
    differentiable = True
    eval_on_normalized = True  # soft-DTW overflows on large denormalized values

    def __init__(self, alpha=0.5, gamma=0.01):
        super().__init__(dim=None)
        self.alpha = alpha
        self.gamma = gamma
        from src.utils.DILATE.dilate_loss import dilate_loss as _fn
        self._dilate_fn = _fn

    def forward(self, y_pred, y_true):
        # dilate_loss expects (batch_size, N_output, 1)
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(-1)
            y_true = y_true.unsqueeze(-1)
        loss, _, _ = self._dilate_fn(y_pred, y_true, self.alpha, self.gamma, y_pred.device)
        return loss


class NSE(_Metric):
    """
    Nash-Sutcliffe Efficiency. 1.0 = perfect fit, 0.0 = as good as predicting the mean,
    negative = worse than predicting the mean.

    NSE is higher-is-better, so minimizing it directly would push a model the wrong way —
    it is blocked from use as a training loss (differentiable=False).

    Note: forward() is computed on whatever tensor is passed in, using that tensor's own
    mean as the reference — call it on a full pooled array (as check.ipynb's evaluation
    cells do) for the exact dataset-level NSE. Forecaster's evaluation loop instead uses
    the poolable path below (accumulating exact statistics across batches via
    ChanAccumulator), since naively averaging per-batch NSE values is a biased estimate of
    the pooled NSE — the denominator uses each batch's own local mean, so a low-variance
    batch (e.g. a baseflow-only window) can produce a wildly distorted per-batch value.
    """
    differentiable = False
    poolable = True

    def forward(self, y_pred, y_true):
        numerator = (y_true - y_pred).pow(2).sum()
        denominator = (y_true - y_true.mean()).pow(2).sum()
        return 1 - numerator / denominator.clamp(min=1e-8)

    def new_accumulators(self):
        return {'residual_sq': ChanAccumulator(), 'obs': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['residual_sq'].update((y_true - y_pred).pow(2))
        accs['obs'].update(y_true)

    def pooled_value(self, accs):
        denom = accs['obs'].variance
        denom = denom if denom > 1e-8 else 1e-8
        return 1 - accs['residual_sq'].mean / denom


class AlphaNSE(_Metric):
    """
    Variability ratio: std(pred) / std(obs). The "alpha" term from the NSE/KGE
    decomposition (Gupta et al. 2009) — matches neuralhydrology's alpha_nse (verified
    against its source, sim.std()/obs.std()), the same metric used in Kratzert et al.'s
    HESS papers. 1.0 = model reproduces observed variability exactly; < 1 under-predicts
    variability (over-smooths); > 1 over-predicts it.

    Same per-batch-averaging caveat as NSE applies (see NSE's docstring); fixed the same
    way via the poolable path. differentiable=False for the same reason: it is not a
    lower-is-better loss.

    Uses population variance (ddof=0, matching numpy/xarray's default and hence
    neuralhydrology's reference) rather than torch.std()'s default ddof=1 — for AlphaNSE
    this is a ratio of two same-n variances so the Bessel correction would cancel either
    way, but written explicitly here to stay consistent with BetaNSE and pooled_value.
    """
    differentiable = False
    poolable = True

    def forward(self, y_pred, y_true):
        denom = y_true.var(unbiased=False).clamp(min=1e-8).sqrt()
        return y_pred.var(unbiased=False).sqrt() / denom

    def new_accumulators(self):
        return {'pred': ChanAccumulator(), 'obs': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['pred'].update(y_pred)
        accs['obs'].update(y_true)

    def pooled_value(self, accs):
        denom = accs['obs'].variance
        denom = denom if denom > 1e-8 else 1e-8
        return (accs['pred'].variance / denom) ** 0.5


class BetaNSE(_Metric):
    """
    Bias term from the same NSE/KGE decomposition (Gupta et al. 2009) AlphaNSE
    implements — matches neuralhydrology's beta_nse (verified against its source):
    (mean(pred) - mean(obs)) / std(obs). 0.0 = no bias; positive = over-prediction of the
    mean; negative = under-prediction.

    differentiable=False: the ideal value is 0, not an extreme, so directly minimizing or
    maximizing it would push the model toward *more* bias rather than less — it isn't a
    valid loss shape, same reasoning as AlphaNSE/NSE. Same per-batch-averaging caveat as
    NSE applies, fixed the same way via the poolable path.

    Uses population variance (ddof=0, matching numpy/xarray's default and hence
    neuralhydrology's reference beta_nse) rather than torch.std()'s default ddof=1 — unlike
    AlphaNSE this is NOT a ratio that cancels the Bessel correction, so the convention here
    actually changes the numeric result and must match the reference.
    """
    differentiable = False
    poolable = True

    def forward(self, y_pred, y_true):
        denom = y_true.var(unbiased=False).clamp(min=1e-8).sqrt()
        return (y_pred.mean() - y_true.mean()) / denom

    def new_accumulators(self):
        return {'pred': ChanAccumulator(), 'obs': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['pred'].update(y_pred)
        accs['obs'].update(y_true)

    def pooled_value(self, accs):
        denom = accs['obs'].variance
        denom = denom if denom > 1e-8 else 1e-8
        return (accs['pred'].mean - accs['obs'].mean) / (denom ** 0.5)


REGISTRY = {
    'mae': MAE,
    'mse': MSE,
    'rmse': RMSE,
    'mape': MAPE,
    'smape': SMAPE,
    'dilate': DILATE,
    'nse': NSE,
    'alpha_nse': AlphaNSE,
    'beta_nse': BetaNSE,
}
