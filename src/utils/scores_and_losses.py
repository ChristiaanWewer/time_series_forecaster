import ast
import torch
import torch.nn as nn


REGISTRY: dict = {}  # populated after class definitions


def resolve_metric(metric):
    """Accept a metric instance or a name string and return an instance.

    Supported string forms:
        'mae'                           — no-arg instantiation
        'DILATE(alpha=0.5, gamma=0.01)' — kwargs parsed safely via ast
    """
    if not isinstance(metric, str):
        return metric
    if '(' in metric:
        call = ast.parse(metric.strip(), mode='eval').body
        name = call.func.id.lower()
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    else:
        name, kwargs = metric.strip().lower(), {}
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
    """
    differentiable: bool = True

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def _reduce(self, x):
        return x.mean() if self.dim is None else x.mean(dim=self.dim)


class MAE(_Metric):
    def forward(self, y_pred, y_true):
        return self._reduce(torch.abs(y_pred - y_true))


class MSE(_Metric):
    def forward(self, y_pred, y_true):
        return self._reduce((y_pred - y_true).pow(2))


class RMSE(_Metric):
    def forward(self, y_pred, y_true):
        sq = (y_pred - y_true).pow(2)
        if self.dim is None:
            return sq.mean().sqrt()
        return sq.mean(dim=self.dim).sqrt()


class MAPE(_Metric):
    def forward(self, y_pred, y_true):
        return self._reduce(
            100.0 * torch.abs(y_pred - y_true) / torch.abs(y_true).clamp(min=1e-8)
        )


class SMAPE(_Metric):
    def forward(self, y_pred, y_true):
        denom = (torch.abs(y_pred) + torch.abs(y_true)).clamp(min=1e-8)
        return self._reduce(200.0 * torch.abs(y_pred - y_true) / denom)


class DILATE(_Metric):
    """
    DILATE loss: alpha * shape_loss + (1 - alpha) * temporal_loss.
    Le Guen & Thome, "Shape and Time Distortion Loss", NeurIPS 2019.

    Args:
        alpha: Weight between shape [1.0] and temporal [0.0] loss.
        gamma: Soft-DTW smoothing parameter (smaller = closer to hard DTW).
        use_torch_compile: Wrap the loss function with torch.compile (PyTorch 2.0+).
            Defaults to False — Forecaster.use_torch_compile is the single source of
            truth for whether DILATE compiles (see Forecaster.__init__'s compile-sharing
            block, which overrides this via `_dilate_fn` when it's True). A bare DILATE()
            should not silently self-compile independent of that.
    """
    differentiable = True
    eval_on_normalized = True  # soft-DTW overflows on large denormalized values

    def __init__(self, alpha=0.5, gamma=0.01, use_torch_compile=False):
        super().__init__(dim=None)
        self.alpha = alpha
        self.gamma = gamma
        from src.utils.DILATE.dilate_loss import dilate_loss as _fn
        self._dilate_fn = torch.compile(_fn, dynamic=True) if use_torch_compile else _fn

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

    Note: computed on whatever tensor is passed in, using that tensor's own mean as the
    reference. Forecaster's evaluation loop calls metrics per batch and averages the
    results, so validation_score='nse' reports the mean of per-batch NSE values, not the
    single NSE of the full validation set computed at once.
    """
    differentiable = False

    def forward(self, y_pred, y_true):
        numerator = (y_true - y_pred).pow(2).sum()
        denominator = (y_true - y_true.mean()).pow(2).sum()
        return 1 - numerator / denominator.clamp(min=1e-8)


class AlphaNSE(_Metric):
    """
    Variability ratio: std(pred) / std(obs). The "alpha" term from KGE (Gupta et al. 2009),
    reported standalone here. 1.0 = model reproduces observed variability exactly;
    < 1 under-predicts variability (over-smooths); > 1 over-predicts it.

    Same per-batch-averaging caveat as NSE applies. differentiable=False for the same
    reason: it is not a lower-is-better loss.
    """
    differentiable = False

    def forward(self, y_pred, y_true):
        return y_pred.std() / y_true.std().clamp(min=1e-8)


REGISTRY = {
    'mae': MAE,
    'mse': MSE,
    'rmse': RMSE,
    'mape': MAPE,
    'smape': SMAPE,
    'dilate': DILATE,
    'nse': NSE,
    'alpha_nse': AlphaNSE,
}
