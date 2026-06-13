import torch
import torch.nn as nn


REGISTRY: dict = {}  # populated after class definitions


def resolve_metric(metric):
    """Accept a metric instance or a name string (e.g. 'mae', 'dilate') and return an instance."""
    if isinstance(metric, str):
        key = metric.lower()
        if key not in REGISTRY:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(REGISTRY)}")
        return REGISTRY[key]()
    return metric


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
    """
    differentiable = True

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


REGISTRY = {
    'mae': MAE,
    'mse': MSE,
    'rmse': RMSE,
    'mape': MAPE,
    'smape': SMAPE,
    'dilate': DILATE,
}
