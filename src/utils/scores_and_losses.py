import math
import torch
import torch.nn as nn

from src.data.normalization import _merge_stats


def _require_bound_levels(metric) -> None:
    """CRPS/SCRPS/RSCRPS iterate self._level_index.items() rather than doing a direct
    self._level_index[level] lookup (unlike WinklerScore/PICP/PINAW) — an empty dict
    (bind_levels never called, or a point-kind head) would otherwise silently select
    zero channels and produce a NaN instead of a loud failure. Called first thing in
    each of their channel-selection methods to keep the same "deliberately uncaught,
    fail loudly" convention every other level-dependent metric already gets for free
    from a direct dict lookup.
    """
    if not metric._level_index:
        raise KeyError(
            f"{type(metric).__name__} requires a bound quantile/expectile output head "
            "— self._level_index is empty (bind_levels was never called, or the "
            "model's output head has no levels)"
        )


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

    Probabilistic output heads (see QuantileLoss/ExpectileLoss below): a loss function
    can declare output_kind/output_levels/n_outputs to have Forecaster build a
    multi-channel output head (src/models/heads.py) sized for it. Every other resolved
    metric (validation_score, validation_logging_criteria) gets bind_levels(kind,
    levels) called on it once, unconditionally, by Forecaster right after — a no-op for
    metrics that never reference self._level_index. Metrics that need a *specific*
    channel (WinklerScore, PICP, PINAW, CRPS) look it up via self._level_index[level],
    letting a plain KeyError propagate if that level isn't among the head's
    output_levels (or the head has no levels at all, e.g. a plain point head) — no
    graceful fallback, by design. bind_levels accepts both output_kind='quantile' and
    'expectile' heads — the interval/coverage/width formulas (Winkler/PICP/PINAW/CRPS)
    are only exactly correct for true quantiles, but reusing them against expectile
    levels as an approximation is a deliberate, accepted trade-off (channel lookup by
    level doesn't care which kind produced the level). Ordinary point-style metrics
    (MAE, NSE, ...) call _select_point() to default to the level-0.5 channel when
    y_pred has more than one channel (the median under a quantile head, the mean under
    an expectile head), crashing the same way if there's no literal 0.5 level to read.
    """
    differentiable: bool = True
    poolable: bool = False
    output_kind: str = 'point'
    output_levels: list | None = None
    n_outputs: int = 1

    # Group-reweighting (see MAE/MSE below): a subclass opts in purely by overriding
    # _error() and declaring _reweight_power. None means "doesn't support it" — passing
    # reweight=True to a metric that hasn't overridden _error raises in __init__ below.
    # The exponent is tied to the error's own power (2 for squared error/variance, 1 for
    # absolute error/std) so the reweighted term stays dimensionless regardless of a
    # series' scale — see the group_mse/group_mae discussion this implements.
    _reweight_power: int | None = None
    needs_group_idx: bool = False

    def __init__(self, dim=None, reweight=False, eps=0.1):
        super().__init__()
        self.dim = dim
        self._level_index: dict = {}
        self.reweight = reweight
        self.eps = eps
        # Only metrics that implement _error() (currently MAE/MSE) participate in this
        # dispatch — everything else keeps its own class-level forward() untouched.
        supports_dispatch = type(self)._error is not _Metric._error
        if reweight:
            if not supports_dispatch or self._reweight_power is None:
                raise ValueError(f"{type(self).__name__} does not support reweight=True")
            self.needs_group_idx = True
            self.register_buffer('group_std', torch.empty(0))
            # Picked once, here, rather than branched on inside forward() every batch —
            # an instance attribute shadows the class's forward() method, so every
            # subsequent call dispatches straight to the right implementation with no
            # per-call conditional.
            self.forward = self._reweighted_forward
        elif supports_dispatch:
            self.forward = self._plain_forward

    def bind(self, group_std: torch.Tensor) -> None:
        """Attach the precomputed per-series std this instance reweights by (raw,
        pre-normalization units) — called once by Forecaster, after construction,
        the same way RLN.bind/bind_levels attach data-dependent state that can't be
        known until after this instance is constructed.
        """
        self.register_buffer('group_std', group_std)

    def _error(self, y_pred, y_true):
        """Unweighted elementwise error, in this metric's own output shape (point
        metrics call _select_point themselves here; multi-channel losses like
        QuantileLoss/ExpectileLoss keep every channel). Overriding this (instead of
        forward() directly) is what opts a metric into the reweight=True mechanism
        above."""
        raise NotImplementedError

    def _plain_forward(self, y_pred, y_true, group_idx=None):
        return self._reduce(self._error(y_pred, y_true))

    def _reweighted_forward(self, y_pred, y_true, group_idx):
        err = self._error(y_pred, y_true)
        weight = 1.0 / (self.group_std[group_idx] + self.eps) ** self._reweight_power
        weight = weight.view(-1, *([1] * (err.dim() - 1)))
        return self._reduce(err * weight)

    def _reduce(self, x):
        return x.mean() if self.dim is None else x.mean(dim=self.dim)

    def bind_levels(self, kind: str, levels: list | None) -> None:
        """Build the {level: channel_index} lookup this metric's forward()/accumulate()
        may consult. Called by Forecaster on every validation criterion after resolving
        the loss function's own output_kind/output_levels — kind/levels describe the
        model's actual output head, not this metric's own configuration. Populated for
        both 'quantile' and 'expectile' heads (empty for a plain 'point' head) — level
        lookup by index doesn't care which kind produced the level; see the class
        docstring for the accepted approximation this implies for expectile heads.
        """
        self._level_index = (
            {round(l, 6): i for i, l in enumerate(levels)}
            if kind in ('quantile', 'expectile') and levels else {}
        )

    def _select_point(self, y_pred: torch.Tensor) -> torch.Tensor:
        """For point-style metrics: passes y_pred through unchanged if it already has
        one output channel; otherwise selects the level-0.5 channel (the median under a
        quantile head, the mean under an expectile head). Lets this metric be used as a
        validation criterion even when the model's output head is multi-channel
        (quantile_loss/expectile_loss) — KeyError if 0.5 isn't one of the head's levels,
        deliberately uncaught.
        """
        if y_pred.shape[-1] == 1:
            return y_pred
        idx = self._level_index[0.5]
        return y_pred[..., idx:idx + 1]

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
    """Mean absolute error. reweight=True (see _Metric) weights each sample's error by
    1/(group_std+eps) — std, matching absolute error's own [y]^1 units — using the
    per-series std bound via bind(); see group_target_std (src/data/normalization.py)
    and Forecaster's wiring of it.
    """
    poolable = True
    _reweight_power = 1

    def _error(self, y_pred, y_true):
        return torch.abs(self._select_point(y_pred) - y_true)

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        accs['val'].update(torch.abs(y_pred - y_true))

    def pooled_value(self, accs):
        return accs['val'].mean


class MSE(_Metric):
    """Mean squared error. reweight=True (see _Metric) weights each sample's error by
    1/(group_std+eps)^2 — variance, matching squared error's own [y]^2 units — using the
    per-series std bound via bind(); see group_target_std (src/data/normalization.py)
    and Forecaster's wiring of it. Matches neuralhydrology's MaskedNSELoss formula
    exactly (verified against its source) when applied on top of this codebase's
    existing globally-normalized target.
    """
    poolable = True
    _reweight_power = 2

    def _error(self, y_pred, y_true):
        return (self._select_point(y_pred) - y_true).pow(2)

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        accs['val'].update((y_pred - y_true).pow(2))

    def pooled_value(self, accs):
        return accs['val'].mean


class RMSE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        sq = (y_pred - y_true).pow(2)
        if self.dim is None:
            return sq.mean().sqrt()
        return sq.mean(dim=self.dim).sqrt()

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        accs['val'].update((y_pred - y_true).pow(2))

    def pooled_value(self, accs):
        return accs['val'].mean ** 0.5


class MAPE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        return self._reduce(
            100.0 * torch.abs(y_pred - y_true) / torch.abs(y_true).clamp(min=1e-8)
        )

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        accs['val'].update(100.0 * torch.abs(y_pred - y_true) / torch.abs(y_true).clamp(min=1e-8))

    def pooled_value(self, accs):
        return accs['val'].mean


class SMAPE(_Metric):
    poolable = True

    def forward(self, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        denom = (torch.abs(y_pred) + torch.abs(y_true)).clamp(min=1e-8)
        return self._reduce(200.0 * torch.abs(y_pred - y_true) / denom)

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
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
        # after the dim==2 normalization above, y_pred always has a trailing channel
        # dim here — _select_point is a no-op for the legacy 2-D calling convention
        # (single channel already) and only kicks in for a genuinely multi-channel head
        y_pred = self._select_point(y_pred)
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
        y_pred = self._select_point(y_pred)
        numerator = (y_true - y_pred).pow(2).sum()
        denominator = (y_true - y_true.mean()).pow(2).sum()
        return 1 - numerator / denominator.clamp(min=1e-8)

    def new_accumulators(self):
        return {'residual_sq': ChanAccumulator(), 'obs': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
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
        y_pred = self._select_point(y_pred)
        denom = y_true.var(unbiased=False).clamp(min=1e-8).sqrt()
        return y_pred.var(unbiased=False).sqrt() / denom

    def new_accumulators(self):
        return {'pred': ChanAccumulator(), 'obs': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
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
        y_pred = self._select_point(y_pred)
        denom = y_true.var(unbiased=False).clamp(min=1e-8).sqrt()
        return (y_pred.mean() - y_true.mean()) / denom

    def new_accumulators(self):
        return {'pred': ChanAccumulator(), 'obs': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        y_pred = self._select_point(y_pred)
        accs['pred'].update(y_pred)
        accs['obs'].update(y_true)

    def pooled_value(self, accs):
        denom = accs['obs'].variance
        denom = denom if denom > 1e-8 else 1e-8
        return (accs['pred'].mean - accs['obs'].mean) / (denom ** 0.5)


class QuantileLoss(_Metric):
    """
    Pinball (quantile) loss for probabilistic forecasting: one model output channel per
    requested quantile, all trained jointly in a single forward/backward pass.

    output_kind/output_levels/n_outputs tell Forecaster what output head to build (see
    src/models/heads.py) and let quantile-aware evaluation scores (WinklerScore, CRPS,
    PICP, PINAW) look up which channel holds which quantile. Channels are always in
    ascending-quantile order (self.output_levels = sorted(q)), matching the order the
    output head produces them in.

    Args:
        q: quantile levels to train, e.g. (0.025, 0.5, 0.975) for a 95% PI plus median.

    reweight=True (see _Metric) weights each sample's pinball error by 1/(group_std+eps)
    — std, matching pinball loss's own [y]^1 units (it's piecewise-linear in the
    residual, same dimensional character as MAE) — using the per-series std bound via
    bind(); see group_target_std (src/data/normalization.py) and Forecaster's wiring.
    """
    poolable = True
    _reweight_power = 1

    def __init__(self, q=(0.1, 0.5, 0.9), reweight=False, eps=0.1):
        super().__init__(dim=None, reweight=reweight, eps=eps)
        self.output_levels = sorted(q)
        self.output_kind = 'quantile'
        self.n_outputs = len(self.output_levels)
        self._q = torch.tensor(self.output_levels)

    def _error(self, y_pred, y_true):
        q = self._q.to(device=y_pred.device, dtype=y_pred.dtype)
        diff = y_true - y_pred  # (B, H, 1) - (B, H, Q) broadcasts to (B, H, Q)
        return torch.maximum(q * diff, (q - 1) * diff)

    def new_accumulators(self):
        return {l: ChanAccumulator() for l in self.output_levels}

    def accumulate(self, accs, y_pred, y_true):
        elem = self._error(y_pred, y_true)
        for i, l in enumerate(self.output_levels):
            accs[l].update(elem[..., i])

    def pooled_value(self, accs):
        return sum(accs[l].mean for l in self.output_levels) / len(self.output_levels)


class ExpectileLoss(_Metric):
    """
    Expectile loss (Newey & Powell, 1987): asymmetric squared loss, one model output
    channel per requested expectile level, trained jointly like QuantileLoss.

    For residual u = y_true - y_pred and expectile level e:
        L_e(u) = |e - 1{u < 0}| * u^2

    Unlike quantiles, an expectile head has no literal median channel (the 0.5
    expectile is the mean, not the median) — but WinklerScore/PICP/PINAW/CRPS and the
    point-metric level-0.5 auto-select (_select_point) still bind against an expectile
    head's levels the same way they would a quantile head's (see _Metric.bind_levels):
    a deliberate, accepted approximation, since their formulas are only exactly correct
    for true quantiles.

    Args:
        e: expectile levels to train, e.g. (0.025, 0.5, 0.975).

    reweight=True (see _Metric) weights each sample's expectile error by
    1/(group_std+eps)^2 — variance, matching this loss's own [y]^2 units (quadratic in
    the residual, same dimensional character as MSE) — using the per-series std bound
    via bind(); see group_target_std (src/data/normalization.py) and Forecaster's
    wiring.
    """
    poolable = True
    _reweight_power = 2

    def __init__(self, e=(0.1, 0.5, 0.9), reweight=False, eps=0.1):
        super().__init__(dim=None, reweight=reweight, eps=eps)
        self.output_levels = sorted(e)
        self.output_kind = 'expectile'
        self.n_outputs = len(self.output_levels)
        self._e = torch.tensor(self.output_levels)

    def _error(self, y_pred, y_true):
        e = self._e.to(device=y_pred.device, dtype=y_pred.dtype)
        u = y_true - y_pred  # (B, H, 1) - (B, H, E) broadcasts to (B, H, E)
        asym_weight = torch.where(u < 0, 1.0 - e, e)
        return asym_weight * u.pow(2)

    def new_accumulators(self):
        return {l: ChanAccumulator() for l in self.output_levels}

    def accumulate(self, accs, y_pred, y_true):
        elem = self._error(y_pred, y_true)
        for i, l in enumerate(self.output_levels):
            accs[l].update(elem[..., i])

    def pooled_value(self, accs):
        return sum(accs[l].mean for l in self.output_levels) / len(self.output_levels)


class WinklerScore(_Metric):
    """
    Winkler (interval) score for a (1-alpha) central prediction interval — lower-is-
    better. Lower/upper bounds are read from the channels at levels alpha/2 and
    1-alpha/2 (e.g. alpha=0.05 pairs with a q=(0.025, 0.5, 0.975)-trained head for a
    95% PI). Requires the model's output head to be quantile- or expectile-based with
    those exact levels present — self._level_index[level] raises a plain KeyError
    otherwise (see _Metric.bind_levels), by design. Exact for a quantile head;
    accepted as an approximation for an expectile head (see ExpectileLoss).

    For observation y, interval [lower, upper], width = upper - lower:
        score = width                                   if lower <= y <= upper
        score = width + (2/alpha) * (lower - y)          if y < lower
        score = width + (2/alpha) * (y - upper)          if y > upper

    Not usable as a training loss (differentiable=False) — evaluation only, like NSE.
    """
    differentiable = False
    poolable = True

    def __init__(self, alpha):
        super().__init__(dim=None)
        self.alpha = alpha
        self._lo_level = round(alpha / 2, 6)
        self._hi_level = round(1 - alpha / 2, 6)

    def _bounds(self, y_pred):
        lower = y_pred[..., self._level_index[self._lo_level]]
        upper = y_pred[..., self._level_index[self._hi_level]]
        return lower, upper

    def _elementwise(self, y_pred, y_true):
        lower, upper = self._bounds(y_pred)
        y = y_true[..., 0]
        width = upper - lower
        score = width.clone()
        score = torch.where(y < lower, width + (2.0 / self.alpha) * (lower - y), score)
        score = torch.where(y > upper, width + (2.0 / self.alpha) * (y - upper), score)
        return score

    def forward(self, y_pred, y_true):
        return self._reduce(self._elementwise(y_pred, y_true))

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update(self._elementwise(y_pred, y_true))

    def pooled_value(self, accs):
        return accs['val'].mean


class PICP(_Metric):
    """
    Prediction Interval Coverage Probability: the fraction of observations that fall
    within the (1-alpha) prediction interval, read from the channels at levels alpha/2
    and 1-alpha/2 (same convention as WinklerScore). Ideally close to (1-alpha).
    Evaluation only (differentiable=False) — requires a quantile- or expectile-output
    head with those exact levels present, else KeyError (see _Metric.bind_levels).
    Exact for a quantile head; an accepted approximation for an expectile head.
    """
    differentiable = False
    poolable = True

    def __init__(self, alpha):
        super().__init__(dim=None)
        self.alpha = alpha
        self._lo_level = round(alpha / 2, 6)
        self._hi_level = round(1 - alpha / 2, 6)

    def _elementwise(self, y_pred, y_true):
        lower = y_pred[..., self._level_index[self._lo_level]]
        upper = y_pred[..., self._level_index[self._hi_level]]
        y = y_true[..., 0]
        return ((y >= lower) & (y <= upper)).to(y_pred.dtype)

    def forward(self, y_pred, y_true):
        return self._reduce(self._elementwise(y_pred, y_true))

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update(self._elementwise(y_pred, y_true))

    def pooled_value(self, accs):
        return accs['val'].mean


class CoverageGap(PICP):
    """
    PICP minus its own nominal coverage (1-alpha): 0 = perfectly calibrated, positive
    = over-covered, negative = under-covered. Same lower/upper channel convention and
    requirements as PICP (alpha/2, 1-alpha/2 — a plain KeyError if those levels aren't
    present); only the reference point changes, so this stays a thin subclass rather
    than a reimplementation.

    Subtracting the constant (1-alpha) inside _elementwise (rather than in forward()
    alone) is what's needed to make this poolable correctly, since it makes the shift
    itself part of the elementwise value pooled_value's ChanAccumulator averages —
    mean(x - c) == mean(x) - c exactly (mean is linear), so this doesn't bias the
    pooled epoch-level value the way shifting a per-batch average would.
    """
    def _elementwise(self, y_pred, y_true):
        return super()._elementwise(y_pred, y_true) - (1 - self.alpha)


class PINAW(_Metric):
    """
    Prediction Interval Normalized Average Width: mean width of the (1-alpha) interval
    (same lower/upper convention as WinklerScore/PICP), normalized by the range
    (max - min) of y_true over the whole evaluation set. Lower is better (tighter
    intervals), but only meaningful alongside coverage (PICP) — a narrow-but-wrong
    interval scores well here alone. Evaluation only (differentiable=False); requires a
    quantile- or expectile-output head with those exact levels present, else KeyError.
    """
    differentiable = False
    poolable = True

    def __init__(self, alpha):
        super().__init__(dim=None)
        self.alpha = alpha
        self._lo_level = round(alpha / 2, 6)
        self._hi_level = round(1 - alpha / 2, 6)

    def _width(self, y_pred):
        lower = y_pred[..., self._level_index[self._lo_level]]
        upper = y_pred[..., self._level_index[self._hi_level]]
        return upper - lower

    def forward(self, y_pred, y_true):
        width = self._width(y_pred)
        r = (y_true.max() - y_true.min()).clamp(min=1e-8)
        return width.mean() / r

    def new_accumulators(self):
        return {'width': ChanAccumulator(), 'ymin': float('inf'), 'ymax': float('-inf')}

    def accumulate(self, accs, y_pred, y_true):
        accs['width'].update(self._width(y_pred))
        accs['ymin'] = min(accs['ymin'], y_true.min().item())
        accs['ymax'] = max(accs['ymax'], y_true.max().item())

    def pooled_value(self, accs):
        r = accs['ymax'] - accs['ymin']
        r = r if r > 1e-8 else 1e-8
        return accs['width'].mean / r


class CRPS(_Metric):
    """
    Continuous Ranked Probability Score, approximated from a quantile output head via
    the standard pinball-loss decomposition (Gneiting & Raftery, 2007):
        CRPS ~= 2 * mean_q(pinball_q)
    averaged over every channel the head provides (via self._level_index — all of
    them, not just a chosen pair). No parameters. Evaluation only
    (differentiable=False); requires a quantile- or expectile-output head, else
    KeyError (empty self._level_index raises on the first channel lookup). The
    pinball-based formula is exact only for true quantiles — reused against an
    expectile head's levels as an accepted approximation, same as WinklerScore/PICP/
    PINAW.
    """
    differentiable = False
    poolable = True

    def _elementwise(self, y_pred, y_true):
        _require_bound_levels(self)
        items = sorted(self._level_index.items())
        idx = [i for _, i in items]
        levels = torch.tensor([l for l, _ in items], device=y_pred.device, dtype=y_pred.dtype)
        yq = y_pred[..., idx]  # (..., Q)
        diff = y_true - yq  # (..., 1) - (..., Q) broadcasts to (..., Q)
        pinball = torch.maximum(levels * diff, (levels - 1) * diff)
        return 2.0 * pinball.mean(dim=-1)

    def forward(self, y_pred, y_true):
        return self._reduce(self._elementwise(y_pred, y_true))

    def new_accumulators(self):
        return {'val': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        accs['val'].update(self._elementwise(y_pred, y_true))

    def pooled_value(self, accs):
        return accs['val'].mean


class SCRPS(_Metric):
    """
    Scale-invariant CRPS (Bolin & Wallin, 2020, "Local Scale Invariance and Robustness
    for Proper Scoring Rules"): built from the same pinball-loss quantile
    decomposition CRPS uses above, but normalized by the predictive distribution's own
    spread so forecasts at different magnitudes stay comparable — unlike plain CRPS,
    which scales with the units of y. A proper *scoring rule* (higher is better), not
    a loss, following Bolin & Wallin's own sign convention:

        SCRPS(F, y) = -E|X - y| / E|X - X'| - 0.5 * log(E|X - X'|)

    for X, X' ~ F independent. Both expectations are estimated from the same
    quantile-decomposition CRPS already relies on:
        E|X - X'| (Gini mean difference of F) = 4 * mean_q((level_q - 0.5) * Q(level_q))
        E|X - y|                              = CRPS(F, y) + 0.5 * E|X - X'|
    (the first is the standard quantile-function identity for the Gini mean
    difference, discretized the same way CRPS discretizes its own integral —
    averaging over the head's evenly-spaced levels approximates the [0, 1] integral;
    the second follows directly from CRPS's own definition, CRPS = E|X-y| -
    0.5*E|X-X'|). Both terms are pooled over the *whole* evaluation set before
    combining — SCRPS is nonlinear in these expectations, so (unlike CRPS) averaging
    already-combined per-sample values would be a biased estimate, the same reasoning
    NSE's poolable path uses. Requires a quantile- or expectile-output head, else
    KeyError, same as CRPS.
    """
    differentiable = False
    poolable = True

    def _terms(self, y_pred, y_true):
        _require_bound_levels(self)
        items = sorted(self._level_index.items())
        idx = [i for _, i in items]
        levels = torch.tensor([l for l, _ in items], device=y_pred.device, dtype=y_pred.dtype)
        yq = y_pred[..., idx]  # (..., Q)
        diff = y_true - yq
        pinball = torch.maximum(levels * diff, (levels - 1) * diff)
        crps = 2.0 * pinball.mean(dim=-1)  # (...,) elementwise E|X-y| - 0.5*E|X-X'|
        spread = 4.0 * ((levels - 0.5) * yq).mean(dim=-1)  # (...,) elementwise Gini mean difference
        return crps, spread

    def forward(self, y_pred, y_true):
        crps, spread = self._terms(y_pred, y_true)
        crps_m = crps.mean()
        spread_m = spread.mean().clamp(min=1e-8)
        e1 = crps_m + 0.5 * spread_m
        return -(e1 / spread_m) - 0.5 * torch.log(spread_m)

    def new_accumulators(self):
        return {'crps': ChanAccumulator(), 'spread': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        crps, spread = self._terms(y_pred, y_true)
        accs['crps'].update(crps)
        accs['spread'].update(spread)

    def pooled_value(self, accs):
        crps = accs['crps'].mean
        spread = max(accs['spread'].mean, 1e-8)
        e1 = crps + 0.5 * spread
        return -(e1 / spread) - 0.5 * math.log(spread)


class RSCRPS(_Metric):
    """
    Robust scale-invariant CRPS (rSCRPS) -- Bolin & Wallin, 2023, "Local scale
    invariance and robustness of proper scoring rules" (Statistical Science 38(1)),
    Corollary 1 / the "standardized" kernel score (their Theorem 3, h(x) =
    -0.5*log(x)) applied to their robust, outlier-clipped kernel (their eq. 10):

        g_c(x, x') = |x - x'|   if |x - x'| < c
                     c          otherwise

    in place of the plain |x - x'| kernel SCRPS uses. That gives:

        rSCRPS(F, y) = -E[g_c(X, y)] / E[g_c(X, X')] - 0.5 * log(E[g_c(X, X')])

    for X, X' ~ F independent -- same combination SCRPS uses, with E|X-y| and
    E|X-X'| replaced by their g_c-kernel counterparts. Capping the kernel at c
    bounds the influence any single (potentially extreme) observation or draw
    has on the score, at the cost of exact local scale invariance -- c is a
    fixed constant in y's units, not something that rescales with the
    forecast's own spread (see the paper's discussion of rCRPS for why). A
    proper scoring rule (higher is better), same sign convention as SCRPS.

    Unlike SCRPS, the clipped kernel has no closed-form Gini-mean-difference
    shortcut, so both expectations are estimated directly from the head's
    discrete quantile grid: E[g_c(X, y)] as mean_q(g_c(Q(q), y)), E[g_c(X, X')]
    as the mean of g_c(Q(q_i), Q(q_j)) over every (q_i, q_j) pair -- cheap at
    the ~39-level grids used throughout this codebase. Both terms are pooled
    over the whole evaluation set before combining, same nonlinear-combination
    reasoning as SCRPS. Requires a quantile- or expectile-output head, else
    KeyError, same as CRPS/SCRPS.
    """
    differentiable = False
    poolable = True

    def __init__(self, c):
        super().__init__(dim=None)
        self.c = c

    def _terms(self, y_pred, y_true):
        _require_bound_levels(self)
        items = sorted(self._level_index.items())
        idx = [i for _, i in items]
        yq = y_pred[..., idx]  # (..., Q)
        y = y_true[..., 0]  # (...,)
        num = (y.unsqueeze(-1) - yq).abs().clamp(max=self.c).mean(dim=-1)  # (...,) E[g_c(X,y)]
        pair_diff = (yq.unsqueeze(-1) - yq.unsqueeze(-2)).abs().clamp(max=self.c)  # (..., Q, Q)
        denom = pair_diff.mean(dim=(-1, -2))  # (...,) E[g_c(X,X')]
        return num, denom

    def forward(self, y_pred, y_true):
        num, denom = self._terms(y_pred, y_true)
        num_m = num.mean()
        denom_m = denom.mean().clamp(min=1e-8)
        return -(num_m / denom_m) - 0.5 * torch.log(denom_m)

    def new_accumulators(self):
        return {'num': ChanAccumulator(), 'denom': ChanAccumulator()}

    def accumulate(self, accs, y_pred, y_true):
        num, denom = self._terms(y_pred, y_true)
        accs['num'].update(num)
        accs['denom'].update(denom)

    def pooled_value(self, accs):
        num = accs['num'].mean
        denom = max(accs['denom'].mean, 1e-8)
        return -(num / denom) - 0.5 * math.log(denom)
