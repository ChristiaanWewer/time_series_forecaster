import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledHead(nn.Module):
    """Output head for backbones that produce a single pooled hidden vector per sample,
    (B, hidden) — e.g. an encoder-only model whose last hidden state is mapped straight
    to every horizon step at once, with no per-step decoder representation to project
    from. Projects to horizon * n_outputs values in one Linear, reshaped to
    (B, horizon, n_outputs).
    """
    def __init__(self, in_features, horizon, n_outputs):
        super().__init__()
        self.horizon = horizon
        self.n_outputs = n_outputs
        self.linear = nn.Linear(in_features=in_features, out_features=horizon * n_outputs)

    def forward(self, h):
        return self.linear(h).view(h.shape[0], self.horizon, self.n_outputs)


class SequenceHead(nn.Module):
    """Output head for backbones that already produce a per-timestep hidden
    representation, (B, horizon, hidden) — e.g. a decoder's outputs. A plain Linear
    applied at each timestep (nn.Linear broadcasts over the middle dim for free),
    projecting hidden -> n_outputs.
    """
    def __init__(self, in_features, n_outputs):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=n_outputs)

    def forward(self, h):
        return self.linear(h)


class MonotonicHead(nn.Module):
    """Wraps a raw multi-channel head (PooledHead/SequenceHead) and reparametrizes its
    n_outputs raw channels into monotonically ordered levels around a literal center
    channel — guarantees non-crossing quantiles/expectiles by construction, rather than
    relying on the loss alone to discourage crossing (a standard fix for quantile
    crossing; cf. Cannon 2018's monotone composite quantile regression). This is the
    standard (always-on) head for a quantile/expectile output — see
    build_output_head() below, which is what actually decides when to use it.

    The channel at center_idx is used directly and unconstrained as the center value
    (the median, for a quantile head with 0.5 among its levels; the mean, for an
    expectile head — see ExpectileLoss). Every channel above it becomes a strictly
    positive offset via softplus, cumulatively summed outward from the center, so each
    successive higher level is guaranteed >= the one before it; channels below center
    are built the same way and subtracted, guaranteeing each successive lower level is
    <= the one before it. center_idx is the level-0.5 channel's position within
    output_levels (sorted ascending) — see QuantileLoss/ExpectileLoss's output_levels.
    """
    def __init__(self, raw_head, center_idx):
        super().__init__()
        self.raw_head = raw_head
        self.center_idx = center_idx

    def forward(self, h):
        raw = self.raw_head(h)
        center = raw[..., self.center_idx:self.center_idx + 1]
        below = raw[..., :self.center_idx]
        above = raw[..., self.center_idx + 1:]
        parts = []
        if below.shape[-1] > 0:
            # closest-to-center channel accumulates first: flip so cumsum runs
            # center-outward, then flip back to restore ascending level order
            offsets = F.softplus(below.flip(-1)).cumsum(-1).flip(-1)
            parts.append(center - offsets)
        parts.append(center)
        if above.shape[-1] > 0:
            offsets = F.softplus(above).cumsum(-1)
            parts.append(center + offsets)
        return torch.cat(parts, dim=-1)


def build_output_head(representation, in_features, horizon, n_outputs, output_kind, output_levels):
    """Builds the output head a model should use, given its backbone's representation
    shape and the loss function's output_kind/output_levels/n_outputs (see Forecaster,
    which injects these into model_config). representation is 'pooled' (backbone
    produces (B, hidden) -> PooledHead) or 'sequence' (backbone produces
    (B, horizon, hidden) -> SequenceHead, horizon unused).

    Standard (always-on, not opt-in) behavior for a probabilistic head
    (output_kind in ('quantile', 'expectile')): wraps the raw head in a MonotonicHead
    anchored at the level-0.5 channel. A plain point head (output_kind='point',
    n_outputs=1) is returned unwrapped. output_levels must contain 0.5 for a
    probabilistic head — a plain ValueError (list.index) if it doesn't, deliberately
    uncaught, same "let it crash" convention scores_and_losses.py uses elsewhere.
    """
    if representation == 'pooled':
        raw_head = PooledHead(in_features, horizon, n_outputs)
    elif representation == 'sequence':
        raw_head = SequenceHead(in_features, n_outputs)
    else:
        raise ValueError(f"Unknown representation '{representation}'")

    if output_kind in ('quantile', 'expectile'):
        center_idx = sorted(output_levels).index(0.5)
        return MonotonicHead(raw_head, center_idx)
    return raw_head
