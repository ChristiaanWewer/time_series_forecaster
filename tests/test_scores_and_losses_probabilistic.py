import math

import pytest
import torch

from src.utils.scores_and_losses import (
    QuantileLoss, ExpectileLoss, WinklerScore, PICP, CoverageGap, PINAW, CRPS, SCRPS, RSCRPS,
)


# ── QuantileLoss / ExpectileLoss ────────────────────────────────────────────────

def test_quantile_loss_pinball_formula():
    # y_true=3, y_pred=2 -> diff=+1 (under-prediction); y_true=2, y_pred=3 -> diff=-1
    y_pred = torch.tensor([[2.0, 2.0], [3.0, 3.0]])  # (B=2, Q=2) for q=(0.1, 0.9)
    y_true = torch.tensor([[3.0], [2.0]])
    metric = QuantileLoss(q=(0.1, 0.9))
    err = metric._error(y_pred, y_true)
    # row 0 (diff=+1): q=0.1 -> max(0.1, -0.9) = 0.1;  q=0.9 -> max(0.9, -0.1) = 0.9
    # row 1 (diff=-1): q=0.1 -> max(-0.1, 0.9) = 0.9;  q=0.9 -> max(-0.9, 0.1) = 0.1
    assert torch.allclose(err, torch.tensor([[0.1, 0.9], [0.9, 0.1]]))


def test_expectile_loss_formula():
    # u = y_true - y_pred = +1 (u>=0) or -1 (u<0)
    y_pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y_true = torch.tensor([[1.0], [0.0]])  # u = +1, then -1
    metric = ExpectileLoss(e=(0.1, 0.9))
    err = metric._error(y_pred, y_true)
    # u=+1 (not <0): |e-0|*1 -> [0.1, 0.9]
    # u=-1 (<0):     |e-1|*1 -> [0.9, 0.1]
    assert torch.allclose(err, torch.tensor([[0.1, 0.9], [0.9, 0.1]]))


def test_quantile_loss_sets_output_shape_from_q_sorted():
    metric = QuantileLoss(q=[0.9, 0.1, 0.5])
    assert metric.output_kind == 'quantile'
    assert metric.output_levels == [0.1, 0.5, 0.9]
    assert metric.n_outputs == 3


def test_expectile_loss_sets_output_shape_from_e_sorted():
    metric = ExpectileLoss(e=[0.9, 0.5, 0.1])
    assert metric.output_kind == 'expectile'
    assert metric.output_levels == [0.1, 0.5, 0.9]
    assert metric.n_outputs == 3


@pytest.mark.parametrize("metric_factory", [
    lambda: QuantileLoss(q=(0.1, 0.5, 0.9)),
    lambda: ExpectileLoss(e=(0.1, 0.5, 0.9)),
])
def test_poolable_path_matches_forward_on_whole_dataset(metric_factory):
    torch.manual_seed(0)
    batches = [(torch.randn(4, 6, 3), torch.randn(4, 6, 1)) for _ in range(3)]
    metric = metric_factory()
    accs = metric.new_accumulators()
    for yp, yt in batches:
        metric.accumulate(accs, yp, yt)
    pooled = metric.pooled_value(accs)

    yp_all = torch.cat([b[0] for b in batches], dim=0)
    yt_all = torch.cat([b[1] for b in batches], dim=0)
    direct = metric(yp_all, yt_all).item()
    assert math.isclose(pooled, direct, rel_tol=1e-5, abs_tol=1e-6)


@pytest.mark.parametrize("metric_cls,kwarg,power", [(QuantileLoss, 'q', 1), (ExpectileLoss, 'e', 2)])
def test_reweight_applies_documented_per_series_weight_formula(metric_cls, kwarg, power):
    eps = 0.1
    group_std = torch.tensor([1.0, 10.0])
    metric = metric_cls(**{kwarg: (0.5,)}, reweight=True, eps=eps)
    metric.bind(group_std)

    y_pred = torch.zeros(4, 1)
    y_true = torch.ones(4, 1)  # constant elementwise error for every sample/level
    group_idx = torch.tensor([0, 0, 1, 1])
    got = metric(y_pred, y_true, group_idx).item()

    base_err = metric._error(y_pred, y_true)[0, 0].item()  # same for every row here
    w0 = 1.0 / (group_std[0].item() + eps) ** power
    w1 = 1.0 / (group_std[1].item() + eps) ** power
    expected = (base_err * w0 + base_err * w0 + base_err * w1 + base_err * w1) / 4.0
    assert math.isclose(got, expected, rel_tol=1e-5)


# ── WinklerScore / PICP / CoverageGap / PINAW ──────────────────────────────────

_LEVELS = [0.025, 0.5, 0.975]  # alpha=0.05 pair at indices 0 (lower) and 2 (upper)


def _bound(metric_cls, alpha=0.05, **kwargs):
    m = metric_cls(alpha=alpha, **kwargs)
    m.bind_levels('quantile', _LEVELS)
    return m


def test_winkler_score_all_three_branches():
    # channels: [lower=0, center=5, upper=10]
    y_pred = torch.tensor([[0.0, 5.0, 10.0]] * 3)
    y_true = torch.tensor([[4.0], [-3.0], [15.0]])  # inside, below lower, above upper
    metric = _bound(WinklerScore)
    got = metric._elementwise(y_pred, y_true)
    width = 10.0
    expected = torch.tensor([
        width,                              # inside: just the width
        width + (2.0 / 0.05) * (0.0 - (-3.0)),   # below: width + (2/alpha)*(lower-y)
        width + (2.0 / 0.05) * (15.0 - 10.0),    # above: width + (2/alpha)*(y-upper)
    ])
    assert torch.allclose(got, expected)


def test_picp_fraction_covered():
    y_pred = torch.tensor([[0.0, 5.0, 10.0]] * 4)
    y_true = torch.tensor([[4.0], [-3.0], [15.0], [10.0]])  # 2 of 4 covered (inside, on boundary)
    metric = _bound(PICP)
    got = metric(y_pred, y_true).item()
    assert math.isclose(got, 2.0 / 4.0)


def test_coverage_gap_is_picp_minus_nominal_coverage():
    torch.manual_seed(0)
    y_pred = torch.tensor([0.0, 5.0, 10.0]).unsqueeze(0).repeat(10, 1)
    y_true = torch.randn(10, 1) * 8 + 5
    picp = _bound(PICP)
    gap = _bound(CoverageGap)
    assert math.isclose(gap(y_pred, y_true).item(), picp(y_pred, y_true).item() - (1 - 0.05), rel_tol=1e-6)


def test_coverage_gap_pooling_is_exact_via_linearity():
    # mean(x - c) == mean(x) - c exactly, so pooling the shifted elementwise value
    # must match pooling PICP's own elementwise value then shifting once at the end
    torch.manual_seed(1)
    batches = [(torch.tensor([0.0, 5.0, 10.0]).unsqueeze(0).repeat(6, 1), torch.randn(6, 1) * 8 + 5) for _ in range(3)]
    gap = _bound(CoverageGap)
    accs = gap.new_accumulators()
    for yp, yt in batches:
        gap.accumulate(accs, yp, yt)
    pooled = gap.pooled_value(accs)

    yp_all = torch.cat([b[0] for b in batches])
    yt_all = torch.cat([b[1] for b in batches])
    direct = gap(yp_all, yt_all).item()
    assert math.isclose(pooled, direct, rel_tol=1e-5)


def test_pinaw_width_normalized_by_range():
    y_pred = torch.tensor([[0.0, 5.0, 10.0]] * 3)  # width = 10 for every row
    y_true = torch.tensor([[0.0], [20.0], [-5.0]])  # range = 25
    metric = _bound(PINAW)
    got = metric(y_pred, y_true).item()
    assert math.isclose(got, 10.0 / 25.0, rel_tol=1e-5)


@pytest.mark.parametrize("metric_cls", [WinklerScore, PICP, PINAW])
def test_missing_level_raises_keyerror(metric_cls):
    m = metric_cls(alpha=0.05)  # bind_levels never called -> _level_index is {}
    y_pred = torch.tensor([[0.0, 5.0, 10.0]])
    y_true = torch.tensor([[4.0]])
    with pytest.raises(KeyError):
        m(y_pred, y_true)


# ── CRPS / SCRPS / RSCRPS — paper-grounded properties ──────────────────────────

def _normal_quantile_head(mu, sigma, n_levels=999):
    levels = torch.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    dist = torch.distributions.Normal(torch.tensor(float(mu)), torch.tensor(float(sigma)))
    quantiles = dist.icdf(levels)
    return levels.tolist(), quantiles


def _closed_form_normal_crps(mu, sigma, y):
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    Phi = normal.cdf(torch.tensor(float(z)))
    phi = torch.exp(torch.tensor(-0.5 * z * z)) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * Phi.item() - 1) + 2 * phi.item() - 1.0 / math.sqrt(math.pi))


def test_crps_matches_gneiting_raftery_closed_form_for_normal():
    # Gneiting & Raftery (2007): CRPS(N(mu, sigma^2), y) = sigma*(z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    mu, sigma, y = 2.0, 1.5, 3.0
    levels, quantiles = _normal_quantile_head(mu, sigma, n_levels=999)
    metric = CRPS()
    metric.bind_levels('quantile', levels)
    y_pred = quantiles.unsqueeze(0)  # (1, Q)
    y_true = torch.tensor([[y]])
    got = metric(y_pred, y_true).item()
    expected = _closed_form_normal_crps(mu, sigma, y)
    assert math.isclose(got, expected, rel_tol=2e-3)


def test_crps_decomposition_converges_as_grid_gets_finer():
    mu, sigma, y = 2.0, 1.5, 3.0
    expected = _closed_form_normal_crps(mu, sigma, y)

    errors = []
    for n_levels in (19, 199, 1999):
        levels, quantiles = _normal_quantile_head(mu, sigma, n_levels=n_levels)
        metric = CRPS()
        metric.bind_levels('quantile', levels)
        got = metric(quantiles.unsqueeze(0), torch.tensor([[y]])).item()
        errors.append(abs(got - expected))

    assert errors[0] > errors[1] > errors[2]


def test_scrps_raw_value_shifts_by_log_scale_factor_under_rescaling():
    # NOT the paper's "local scale invariance" (see the next test for that) — this is
    # a separate, exact algebraic fact about the raw SCRPS *value*, derived directly
    # from the formula: SCRPS(a*F+b, a*y+b) = SCRPS(F, y) - 0.5*log(a). Under X -> aX+b,
    # E|X-y| and E|X-X'| both scale by exactly a (a>0), so the ratio term is unchanged
    # but the -0.5*log(E|X-X'|) term picks up an extra -0.5*log(a). True for any (F, y),
    # not an approximation. Worth knowing separately: literally averaging/summing raw
    # SCRPS values across series of very different absolute scale still carries this
    # residual additive bias, even though the metric's actual invariance property
    # (below) is about something else entirely.
    torch.manual_seed(0)
    levels, quantiles = _normal_quantile_head(mu=1.0, sigma=2.0, n_levels=201)
    y = torch.tensor([[2.5]])
    y_pred = quantiles.unsqueeze(0)

    metric = SCRPS()
    metric.bind_levels('quantile', levels)
    base = metric(y_pred, y).item()

    for a, b in [(3.0, 0.0), (0.1, 0.0), (5.0, -7.0)]:
        scaled = metric(a * y_pred + b, a * y + b).item()
        expected = base - 0.5 * math.log(a)
        assert math.isclose(scaled, expected, rel_tol=1e-4), f"a={a}, b={b}"


def test_scrps_discrimination_penalty_is_scale_invariant():
    # THE paper's actual "local scale invariance" (Bolin & Wallin, Definition 2 /
    # Proposition 4): not that the raw score value is unchanged under rescaling (see
    # the previous test — it isn't), but that the score's *discriminatory power* is.
    # Their Definition 2 formalizes this via the local behavior
    #   S(Q, Q) - S(Q_perturbed, Q) ~ t^2 * sigma^2 * s(Q)  as t -> 0,
    # a "scale function" s(Q) that is itself required to scale as 1/sigma^2 — so the
    # penalty for a perturbation of a FIXED PROPORTION of the distribution's own
    # spread (t*sigma) is asymptotically the SAME regardless of sigma.
    #
    # For SCRPS specifically this holds exactly, not just asymptotically, and follows
    # directly from the previous test: since SCRPS(a*F+b, a*y+b) = SCRPS(F,y) -
    # 0.5*log(a) for *any* (F, y) pair, that additive -0.5*log(a) term is identical for
    # both the correctly-specified and the perturbed prediction (same a), so it cancels
    # exactly out of their difference — the discrimination penalty is scale-invariant
    # for any perturbation size, not just infinitesimally small ones.
    torch.manual_seed(1)
    mu, sigma = 0.0, 1.0
    t = 0.3  # perturbation size, as a fraction of sigma — not required to be tiny here
    levels, correct_quantiles = _normal_quantile_head(mu, sigma, n_levels=201)
    _, perturbed_quantiles = _normal_quantile_head(mu + t * sigma, sigma, n_levels=201)
    y = torch.tensor([[0.4]])

    metric = SCRPS()
    metric.bind_levels('quantile', levels)
    penalty_base = metric(correct_quantiles.unsqueeze(0), y).item() - \
        metric(perturbed_quantiles.unsqueeze(0), y).item()

    for a in (0.01, 5.0, 100.0):  # wildly different scales
        b = 3.0
        penalty_scaled = (
            metric(a * correct_quantiles.unsqueeze(0) + b, a * y + b).item()
            - metric(a * perturbed_quantiles.unsqueeze(0) + b, a * y + b).item()
        )
        assert math.isclose(penalty_scaled, penalty_base, rel_tol=1e-4), f"a={a}"

    # sanity: a bigger misspecification really does produce a bigger penalty
    _, further_perturbed = _normal_quantile_head(mu + 1.0 * sigma, sigma, n_levels=201)
    bigger_penalty = metric(correct_quantiles.unsqueeze(0), y).item() - \
        metric(further_perturbed.unsqueeze(0), y).item()
    assert bigger_penalty > penalty_base


def test_crps_discrimination_penalty_is_not_scale_invariant():
    # contrast: plain CRPS's discrimination penalty DOES depend on scale (it has no
    # mechanism analogous to SCRPS's cancelling log term) — this is exactly the
    # practical problem SCRPS's actual invariance property fixes
    torch.manual_seed(1)
    mu, sigma, t = 0.0, 1.0, 0.3
    levels, correct_quantiles = _normal_quantile_head(mu, sigma, n_levels=201)
    _, perturbed_quantiles = _normal_quantile_head(mu + t * sigma, sigma, n_levels=201)
    y = torch.tensor([[0.4]])

    metric = CRPS()
    metric.bind_levels('quantile', levels)
    penalty_base = metric(correct_quantiles.unsqueeze(0), y).item() - \
        metric(perturbed_quantiles.unsqueeze(0), y).item()

    a = 5.0
    penalty_scaled = (
        metric(a * correct_quantiles.unsqueeze(0), a * y).item()
        - metric(a * perturbed_quantiles.unsqueeze(0), a * y).item()
    )
    assert not math.isclose(penalty_scaled, penalty_base, rel_tol=1e-2)
    assert math.isclose(penalty_scaled, a * penalty_base, rel_tol=1e-3)


def test_crps_is_not_scale_invariant_unlike_scrps():
    # contrast: plain CRPS is positively homogeneous of degree 1 under this transform
    # (CRPS(a*F+b, a*y+b) == a*CRPS(F,y)), not invariant, for a != 1
    levels, quantiles = _normal_quantile_head(mu=1.0, sigma=2.0, n_levels=201)
    y = torch.tensor([[2.5]])
    y_pred = quantiles.unsqueeze(0)

    metric = CRPS()
    metric.bind_levels('quantile', levels)
    base = metric(y_pred, y).item()

    a = 3.0
    scaled = metric(a * y_pred, a * y).item()
    assert not math.isclose(scaled, base, rel_tol=1e-2)
    assert math.isclose(scaled, a * base, rel_tol=1e-3)


def test_rscrps_is_not_scale_invariant_unlike_scrps():
    # Bolin & Wallin's own Table 3: rSCRPS is robust but NOT scale invariant, because
    # c is a fixed constant in y's absolute units
    levels, quantiles = _normal_quantile_head(mu=1.0, sigma=2.0, n_levels=201)
    y = torch.tensor([[2.5]])
    y_pred = quantiles.unsqueeze(0)

    metric = RSCRPS(c=1.0)
    metric.bind_levels('quantile', levels)
    base = metric(y_pred, y).item()

    a = 5.0
    scaled = metric(a * y_pred, a * y).item()
    assert not math.isclose(scaled, base, rel_tol=1e-2)


def test_rscrps_converges_to_scrps_as_c_grows():
    # as c -> infinity the clip stops binding and g_c(x,x') -> |x-x'|, so rSCRPS -> SCRPS
    levels, quantiles = _normal_quantile_head(mu=1.0, sigma=2.0, n_levels=201)
    y = torch.tensor([[2.5]])
    y_pred = quantiles.unsqueeze(0)

    scrps = SCRPS()
    scrps.bind_levels('quantile', levels)
    scrps_val = scrps(y_pred, y).item()

    rscrps = RSCRPS(c=1000.0)
    rscrps.bind_levels('quantile', levels)
    rscrps_val = rscrps(y_pred, y).item()

    # not exact even at a large c: SCRPS's closed-form Gini-mean-difference shortcut
    # and RSCRPS's direct pairwise-grid estimate are two different discretizations of
    # the same underlying integral, so a small residual gap remains regardless of c
    assert math.isclose(scrps_val, rscrps_val, rel_tol=3e-3)


def test_rscrps_has_bounded_influence_from_a_single_outlier():
    # robustness claim: one extreme outlier observation should move RSCRPS less than
    # it moves SCRPS/CRPS, since the clipped kernel bounds any single pair's influence
    torch.manual_seed(2)
    levels, quantiles = _normal_quantile_head(mu=0.0, sigma=1.0, n_levels=201)
    y_pred = quantiles.unsqueeze(0).repeat(20, 1)

    y_baseline = torch.randn(20, 1) * 1.0
    y_outlier = y_baseline.clone()
    y_outlier[0, 0] = 500.0  # one extreme outlier among 20 observations

    def pooled(metric_cls, y_true, **kwargs):
        m = metric_cls(**kwargs)
        m.bind_levels('quantile', levels)
        accs = m.new_accumulators()
        m.accumulate(accs, y_pred, y_true)
        return m.pooled_value(accs)

    d_crps = abs(pooled(CRPS, y_outlier) - pooled(CRPS, y_baseline))
    d_scrps = abs(pooled(SCRPS, y_outlier) - pooled(SCRPS, y_baseline))
    d_rscrps = abs(pooled(RSCRPS, y_outlier, c=2.0) - pooled(RSCRPS, y_baseline, c=2.0))

    assert d_rscrps < d_scrps
    assert d_rscrps < d_crps


@pytest.mark.parametrize("metric_factory", [CRPS, SCRPS, lambda: RSCRPS(c=1.0)])
def test_crps_scrps_rscrps_require_bound_levels(metric_factory):
    m = metric_factory()  # bind_levels never called
    y_pred = torch.tensor([[0.0, 5.0, 10.0]])
    y_true = torch.tensor([[4.0]])
    with pytest.raises(KeyError):
        m(y_pred, y_true)


# ── bind_levels / _select_point cross-cutting behavior ─────────────────────────

def test_bind_levels_empty_for_point_kind():
    metric = WinklerScore(alpha=0.05)
    metric.bind_levels('point', None)
    assert metric._level_index == {}


def test_bind_levels_populates_for_quantile_and_expectile():
    metric = WinklerScore(alpha=0.05)
    metric.bind_levels('quantile', [0.1, 0.5, 0.9])
    assert metric._level_index == {0.1: 0, 0.5: 1, 0.9: 2}

    metric2 = WinklerScore(alpha=0.05)
    metric2.bind_levels('expectile', [0.1, 0.5, 0.9])
    assert metric2._level_index == {0.1: 0, 0.5: 1, 0.9: 2}
