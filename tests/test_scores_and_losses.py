import math

import pytest
import torch

from src.utils.scores_and_losses import (
    ChanAccumulator, MAE, MSE, RMSE, MAPE, SMAPE, NSE, AlphaNSE, BetaNSE,
)


def test_empty_update_is_a_noop():
    acc = ChanAccumulator()
    acc.update(torch.empty(0))
    assert acc.n == 0.0
    assert acc.mean == 0.0
    assert acc.M2 == 0.0


def test_variance_is_nan_before_enough_samples():
    acc = ChanAccumulator()
    assert math.isnan(acc.variance)
    acc.update(torch.tensor([5.0]))
    assert math.isnan(acc.variance)  # n == 1, still not enough for a variance


def test_single_batch_matches_direct_mean_and_variance():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    acc = ChanAccumulator()
    acc.update(values)
    assert acc.n == 5.0
    assert math.isclose(acc.mean, values.mean().item(), rel_tol=1e-6)
    # ChanAccumulator.M2 is the sum of squared deviations (population, not sample) —
    # variance = M2 / n matches torch.var(unbiased=False)
    assert math.isclose(acc.variance, values.var(unbiased=False).item(), rel_tol=1e-6)


def test_multiple_batches_match_pooling_the_whole_dataset_at_once():
    torch.manual_seed(0)
    batches = [torch.randn(7), torch.randn(3), torch.randn(50), torch.randn(1)]
    full = torch.cat(batches)

    acc = ChanAccumulator()
    for b in batches:
        acc.update(b)

    assert acc.n == full.numel()
    assert math.isclose(acc.mean, full.mean().item(), rel_tol=1e-5)
    assert math.isclose(acc.variance, full.var(unbiased=False).item(), rel_tol=1e-5)


def test_update_reshapes_multidimensional_input():
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # (B, H, C)-shaped
    acc = ChanAccumulator()
    acc.update(values)
    flat = values.reshape(-1)
    assert acc.n == flat.numel()
    assert math.isclose(acc.mean, flat.mean().item(), rel_tol=1e-6)


def test_accumulation_is_order_independent():
    torch.manual_seed(1)
    batches = [torch.randn(10) * 3 + 5 for _ in range(5)]

    acc_forward = ChanAccumulator()
    for b in batches:
        acc_forward.update(b)

    acc_reversed = ChanAccumulator()
    for b in reversed(batches):
        acc_reversed.update(b)

    assert math.isclose(acc_forward.mean, acc_reversed.mean, rel_tol=1e-6)
    assert math.isclose(acc_forward.variance, acc_reversed.variance, rel_tol=1e-6)


# ── point regression metrics: MAE, MSE, RMSE, MAPE, SMAPE ──────────────────────

_Y_PRED = torch.tensor([[1.0], [2.0], [3.0]])
_Y_TRUE = torch.tensor([[2.0], [2.0], [5.0]])


def test_mae_formula():
    assert torch.isclose(MAE()(_Y_PRED, _Y_TRUE), torch.tensor(1.0))


def test_mse_formula():
    # |diff|^2 = [1, 0, 4] -> mean = 5/3
    assert torch.isclose(MSE()(_Y_PRED, _Y_TRUE), torch.tensor(5.0 / 3.0))


def test_rmse_formula():
    assert torch.isclose(RMSE()(_Y_PRED, _Y_TRUE), torch.tensor((5.0 / 3.0) ** 0.5))


def test_mape_formula():
    # 100 * |diff|/|true| = [50, 0, 40] -> mean = 30
    assert torch.isclose(MAPE()(_Y_PRED, _Y_TRUE), torch.tensor(30.0))


def test_smape_formula():
    # 200 * |diff|/(|pred|+|true|) = [200/3, 0, 50] -> mean = (66.666.. + 0 + 50)/3
    expected = (200.0 * (1.0 / 3.0) + 0.0 + 200.0 * (2.0 / 8.0)) / 3.0
    assert torch.isclose(SMAPE()(_Y_PRED, _Y_TRUE), torch.tensor(expected))


@pytest.mark.parametrize("metric_factory", [MAE, MSE, RMSE, MAPE, SMAPE, NSE, AlphaNSE, BetaNSE])
def test_poolable_path_matches_forward_on_whole_dataset(metric_factory):
    torch.manual_seed(0)
    batches = [(torch.randn(5, 7, 1), torch.randn(5, 7, 1) + 3.0) for _ in range(4)]
    # +3.0 offset keeps values away from exactly 0 so MAPE/SMAPE's clamp(min=1e-8)
    # never binds and doesn't skew the comparison

    metric = metric_factory()
    accs = metric.new_accumulators()
    for yp, yt in batches:
        metric.accumulate(accs, yp, yt)
    pooled = metric.pooled_value(accs)

    yp_all = torch.cat([b[0] for b in batches], dim=0)
    yt_all = torch.cat([b[1] for b in batches], dim=0)
    direct = metric(yp_all, yt_all).item()

    assert math.isclose(pooled, direct, rel_tol=1e-5, abs_tol=1e-6)


def test_rmse_pooled_differs_from_naive_per_batch_average():
    # one small batch with a huge error, one large batch with a tiny error — naive
    # per-batch averaging weighs both batches equally regardless of size, which is
    # wrong; the pooled (sqrt of pooled mean squared error) value is the fix
    batch_a = (torch.zeros(2, 1), torch.full((2, 1), 10.0))
    batch_b = (torch.zeros(20, 1), torch.full((20, 1), 0.1))

    metric = RMSE()
    accs = metric.new_accumulators()
    metric.accumulate(accs, *batch_a)
    metric.accumulate(accs, *batch_b)
    pooled = metric.pooled_value(accs)

    naive_per_batch_avg = (metric(*batch_a).item() + metric(*batch_b).item()) / 2.0

    yp_all = torch.cat([batch_a[0], batch_b[0]])
    yt_all = torch.cat([batch_a[1], batch_b[1]])
    exact = metric(yp_all, yt_all).item()

    assert math.isclose(pooled, exact, rel_tol=1e-5)
    assert abs(pooled - naive_per_batch_avg) > 1.0  # meaningfully different, not just fp noise


def test_nse_pooled_differs_from_naive_per_batch_average_with_varying_local_variance():
    # a low-variance ("baseflow-like") batch alongside a high-variance batch — NSE's
    # denominator is the batch's own local variance, so naive per-batch averaging is
    # biased by whichever batch happens to have less local spread; pooling fixes this
    torch.manual_seed(1)
    low_var_true = torch.full((10, 1), 5.0) + torch.randn(10, 1) * 0.01
    low_var_pred = low_var_true + 0.05
    high_var_true = torch.randn(10, 1) * 10
    high_var_pred = high_var_true + torch.randn(10, 1)

    metric = NSE()
    accs = metric.new_accumulators()
    metric.accumulate(accs, low_var_pred, low_var_true)
    metric.accumulate(accs, high_var_pred, high_var_true)
    pooled = metric.pooled_value(accs)

    naive = (metric(low_var_pred, low_var_true).item() + metric(high_var_pred, high_var_true).item()) / 2.0

    yp_all = torch.cat([low_var_pred, high_var_pred])
    yt_all = torch.cat([low_var_true, high_var_true])
    exact = metric(yp_all, yt_all).item()

    assert math.isclose(pooled, exact, rel_tol=1e-4)
    assert abs(pooled - naive) > 0.05


@pytest.mark.parametrize("metric_cls", [RMSE, MAPE, SMAPE, NSE, AlphaNSE, BetaNSE])
def test_reweight_raises_for_metrics_that_dont_override_error(metric_cls):
    # only classes that override _error() (MAE, MSE — and, in the probabilistic
    # family, QuantileLoss/ExpectileLoss) opt into reweight=True; everything else
    # must reject it at construction rather than silently ignoring the flag
    with pytest.raises(ValueError):
        metric_cls(reweight=True)


@pytest.mark.parametrize("metric_cls", [MAE, MSE])
def test_reweight_accepted_for_metrics_that_override_error(metric_cls):
    metric_cls(reweight=True)  # must not raise


@pytest.mark.parametrize("metric_cls,power", [(MAE, 1), (MSE, 2)])
def test_reweight_applies_documented_per_series_weight_formula(metric_cls, power):
    # weight = 1 / (group_std + eps) ** power — verified directly against the
    # documented formula, not by re-deriving it from the class's own code
    eps = 0.1
    group_std = torch.tensor([1.0, 10.0])
    metric = metric_cls(reweight=True, eps=eps)
    metric.bind(group_std)

    y_pred = torch.tensor([[2.0], [2.0], [2.0], [2.0]])
    y_true = torch.tensor([[1.0], [1.0], [1.0], [1.0]])  # constant |diff|=1 for every sample
    group_idx = torch.tensor([0, 0, 1, 1])

    got = metric(y_pred, y_true, group_idx).item()

    err = 1.0 if power == 1 else 1.0  # |diff|=1 -> MAE error=1, MSE error=1**2=1 either way
    w0 = 1.0 / (group_std[0].item() + eps) ** power
    w1 = 1.0 / (group_std[1].item() + eps) ** power
    expected = (err * w0 + err * w0 + err * w1 + err * w1) / 4.0

    assert math.isclose(got, expected, rel_tol=1e-5)


# ── NSE family: formula checks against the documented definitions ─────────────

def _expected_nse(pred, true):
    resid_sq = ((true - pred) ** 2).sum()
    denom = ((true - true.mean()) ** 2).sum()
    return (1 - resid_sq / denom).item()


def _expected_alpha_nse(pred, true):
    return (pred.std(unbiased=False) / true.std(unbiased=False)).item()


def _expected_beta_nse(pred, true):
    return ((pred.mean() - true.mean()) / true.std(unbiased=False)).item()


def test_nse_family_formulas_match_documented_definitions():
    torch.manual_seed(2)
    pred = torch.randn(20, 1) * 3 + 1
    true = torch.randn(20, 1) * 3

    assert math.isclose(NSE()(pred, true).item(), _expected_nse(pred, true), rel_tol=1e-5)
    assert math.isclose(AlphaNSE()(pred, true).item(), _expected_alpha_nse(pred, true), rel_tol=1e-5)
    assert math.isclose(BetaNSE()(pred, true).item(), _expected_beta_nse(pred, true), rel_tol=1e-5)


@pytest.mark.parametrize("metric_cls", [NSE, AlphaNSE, BetaNSE])
def test_nse_family_not_differentiable(metric_cls):
    from src.utils.scores_and_losses import assert_differentiable
    with pytest.raises(TypeError):
        assert_differentiable(metric_cls())
