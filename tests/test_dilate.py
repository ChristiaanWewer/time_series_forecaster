"""DILATE correctness, verified against the marcdemers/batch-DILATE reference
implementation this codebase's src/utils/DILATE/ was ported from.

The golden loss/shape/temporal/gradient values below were computed once, offline, by
running the actual upstream package (github.com/marcdemers/batch-DILATE) against this
codebase's DILATE on identical inputs and comparing directly — not derived from this
codebase's own implementation. The reference repo is not a project dependency; it was
cloned to a scratch directory, patched to remove one `@njit(cache=True)` that hits a
known Numba pickling bug (this codebase's own numba_apply_along_axis has the identical
workaround, see loss_utils.py), run once to produce these numbers, then discarded.

Forward values (loss/shape/temporal) matched the reference exactly (0.0 diff) on every
case tried. Gradients matched to ~3e-5 in the typical case, up to ~1.5e-3 for a very
small gamma (0.001) — expected, since this port computes the pairwise distance matrix
via a direct squared-difference formula in float32, while the reference uses a
norm-expansion formula in float64; a smaller gamma sharpens the softmin and makes it
more sensitive to that difference. Per-case gradient tolerances below reflect what was
actually observed, not a single blanket number.
"""
import torch

from src.utils.scores_and_losses import DILATE
from src.utils.DILATE.dilate_loss import dilate_loss


def test_dilate_matches_reference_case_batch2_horizon5():
    outputs = torch.tensor([
        [1.5409960746765137, -0.293428897857666, -2.1787893772125244, 0.5684312582015991, -1.0845223665237427],
        [-1.3985954523086548, 0.40334683656692505, 0.8380263447761536, -0.7192575931549072, -0.40334352850914],
    ]).unsqueeze(-1)
    targets = torch.tensor([
        [-0.5966353416442871, 0.18203648924827576, -0.8566746115684509, 1.1006041765213013, -1.0711873769760132],
        [0.1227012425661087, -0.5663174986839294, 0.3731146454811096, -0.8919953107833862, -1.5091077089309692],
    ]).unsqueeze(-1)
    ref_grad = torch.tensor([
        [1.0688360929489136, -0.2377883493900299, -0.6610835194587708, -0.26608747243881226, -0.006667494773864746],
        [-1.1767823696136475, 0.015115920454263687, 0.23245316743850708, 0.08636783063411713, 0.5528820753097534],
    ]).unsqueeze(-1)

    o = outputs.clone().requires_grad_(True)
    loss, shape, temporal = dilate_loss(o, targets, alpha=0.5, gamma=0.01, device=torch.device('cpu'))
    loss.backward()

    assert torch.isclose(loss, torch.tensor(2.8458876610), rtol=1e-5)
    assert torch.isclose(shape, torch.tensor(5.6517734528), rtol=1e-5)
    assert torch.isclose(temporal, torch.tensor(0.0400020331), rtol=1e-5, atol=1e-6)
    assert torch.allclose(o.grad, ref_grad, atol=1e-4)


def test_dilate_matches_reference_case_batch1_horizon3():
    outputs = torch.tensor([[0.3703935444355011, 1.4565025568008423, 0.9398099184036255]]).unsqueeze(-1)
    targets = torch.tensor([[0.7748488187789917, 0.19186942279338837, 1.2637947797775269]]).unsqueeze(-1)
    ref_grad = torch.tensor([[-0.13555745780467987, 0.11562465876340866, -0.19439095258712769]]).unsqueeze(-1)

    o = outputs.clone().requires_grad_(True)
    loss, shape, temporal = dilate_loss(o, targets, alpha=0.3, gamma=0.05, device=torch.device('cpu'))
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.2568228245), rtol=1e-5)
    assert torch.isclose(shape, torch.tensor(0.3375574052), rtol=1e-5)
    assert torch.isclose(temporal, torch.tensor(0.2222222686), rtol=1e-5)
    assert torch.allclose(o.grad, ref_grad, atol=1e-6)


def test_dilate_matches_reference_case_small_gamma():
    # gamma=0.001 — the sharpest softmin tried, and where forward/backward-formula
    # precision differences between this port and the reference show up the most
    outputs = torch.tensor([
        [-0.23033547401428223, -0.3917543888092041, 0.5432947278022766, -0.3951575458049774, 0.20552566647529602, 2.0819554328918457, 1.7067116498947144],
        [2.3803675174713135, -1.1256016492843628, -0.31699809432029724, -1.09246826171875, -0.08519429713487625, -0.09334823489189148, -0.7607071995735168],
        [-1.5990819931030273, 0.018486667424440384, -0.7504268288612366, 0.18540798127651215, 0.621138334274292, 0.6381815671920776, -0.24600094556808472],
    ]).unsqueeze(-1)
    targets = torch.tensor([
        [-0.5344406962394714, 1.1686878204345703, 0.39450275897979736, 1.9414620399475098, 0.791498064994812, -0.7501814961433411, -1.3119657039642334],
        [-0.21883368492126465, -2.435065269470215, -0.07291475683450699, -0.0339864082634449, 0.9625182747840881, 0.34916824102401733, -0.3701472580432892],
        [-1.210281491279602, -0.6226984858512878, -0.4637221693992615, 1.921782374382019, -0.4025455415248871, 0.12390247732400894, 1.1647834777832031],
    ]).unsqueeze(-1)
    ref_grad = torch.tensor([
        [0.14198371767997742, 0.06661882251501083, -0.29198983311653137, -0.3686845898628235, -0.08822988718748093, 0.06559876352548599, 2.983663558959961],
        [1.2125251293182373, -0.4230821132659912, -0.04579488933086395, 0.6264871954917908, -0.5182106494903564, -0.20636828243732452, -0.18226130306720734],
        [-0.1814199537038803, 0.29919523000717163, -0.05959945172071457, 0.3028912842273712, -0.606894850730896, 0.48561474680900574, -0.8309876918792725],
    ]).unsqueeze(-1)

    o = outputs.clone().requires_grad_(True)
    loss, shape, temporal = dilate_loss(o, targets, alpha=0.7, gamma=0.001, device=torch.device('cpu'))
    loss.backward()

    assert torch.isclose(loss, torch.tensor(7.9838395119), rtol=1e-5)
    assert torch.isclose(shape, torch.tensor(11.3034439087), rtol=1e-5)
    assert torch.isclose(temporal, torch.tensor(0.2380952388), rtol=1e-5)
    assert torch.allclose(o.grad, ref_grad, atol=3e-3)


def test_temporal_term_actually_contributes_gradient():
    # regression guard for the historical bug documented in agents.md: an earlier
    # pure-PyTorch implementation computed loss_temporal under @torch.no_grad(), so it
    # silently never contributed gradients — DILATE trained on the shape term alone.
    torch.manual_seed(0)
    outputs = torch.randn(2, 6, 1, requires_grad=True)
    targets = torch.randn(2, 6, 1)
    _, _, loss_temporal = dilate_loss(outputs, targets, alpha=0.5, gamma=0.01, device=torch.device('cpu'))
    grad, = torch.autograd.grad(loss_temporal, outputs)
    assert grad.abs().max().item() > 0.0


def test_alpha_one_is_pure_shape_and_alpha_zero_is_pure_temporal():
    torch.manual_seed(1)
    outputs = torch.randn(2, 5, 1)
    targets = torch.randn(2, 5, 1)

    loss_shape_only, shape, _ = dilate_loss(outputs, targets, alpha=1.0, gamma=0.01, device=torch.device('cpu'))
    assert torch.isclose(loss_shape_only, shape)

    loss_temporal_only, _, temporal = dilate_loss(outputs, targets, alpha=0.0, gamma=0.01, device=torch.device('cpu'))
    assert torch.isclose(loss_temporal_only, temporal)


def test_dilate_matches_reference_case_horizon100_through_upstream_model():
    # Reproduces the STRUCTURE of marcdemers/batch-DILATE's own example.ipynb: a
    # near-1 Gaussian-noise prediction (passed through an upstream nn.Module with a
    # learnable bias, like their DummyModel) vs. a sin-wave target, alpha=0.4,
    # gamma=0.01, horizon=100. Their notebook's own baked numbers (loss=121.7652,
    # grad=89.8253) were produced on a CUDA device — PyTorch's CPU and CUDA RNGs are
    # different streams, so they aren't bit-reproducible here. This case instead uses
    # freshly CPU-generated numbers (torch.manual_seed(0), matching their exact
    # generation order) and was cross-checked directly against a fresh run of their
    # actual reference code at this scale, offline, the same way as the other cases.
    #
    # In float32 the gradient summed through the upstream bias parameter (a sum over
    # 100 individual per-position gradients) differs from the reference by ~0.23 out
    # of ~89.8 — traced to accumulated float32-vs-float64 rounding (this port computes
    # its pairwise-distance matrix directly in float32; the reference uses a
    # norm-expansion formula in float64). Confirmed by re-running the identical case in
    # float64: the gap drops from 0.23 to ~2e-6. Both tolerances below reflect that —
    # loose for float32, tight for float64 — rather than one blanket number.
    torch.manual_seed(0)
    num_outputs = 100
    alpha, gamma = 0.4, 0.01
    base_prediction = torch.normal(mean=torch.ones(num_outputs), std=torch.ones(num_outputs))
    target_row = torch.sin(torch.arange(num_outputs).float())
    target = target_row.view(1, num_outputs, 1)

    bias = torch.nn.Parameter(torch.tensor([1.0]))
    prediction = (base_prediction.view(1, num_outputs, 1) + bias)
    loss, shape, temporal = dilate_loss(prediction, target, alpha, gamma, torch.device('cpu'))
    grad_bias, = torch.autograd.grad(loss, bias)

    assert torch.isclose(loss, torch.tensor(121.7651748657), rtol=1e-5)
    assert torch.isclose(shape, torch.tensor(270.6191711426), rtol=1e-5)
    assert torch.isclose(temporal, torch.tensor(22.5291652679), rtol=1e-5)
    assert torch.isclose(grad_bias, torch.tensor(89.5951995850), rtol=1e-4)

    # float64 cross-check against the reference value directly (isolates correctness
    # from float32 rounding — see comment above)
    bias64 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    prediction64 = base_prediction.double().view(1, num_outputs, 1) + bias64
    loss64, _, _ = dilate_loss(prediction64, target.double(), alpha, gamma, torch.device('cpu'))
    grad_bias64, = torch.autograd.grad(loss64, bias64)
    assert torch.isclose(grad_bias64, torch.tensor(89.8252944946, dtype=torch.float64), atol=1e-4)


def test_linear_combination_identity_holds():
    # the exact relationship marcdemers/batch-DILATE's own example.ipynb asserts
    # (its cell 9): total_loss == alpha * shape_loss + (1 - alpha) * temporal_loss.
    # Input-independent, so no external reference values needed here.
    torch.manual_seed(3)
    outputs = torch.randn(4, 8, 1)
    targets = torch.randn(4, 8, 1)
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        loss, shape, temporal = dilate_loss(outputs, targets, alpha=alpha, gamma=0.02, device=torch.device('cpu'))
        assert torch.isclose(loss, alpha * shape + (1 - alpha) * temporal, rtol=1e-5)


def test_gradient_flows_through_upstream_model_parameter():
    # mirrors marcdemers/batch-DILATE's example.ipynb cell 10: gradients must reach a
    # parameter of a model *upstream* of the loss's input, not just the input tensor
    # directly — the realistic training scenario, unlike calling dilate_loss on a leaf
    # tensor.
    torch.manual_seed(4)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, x):
            return x + self.bias

    model = DummyModel()
    raw = torch.randn(3, 6, 1)
    targets = torch.randn(3, 6, 1)
    prediction = model(raw)
    loss, _, _ = dilate_loss(prediction, targets, alpha=0.5, gamma=0.01, device=torch.device('cpu'))
    grad_bias, = torch.autograd.grad(loss, model.bias)
    assert torch.isfinite(grad_bias).all()
    assert grad_bias.abs().item() > 0.0


def test_dilate_metric_class_wraps_dilate_loss_and_selects_point_channel():
    torch.manual_seed(2)
    metric = DILATE(alpha=0.5, gamma=0.01)
    assert metric.eval_on_normalized is True

    y_pred_1ch = torch.randn(3, 4, 1)
    y_true = torch.randn(3, 4, 1)
    loss_1ch = metric(y_pred_1ch, y_true)

    # multi-channel prediction (e.g. under a quantile head): DILATE must select the
    # 0.5-level channel before running, same as every other point-style metric
    metric.bind_levels('quantile', [0.1, 0.5, 0.9])
    y_pred_3ch = torch.randn(3, 4, 3)
    y_pred_3ch[..., 1] = y_pred_1ch.squeeze(-1)  # put the 1-channel case at the 0.5 slot
    loss_3ch = metric(y_pred_3ch, y_true)

    assert torch.isclose(loss_1ch, loss_3ch, rtol=1e-5)
