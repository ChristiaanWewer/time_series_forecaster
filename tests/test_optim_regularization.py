import torch

from src.optim import Adam, AdamW, SGD
from src.utils.regularization import L1Regularizer, L2Regularizer, RLN


def _fake_param(value=1.0, grad=0.5):
    p = torch.nn.Parameter(torch.full((3,), value))
    p.grad = torch.full((3,), grad)
    return p


def test_adamw_bind_produces_real_optimizer_with_settings():
    settings = AdamW(lr=0.01, weight_decay=0.05)
    p = _fake_param()
    opt = settings.bind([p])
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]['lr'] == 0.01
    assert opt.param_groups[0]['weight_decay'] == 0.05


def test_adam_and_sgd_bind():
    p1 = _fake_param()
    opt1 = Adam(lr=0.002).bind([p1])
    assert isinstance(opt1, torch.optim.Adam)

    p2 = _fake_param()
    opt2 = SGD(lr=0.1, momentum=0.9).bind([p2])
    assert isinstance(opt2, torch.optim.SGD)
    assert opt2.param_groups[0]['momentum'] == 0.9


def test_optimizer_bind_respects_per_group_overrides():
    # mirrors Forecaster's bias/head groups pinning weight_decay=0.0 regardless of
    # the optimizer's own default
    settings = AdamW(lr=0.01, weight_decay=0.05)
    weights = _fake_param()
    biases = _fake_param()
    opt = settings.bind([{'params': [weights]}, {'params': [biases], 'weight_decay': 0.0}])
    assert opt.param_groups[0]['weight_decay'] == 0.05
    assert opt.param_groups[1]['weight_decay'] == 0.0


def test_l1_regularizer_step_adds_signed_penalty():
    reg = L1Regularizer(lambda_=0.1)
    p = _fake_param(value=2.0, grad=0.0)
    reg.step([p])
    assert torch.allclose(p.grad, torch.full((3,), 0.1))


def test_l2_regularizer_step_adds_value_scaled_penalty():
    reg = L2Regularizer(gamma=0.1)
    p = _fake_param(value=2.0, grad=0.0)
    reg.step([p])
    assert torch.allclose(p.grad, torch.full((3,), 0.2))


def test_rln_bind_and_step_mutates_grad():
    reg = RLN(theta=-4.0, lr=0.01)
    p = _fake_param(value=1.0, grad=0.3)
    reg.bind([p], lr=0.01)
    grad_before = p.grad.clone()
    reg.step([p])
    assert not torch.allclose(p.grad, grad_before)  # regularization term was added


def test_rln_state_roundtrip():
    reg = RLN(theta=-4.0, lr=0.01)
    p = _fake_param(value=1.0, grad=0.3)
    reg.bind([p], lr=0.01)
    reg.step([p])
    state = reg.get_state()

    reg2 = RLN(theta=-4.0, lr=0.01)
    p2 = _fake_param(value=1.0, grad=0.3)
    reg2.bind([p2], lr=0.01)
    reg2.load_state(state)
    assert torch.allclose(reg2._lambdas[0], reg._lambdas[0])
