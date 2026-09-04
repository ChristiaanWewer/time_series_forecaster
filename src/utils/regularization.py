import torch

from src.utils.scores_and_losses import parse_spec


REGISTRY: dict = {}  # populated after class definitions


class _Regularizer:
    """Base class for Forecaster's pluggable weight regularizers. Never touches the
    loss graph — Forecaster.__training_step calls step() right after loss.backward()
    (p.grad already holds the plain data gradient) and before optimizer.step().
    Subclasses may mutate p.grad in place and hold state across steps (see RLN).
    Only ever sees the weight-only parameter list (biases excluded — same convention
    Forecaster already uses for L2/weight_decay).
    """
    def bind(self, params: list[torch.nn.Parameter], lr: float) -> None:
        """Called once at Forecaster construction: the weight-only parameter list,
        and the Forecaster's own learning_rate (so a regularizer's own rate, if
        unset, can default to it — see RLN)."""
        pass

    def step(self, params: list[torch.nn.Parameter]) -> None:
        raise NotImplementedError

    def get_state(self) -> dict:
        """Checkpointable state beyond __init__'s own spec kwargs (e.g. RLN's
        learned per-weight coefficients). Empty for stateless regularizers."""
        return {}

    def load_state(self, state: dict) -> None:
        pass


class RLN(_Regularizer):
    """Regularization Learning Networks (Shavitt & Segal, arXiv:1805.06440) — one
    L1 coefficient learned per weight, jointly with the weights, via the paper's
    Counterfactual Loss gradient (see agents.md for the full derivation). Summary:

        g_t = plain data-loss gradient (p.grad right after loss.backward())
        r_t = exp(lambda_t) * sign(w_t)                        — regularization grad
        w_{t+1} = w_t - eta * (g_t + r_t)                      — the weight update
        lambda_{t+1} = lambda_t + nu * eta * g_{t+1} * r_t     — learns lambda using
            *next* step's g, reusing that step's already-computed gradient (no extra
            backward pass) — this is the paper's efficiency trick.

    lambda is then re-centered so its mean across every weight in the network equals
    `theta` (a simplex projection) — without this lambda drifts to -inf and
    regularization collapses network-wide. theta is the one hyperparameter this
    replaces the whole per-weight search with.

    Args:
        theta: target mean of the per-weight log-coefficients (tune via CV — this
            replaces what would otherwise be a per-weight hyperparameter search).
        lr: learning rate for the coefficients themselves (nu in the paper);
            defaults to the Forecaster's own learning_rate if not given.
    """
    def __init__(self, theta=-4.0, lr=None):
        self.theta = theta
        self.lr = lr

    def bind(self, params, lr):
        self._lambdas = [torch.full_like(p, self.theta) for p in params]
        self._prev_r = [None] * len(params)
        self._eta = lr
        self._nu = self.lr if self.lr is not None else lr

    def step(self, params):
        for idx, p in enumerate(params):
            prev_r = self._prev_r[idx]
            if prev_r is not None:  # skip the lambda update on the very first step
                self._lambdas[idx] += self._nu * self._eta * p.grad * prev_r
        total = sum(l.sum() for l in self._lambdas)
        count = sum(l.numel() for l in self._lambdas)
        shift = self.theta - total / count
        for l in self._lambdas:
            l += shift
        for idx, p in enumerate(params):
            # clip so the regularization step can never exceed the weight's own current
            # magnitude — without this, a large exp(lambda) relative to a near-zero
            # weight can repeatedly overshoot past zero and destabilize training (matches
            # the reference Keras implementation's max_lambda = log(|w|) clip, done here
            # directly on exp(lambda) instead of on lambda in log-space, which avoids that
            # implementation's need to patch log(0) = -inf for exactly-zero weights)
            r = torch.exp(self._lambdas[idx]).clamp(max=p.data.abs()) * torch.sign(p.data)
            p.grad += r
            self._prev_r[idx] = r.detach()

    def get_state(self):
        return {'lambdas': [l.detach().cpu() for l in self._lambdas]}

    def load_state(self, state):
        self._lambdas = [t.to(self._lambdas[0].device) for t in state['lambdas']]


class L1(_Regularizer):
    """Flat L1 — one shared coefficient for every weight. The trivial case of the
    same interface RLN uses (one shared lambda instead of one lambda per weight).
    """
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def step(self, params):
        if self.lambda_:
            for p in params:
                p.grad += self.lambda_ * torch.sign(p.data)


def resolve_regularization(spec):
    """None -> None. String forms: 'l1(lambda_=0.01)', 'rln(theta=-4.0, lr=0.01)'."""
    if spec is None:
        return None
    name, kwargs = parse_spec(spec)
    if name not in REGISTRY:
        raise ValueError(f"Unknown regularizer '{spec}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


REGISTRY = {'l1': L1, 'rln': RLN}
