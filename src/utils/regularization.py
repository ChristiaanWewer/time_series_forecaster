import torch


class _Regularizer:
    """Base class for Forecaster's pluggable weight regularizers. Never touches the
    loss graph — Forecaster.__training_step calls step() right after loss.backward()
    (p.grad already holds the plain data gradient) and before optimizer.step().
    Subclasses may mutate p.grad in place and hold state across steps (see RLN).
    Only ever sees one weight-only parameter pool — either the embedding weights or
    the backbone weights (see Forecaster's embedding_regularization/model_regularizer,
    both bound separately since embeddings and the model backbone are now two
    independently-owned pieces). Biases and the output head are never regularized.
    """
    def bind(self, params: list[torch.nn.Parameter], lr: float) -> None:
        """Called once by Forecaster, after the relevant module (embeddings or model)
        has been bound and its real parameters exist: the weight-only parameter list,
        and a learning rate (so a regularizer's own rate, if unset, can default to it
        — see RLN)."""
        pass

    def step(self, params: list[torch.nn.Parameter]) -> None:
        raise NotImplementedError

    def get_state(self) -> dict:
        """Checkpointable state beyond __init__'s own settings (e.g. RLN's
        learned per-weight coefficients). Empty for stateless regularizers."""
        return {}

    def load_state(self, state: dict) -> None:
        pass


class RLN(_Regularizer):
    """Regularization Learning Networks (Shavitt & Segal, arXiv:1805.06440) — one
    L1 coefficient learned per weight, jointly with the weights, via the paper's
    Counterfactual Loss gradient. Summary:

        g_t = plain data-loss gradient (p.grad right after loss.backward())
        r_t = exp(lambda_t) * sign(w_t)                        — regularization grad
        w_{t+1} = w_t - eta * (g_t + r_t)                      — the weight update
        lambda_{t+1} = lambda_t + nu * eta * g_{t+1} * r_t     — learns lambda using
            *next* step's g, reusing that step's already-computed gradient (no extra
            backward pass) — this is the paper's efficiency trick.

    lambda is then re-centered so its mean across every weight in its pool equals
    `theta` (a simplex projection) — without this lambda drifts to -inf and
    regularization collapses. theta is the one hyperparameter this replaces the
    whole per-weight search with.

    Args:
        theta: target mean of the per-weight log-coefficients (tune via CV — this
            replaces what would otherwise be a per-weight hyperparameter search).
        lr: learning rate for the coefficients themselves (nu in the paper);
            defaults to whatever Forecaster passes into bind() if not given.
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


class L1Regularizer(_Regularizer):
    """Flat L1 — one shared coefficient for every weight in its pool. The trivial
    case of the same interface RLN uses (one shared lambda instead of one per weight).
    """
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def step(self, params):
        if self.lambda_:
            for p in params:
                p.grad += self.lambda_ * torch.sign(p.data)


class L2Regularizer(_Regularizer):
    """Flat L2 — one shared coefficient for every weight in its pool, applied the same
    way L1Regularizer is (a gradient-space penalty added in step(), not folded into the
    loss graph). Deliberately independent of the optimizer's own weight_decay kwarg —
    the two are not mutually exclusive and may be combined; it's on the caller to avoid
    double-penalizing if that's not intended.
    """
    def __init__(self, gamma=0.01):
        self.gamma = gamma

    def step(self, params):
        if self.gamma:
            for p in params:
                p.grad += self.gamma * p.data
