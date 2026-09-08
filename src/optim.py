import torch


class _OptimizerSettings:
    """Base class for Forecaster's optimizer settings wrappers. A real torch.optim
    optimizer can't be constructed without `params` (Optimizer.__init__ requires it,
    no default), and `params` don't exist until Forecaster has built and bound the
    model/embeddings — so these wrappers hold just the optimizer's own hyperparameters
    (typed, one class per torch.optim class, so they're autocomplete-friendly against
    that class's real signature) and .bind(params) is what actually constructs the
    real thing, once Forecaster has real parameters to hand it. Unlike a regularizer's
    bind(), this needs no extra setup step of its own — it's a plain, one-shot
    construction call, not stateful — but keeping it a typed class (rather than e.g. a
    functools.partial) is what lets Forecaster save "which optimizer, which settings"
    into the checkpoint via plain attribute access.
    """
    _torch_cls: type[torch.optim.Optimizer]

    def bind(self, params) -> torch.optim.Optimizer:
        return self._torch_cls(params, **vars(self))


class Adam(_OptimizerSettings):
    _torch_cls = torch.optim.Adam

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay


class AdamW(_OptimizerSettings):
    _torch_cls = torch.optim.AdamW

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay


class SGD(_OptimizerSettings):
    _torch_cls = torch.optim.SGD

    def __init__(self, lr, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
