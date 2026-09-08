import torch.nn as nn


_OUTPUT_ACTIVATIONS = {
    None: nn.Identity,
    'gelu': nn.GELU,
    'softmax': lambda: nn.Softmax(dim=-1),
}


def _resolve_output_activation(spec):
    if spec not in _OUTPUT_ACTIVATIONS:
        raise ValueError(f"Unknown output_activation '{spec}'. Available: {list(_OUTPUT_ACTIVATIONS)}")
    return _OUTPUT_ACTIVATIONS[spec]()


class EmbeddingNetwork(nn.Module):
    """Per-timestep embedding for one named group of raw input variables. Constructed
    with settings only (name/variables/embedding_dim/historic/future/...) — no weights
    exist yet. Forecaster calls .bind(historic_cols, future_cols) once it knows the
    actual column layout; that's what allocates the real Linear/MLP and works out which
    raw column indices this group reads on each side. A group with historic=True and
    future=True shares this one instance (one set of weights) across both sides —
    Forecaster applies it twice, once per side, using the index lists bound below.

    Forecaster owns combining multiple bound EmbeddingNetworks (plus untouched
    passthrough columns) into the final historic/future feature tensors a model
    consumes — this class only ever does its own local Linear/MLP transform on
    whatever slice of columns it's handed.
    """
    def __init__(self, name, variables, embedding_dim, historic, future,
                 num_layers=1, hidden_size=None, output_activation=None):
        super().__init__()
        if not name:
            raise ValueError("EmbeddingNetwork requires a non-empty 'name'")
        if not historic and not future:
            raise ValueError(f"embedding '{name}' {variables} must have historic=True and/or future=True")
        self.name = name
        self.variables = list(variables)
        self.embedding_dim = embedding_dim
        self.historic = historic
        self.future = future
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_activation = output_activation

        self._historic_idx = None
        self._future_idx = None
        self._bound = False

    def bind(self, historic_cols, future_cols):
        if self._bound:
            raise RuntimeError(f"embedding '{self.name}' is already bound")

        if self.historic:
            missing = [v for v in self.variables if v not in historic_cols]
            if missing:
                raise ValueError(f"embedding '{self.name}' {self.variables}: historic=True but missing from historic_cols: {missing}")
            self._historic_idx = [historic_cols.index(v) for v in self.variables]
        if self.future:
            missing = [v for v in self.variables if v not in future_cols]
            if missing:
                raise ValueError(f"embedding '{self.name}' {self.variables}: future=True but missing from future_cols: {missing}")
            self._future_idx = [future_cols.index(v) for v in self.variables]

        hidden_size = self.hidden_size or self.embedding_dim
        in_features = len(self.variables)
        dims = [in_features] + [hidden_size] * (self.num_layers - 1) + [self.embedding_dim]
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < self.num_layers - 1:
                layers.append(nn.GELU())
        layers.append(_resolve_output_activation(self.output_activation))
        self.net = nn.Sequential(*layers)

        self._bound = True
        return self

    def forward(self, x):
        if not self._bound:
            raise RuntimeError(f"embedding '{self.name}' must be bound before use — call .bind(historic_cols, future_cols)")
        return self.net(x)
