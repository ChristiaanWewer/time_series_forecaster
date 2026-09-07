import torch
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
    """Per-timestep embedding network for one group of raw input variables. A plain
    nn.Linear applied at every timestep when num_layers=1 (nn.Linear broadcasts over
    every leading dim for free, so this needs no explicit time loop — the same trick
    SequenceHead uses in heads.py); secretly an MLP with GELU between hidden layers
    when num_layers>1. Matches neuralhydrology's embedding-network convention:
    variables are embedded independently and identically at every timestep.
    output_activation shapes the final embedding — e.g. 'gelu' for a continuous
    embedding, 'softmax' for a categorical one — see build_embeddings() below, which
    is what actually decides a group's variables/dims from model_config.
    """
    def __init__(self, in_features, embedding_dim, num_layers=1, hidden_size=None, output_activation=None):
        super().__init__()
        hidden_size = hidden_size or embedding_dim
        dims = [in_features] + [hidden_size] * (num_layers - 1) + [embedding_dim]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        layers.append(_resolve_output_activation(output_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InputEmbedding(nn.Module):
    """Applies every configured embedding group to one input side (historic or
    future) and concatenates their outputs, in config order, followed by the
    untouched passthrough columns — raw columns claimed by no group, in their
    original order — producing the tensor a backbone actually consumes as x_h/x_f.
    A column may be claimed by more than one group (each gets its own copy of that
    column as part of its input); only membership in *some* group excludes it from
    the passthrough tail. See build_embeddings() below, which builds one of these
    per side.

    `networks` is parallel to `group_specs`: None at a position means that group has
    no variables on this side (e.g. a group defined with historic=False) and is
    skipped entirely here.
    """
    def __init__(self, cols, group_specs, networks):
        super().__init__()
        self.networks = nn.ModuleList()
        self._group_idx: list[list[int]] = []
        output_size = 0
        claimed = set()
        for spec, net in zip(group_specs, networks):
            if net is None:
                continue
            idx = [cols.index(v) for v in spec['variables']]
            self.networks.append(net)
            self._group_idx.append(idx)
            output_size += spec['embedding_dim']
            claimed.update(idx)
        self._passthrough_idx = [i for i in range(len(cols)) if i not in claimed]
        self.output_size = output_size + len(self._passthrough_idx)

    def forward(self, x):
        parts = [net(x[..., idx]) for net, idx in zip(self.networks, self._group_idx)]
        if self._passthrough_idx:
            parts.append(x[..., self._passthrough_idx])
        return torch.cat(parts, dim=-1) if parts else x


def build_embeddings(config, historic_cols, future_cols):
    """Builds the (embedding_h, embedding_f, group_networks) triple a model should use
    in place of raw X['x_h']/X['x_f'], given model_config['embeddings'] — a list of
    group specs:

        {
            'name': str,                # required, unique — identifies this group in
                                         # parameter-count reports (see
                                         # BaseForecastModel.parameter_breakdown)
            'variables': [...],         # column names, checked explicitly against
                                         # historic_cols and/or future_cols below
            'embedding_dim': int,
            'num_layers': int = 1,      # >1 makes this group's network an MLP (GELU
                                         # between hidden layers)
            'hidden_size': int = embedding_dim,
            'output_activation': None | 'gelu' | 'softmax' = None,
            'historic': bool,           # required — apply this group to the historic
                                         # side?
            'future': bool,             # required — apply this group to the future
                                         # side?
        }

    historic/future are explicit and independent: a group with historic=True must have
    every one of its `variables` present in historic_cols (else ValueError — no silent
    partial matching), and likewise future=True against future_cols. A group with both
    False is a config error (ValueError). A group with both True gets a single shared
    EmbeddingNetwork instance applied identically to its historic and future
    occurrences — weight sharing is automatic whenever a group spans both sides, not a
    separate opt-in. Independent (unshared) per-side networks over the same columns are
    expressed as two separate groups instead — one historic-only, one future-only.

    `name` must be present and unique across the config's groups; it's how a group is
    identified in parameter-count reporting.

    No 'embeddings' key (or an empty list) makes both returned modules pure
    passthroughs — output_size == len(cols) — so models that don't opt in are
    unaffected.
    """
    group_specs = config.get('embeddings') or []

    names = [spec.get('name') for spec in group_specs]
    for i, name in enumerate(names):
        if not name:
            raise ValueError(f"embedding group {i} is missing a required non-empty 'name'")
    if len(set(names)) != len(names):
        raise ValueError(f"embedding group names must be unique, got: {names}")

    hist_nets, fut_nets, group_networks = [], [], []
    for i, spec in enumerate(group_specs):
        name = spec['name']
        variables = spec['variables']
        embedding_dim = spec['embedding_dim']
        num_layers = spec.get('num_layers', 1)
        hidden_size = spec.get('hidden_size')
        output_activation = spec.get('output_activation')

        if 'historic' not in spec or 'future' not in spec:
            raise ValueError(f"embedding group '{name}' {variables} must specify both 'historic' and 'future' (bool)")
        historic = spec['historic']
        future = spec['future']
        if not historic and not future:
            raise ValueError(f"embedding group '{name}' {variables} must have historic=True and/or future=True")

        if historic:
            missing = [v for v in variables if v not in historic_cols]
            if missing:
                raise ValueError(f"embedding group '{name}' {variables}: historic=True but missing from historic_cols: {missing}")
        if future:
            missing = [v for v in variables if v not in future_cols]
            if missing:
                raise ValueError(f"embedding group '{name}' {variables}: future=True but missing from future_cols: {missing}")

        if historic and future:
            net = EmbeddingNetwork(len(variables), embedding_dim, num_layers, hidden_size, output_activation)
            hist_nets.append(net)
            fut_nets.append(net)
        else:
            net = EmbeddingNetwork(len(variables), embedding_dim, num_layers, hidden_size, output_activation)
            hist_nets.append(net if historic else None)
            fut_nets.append(net if future else None)
        group_networks.append((name, net))

    embedding_h = InputEmbedding(historic_cols, group_specs, hist_nets)
    embedding_f = InputEmbedding(future_cols, group_specs, fut_nets)
    return embedding_h, embedding_f, group_networks
