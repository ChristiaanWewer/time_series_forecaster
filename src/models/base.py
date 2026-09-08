import torch.nn as nn

from src.models.heads import build_output_head


class BaseForecastModel(nn.Module):
    """Shared scaffolding every forecasting model builds on, regardless of backbone
    architecture. Constructed with pure architecture settings only (hidden_size/
    num_layers/dropout_rate) — no columns, no embeddings, no output shape, so no real
    layers exist yet. Forecaster calls .bind(...) once it knows the (embedding-derived)
    input sizes and the (loss-derived) output shape; that's what triggers
    _build_backbone(), the subclass hook that actually allocates the LSTM(s)/head.

    Embeddings are never owned by a model — Forecaster builds and applies them, handing
    forward() already-embedded x_h/x_f tensors directly.
    """
    def __init__(self, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self._bound = False

    def bind(self, historic_input_size, future_input_size, forecasting_horizon,
             n_outputs, output_kind, output_levels):
        if self._bound:
            raise RuntimeError(f"{type(self).__name__} is already bound")
        self.historic_input_size = historic_input_size
        self.future_input_size = future_input_size
        self.forecasting_horizon = forecasting_horizon
        self.n_outputs = n_outputs
        self.output_kind = output_kind
        self.output_levels = output_levels
        self._build_backbone()
        self._bound = True
        return self

    def _build_backbone(self):
        raise NotImplementedError

    def _check_bound(self):
        if not self._bound:
            raise RuntimeError(f"{type(self).__name__} must be bound before use — call .bind(...)")

    def build_output_head(self, representation, in_features, horizon=None):
        return build_output_head(
            representation, in_features, horizon, self.n_outputs,
            self.output_kind, self.output_levels,
        )

    def parameter_breakdown(self):
        """Per-component parameter counts: a backbone row (labeled by this model's
        class name — everything not under output_head., so it works unchanged for any
        future non-LSTM architecture), a head row (labeled by the output head's class
        name), and a total row. Embedding rows aren't included here — embeddings live
        on Forecaster now, which prepends its own rows before this model's.
        """
        self._check_bound()
        backbone = sum(
            p.numel() for name, p in self.named_parameters()
            if not name.startswith('output_head.')
        )
        rows = [(type(self).__name__, backbone)]
        rows.append((type(self.output_head).__name__, sum(p.numel() for p in self.output_head.parameters())))
        rows.append(('total', sum(p.numel() for p in self.parameters())))
        return rows
