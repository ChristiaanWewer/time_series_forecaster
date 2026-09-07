import torch.nn as nn

from src.models.embeddings import build_embeddings
from src.models.heads import build_output_head


class BaseForecastModel(nn.Module):
    """Shared scaffolding every forecasting model builds on, regardless of backbone
    architecture. Centralizes the config that's either injected unconditionally by
    Forecaster (historic_cols/future_cols/forecasting_horizon/n_outputs/output_kind/
    output_levels — see Forecaster.__init__) or duplicated identically across model
    subclasses today (hidden_size/num_layers/dropout_rate). Values that only make
    sense for one architecture (e.g. an encoder-decoder's downscale-layer width) stay
    as plain config[...] reads inside that subclass instead.

    embedding_h/embedding_f are always built here (identity passthrough when
    model_config has no 'embeddings' key) so every model gets the same input-embedding
    mechanism for free, without each model author having to wire it in.
    """
    def __init__(self, config):
        super().__init__()

        self.embedding_h, self.embedding_f, self._embedding_group_networks = build_embeddings(
            config, config['historic_cols'], config['future_cols']
        )
        self.historic_input_size = self.embedding_h.output_size
        self.future_input_size = self.embedding_f.output_size

        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.forecasting_horizon = config['forecasting_horizon']
        self.n_outputs = config.get('n_outputs', 1)
        self.output_kind = config.get('output_kind', 'point')
        self.output_levels = config.get('output_levels')

    def embed(self, X):
        return self.embedding_h(X['x_h']), self.embedding_f(X['x_f'])

    def build_output_head(self, representation, in_features, horizon=None):
        return build_output_head(
            representation, in_features, horizon, self.n_outputs,
            self.output_kind, self.output_levels,
        )

    def parameter_breakdown(self):
        """Per-component parameter counts: one row per configured embedding group
        (labeled by its config 'name'), a backbone row (labeled by this model's class
        name — everything not under embedding_h./embedding_f./output_head., so it
        works unchanged for any future non-LSTM architecture), a head row (labeled by
        the output head's class name), and a total row.
        """
        rows = [(name, sum(p.numel() for p in net.parameters()))
                for name, net in self._embedding_group_networks]

        backbone = sum(
            p.numel() for name, p in self.named_parameters()
            if not name.startswith(('embedding_h.', 'embedding_f.', 'output_head.'))
        )
        rows.append((type(self).__name__, backbone))
        rows.append((type(self.output_head).__name__, sum(p.numel() for p in self.output_head.parameters())))
        rows.append(('total', sum(p.numel() for p in self.parameters())))
        return rows
