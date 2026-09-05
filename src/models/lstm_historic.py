import torch.nn as nn

from src.models.heads import build_output_head


class LSTMHistoric(nn.Module):
    def __init__(self, config):
        super(LSTMHistoric, self).__init__()

        LSTM_hidden_size = config['LSTM_hidden_size']
        LSTM_num_layers = config['LSTM_num_layers']
        dropout_rate = config['dropout_rate']
        LSTM_1_input_size = len(config['historic_cols'])
        forecasting_horizon = config['forecasting_horizon']
        n_outputs = config.get('n_outputs', 1)

        self.historic_lstm = nn.LSTM(
            input_size=LSTM_1_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=LSTM_num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.output_head = build_output_head(
            'pooled', LSTM_hidden_size, forecasting_horizon, n_outputs,
            config.get('output_kind', 'point'), config.get('output_levels'),
        )

    def forward(self, X):
        x_encoded_h, _ = self.historic_lstm(X['x_h'])
        x_encoded_h = x_encoded_h[:, -1, :]
        x_encoded_h = self.dropout_layer(x_encoded_h)
        return self.output_head(x_encoded_h)
