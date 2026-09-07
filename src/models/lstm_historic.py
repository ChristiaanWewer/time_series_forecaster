import torch.nn as nn

from src.models.base import BaseForecastModel


class LSTMHistoric(BaseForecastModel):
    def __init__(self, config):
        super().__init__(config)

        self.historic_lstm = nn.LSTM(
            input_size=self.historic_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.output_head = self.build_output_head('pooled', self.hidden_size, self.forecasting_horizon)

    def forward(self, X):
        x_h = self.embedding_h(X['x_h'])
        x_encoded_h, _ = self.historic_lstm(x_h)
        x_encoded_h = x_encoded_h[:, -1, :]
        x_encoded_h = self.dropout_layer(x_encoded_h)
        return self.output_head(x_encoded_h)
