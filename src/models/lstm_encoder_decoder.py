import torch
import torch.nn as nn

from src.models.base import BaseForecastModel


class LSTMEncoderDecoder(BaseForecastModel):
    def __init__(self, hidden_size, num_layers, dropout_rate, encoder_downscale_layer_size):
        super().__init__(hidden_size, num_layers, dropout_rate)
        self.encoder_downscale_layer_size = encoder_downscale_layer_size

    def _build_backbone(self):
        self.encoder_lstm = nn.LSTM(
            input_size=self.historic_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.future_input_size + self.encoder_downscale_layer_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True,
        )

        self.linear_downscale_layer = nn.Linear(in_features=self.hidden_size, out_features=self.encoder_downscale_layer_size)
        self.output_head = self.build_output_head('sequence', self.hidden_size)

    def forward(self, x_h, x_f):
        self._check_bound()
        _, (x_encoded_h, x_encoded_c) = self.encoder_lstm(x_h)

        x_encoded_linear_downscaled = self.linear_downscale_layer(x_encoded_h[-1])

        x_encoded_linear_downscaled = x_encoded_linear_downscaled.unsqueeze(1).repeat(1, x_f.shape[1], 1)
        x_encoded_linear_downscaled = torch.cat((x_encoded_linear_downscaled, x_f), dim=2)

        # [batch_size, seq_len, hidden_size]
        x_decoded, (_, _) = self.decoder_lstm(x_encoded_linear_downscaled, (x_encoded_h, x_encoded_c))
        return self.output_head(x_decoded)
