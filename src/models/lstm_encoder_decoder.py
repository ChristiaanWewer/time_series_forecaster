import torch
import torch.nn as nn

from src.models.base import BaseForecastModel


class LSTMEncoderDecoder(BaseForecastModel):
    def __init__(self, config):
        super().__init__(config)

        encoder_downscale_layer_size = config['encoder_downscale_layer_size']

        self.encoder_lstm = nn.LSTM(
            input_size=self.historic_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.future_input_size + encoder_downscale_layer_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.linear_downscale_layer = nn.Linear(in_features=self.hidden_size, out_features=encoder_downscale_layer_size)
        self.output_head = self.build_output_head('sequence', self.hidden_size)

    def forward(self, X):
        x_h, x_f = self.embed(X)

        _, (x_encoded_h, x_encoded_c) = self.encoder_lstm(x_h)

        x_encoded_linear_downscaled = self.linear_downscale_layer(x_encoded_h[-1])

        x_encoded_linear_downscaled = x_encoded_linear_downscaled.unsqueeze(1).repeat(1, x_f.shape[1], 1)
        x_encoded_linear_downscaled = torch.cat((x_encoded_linear_downscaled, x_f), dim=2)

        # [batch_size, seq_len, hidden_size]
        x_decoded, (_, _) = self.decoder_lstm(x_encoded_linear_downscaled, (x_encoded_h, x_encoded_c))
        return self.output_head(x_decoded)
