import torch
import torch.nn as nn


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoderDecoder, self).__init__()

        LSTM_hidden_size = config['LSTM_hidden_size']
        LSTM_num_layers = config['LSTM_num_layers']
        dropout_rate = config['dropout_rate']
        encoder_downscale_layer_size = config['encoder_downscale_layer_size']
        LSTM_1_input_size = len(config['historic_cols'])
        LSTM_2_input_size = len(config['future_cols']) + encoder_downscale_layer_size

        self.encoder_lstm = nn.LSTM(
            input_size=LSTM_1_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=LSTM_num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=LSTM_2_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=LSTM_num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self.output_layer = nn.Linear(in_features=LSTM_hidden_size, out_features=1)
        self.linear_downscale_layer = nn.Linear(in_features=LSTM_hidden_size, out_features=encoder_downscale_layer_size)

    def forward(self, X):

        _, (x_encoded_h, x_encoded_c) = self.encoder_lstm(X['x_h'])

        x_encoded_linear_downscaled = self.linear_downscale_layer(x_encoded_h[-1])

        x_encoded_linear_downscaled = x_encoded_linear_downscaled.unsqueeze(1).repeat(1, X['x_f'].shape[1], 1)
        x_encoded_linear_downscaled = torch.cat((x_encoded_linear_downscaled, X['x_f']), dim=2)

        # [batch_size, seq_len, hidden_size]
        x_decoded, (_, _) = self.decoder_lstm(x_encoded_linear_downscaled, (x_encoded_h, x_encoded_c))
        x = self.output_layer(x_decoded)

        return x
