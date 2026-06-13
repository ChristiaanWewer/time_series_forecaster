import torch.nn as nn
import torch

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoderDecoder, self).__init__()

        # lstm
        self.LSTM_hidden_size = config['LSTM_hidden_size']
        self.LSTM_num_layers = config['LSTM_num_layers']
        self.dropout_rate = config['dropout_rate']
        self.encoder_downscale_layer_size = config['encoder_downscale_layer_size']
        self.LSTM_1_input_size = len(config['historic_cols']) # number of features in the input data
        self.LSTM_2_input_size = len(config['future_cols']) + self.encoder_downscale_layer_size # number of features in the future data

        # lstm
        self.encoder_lstm = nn.LSTM(
            input_size=self.LSTM_1_input_size,
            hidden_size=self.LSTM_hidden_size,
            num_layers=self.LSTM_num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.LSTM_2_input_size,
            hidden_size=self.LSTM_hidden_size,
            num_layers=self.LSTM_num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

        # output layer
        self.output_layer = nn.Linear(in_features=self.LSTM_hidden_size, out_features=1)
        self.linear_downscale_layer = nn.Linear(in_features=self.LSTM_hidden_size, out_features=self.encoder_downscale_layer_size)

    def forward(self, X):

        _, (x_encoded_h, x_encoded_c) = self.encoder_lstm(X['x_h'])

        x_encoded_linear_downscaled = self.linear_downscale_layer(x_encoded_h[-1])

        x_encoded_linear_downscaled = x_encoded_linear_downscaled.unsqueeze(1).repeat(1, X['x_f'].shape[1], 1)
        x_encoded_linear_downscaled = torch.cat((x_encoded_linear_downscaled, X['x_f']), dim=2)

        # [batch_size, seq_len, hidden_size]
        x_decoded, (_, _) = self.decoder_lstm(x_encoded_linear_downscaled, (x_encoded_h, x_encoded_c))
        x = self.output_layer(x_decoded)

        return x
    
class LSTMHistoric(nn.Module):
    def __init__(self, config):
        super(LSTMHistoric, self).__init__()

        # lstm
        self.LSTM_hidden_size = config['LSTM_hidden_size']
        self.LSTM_num_layers = config['LSTM_num_layers']
        self.dropout_rate = config['dropout_rate']
        self.LSTM_1_input_size = len(config['historic_cols']) # number of features in the input data
        self.forecasting_horizon = config['forecasting_horizon']

        # lstm
        self.historic_lstm = nn.LSTM(
            input_size=self.LSTM_1_input_size,
            hidden_size=self.LSTM_hidden_size,
            num_layers=self.LSTM_num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.dropout_layer = nn.Dropout(self.dropout_rate)

        # output layer
        self.output_layer = nn.Linear(in_features=self.LSTM_hidden_size, out_features=self.forecasting_horizon)

    def forward(self, X):
        # X: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        x_encoded_h, _ = self.historic_lstm(X['x_h'])
        x_encoded_h = x_encoded_h[:, -1, :]
        x_encoded_h = self.dropout_layer(x_encoded_h)
        x_decoded = self.output_layer(x_encoded_h).unsqueeze(-1)  # (B, hidden) -> (B, horizon, 1)
        # print('x_decoded.shape')
        # print(x_decoded.shape)


        return x_decoded