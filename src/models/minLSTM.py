import torch 
import torch.nn as nn 
import torch.nn.functional as F


class minLSTM(nn.Module):
    def __init__(self, input_size:int,hidden_size:int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_f = nn.Linear(input_size,hidden_size)
        self.linear_i = nn.Linear(input_size,hidden_size)
        self.linear_h = nn.Linear(input_size,hidden_size)
    
    @staticmethod
    def g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
    
    @staticmethod
    def log_g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
    
    @staticmethod
    def parallel_scan_log(log_coeffs:torch.Tensor, log_values:torch.Tensor) -> torch.Tensor:
        # log_coeffs: (batch_size, seq_len, input_size)
        # log_values: (batch_size, seq_len + 1, input_size)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]

    def forward(self, x:torch.Tensor, h_0:torch.Tensor=None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        if(h_0 is None):
            h_0 = torch.zeros((x.size(0),1,self.hidden_size),device=x.device)

        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = self.log_g(h_0) 
        log_tilde_h = self.log_g(self.linear_h(x))
        h = self.parallel_scan_log(log_f,torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        return h

    def sequential_forward(self, x_t:torch.Tensor, h_prev:torch.Tensor=None) -> torch.Tensor:
        # x_t: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)
        if(h_prev is None):
            h_prev = self.g(torch.zeros((x_t.size(0),self.hidden_size),device=x_t.device))

    
        f_t = torch.sigmoid(self.linear_f(x_t))
        i_t = torch.sigmoid(self.linear_i(x_t))
        tilde_h_t = self.g(self.linear_h(x_t))
        f_prime_t = f_t / (f_t + i_t)
        i_prime_t = i_t / (f_t + i_t)
        h_t = f_prime_t * h_prev + i_prime_t * tilde_h_t
        return h_t


class minLSTMHistoric(nn.Module):
    def __init__(self, config):
        super(minLSTMHistoric, self).__init__()
        # lstm
        self.minLSTM_hidden_size = config['minLSTM_hidden_size']
        self.minLSTM_num_layers = config['minLSTM_num_layers']
        self.dropout_rate = config['dropout_rate']
        self.minLSTM_1_input_size = len(config['historic_cols'])
        self.forecasting_horizon = config['forecasting_horizon']

        self.historic_minLSTM = nn.Sequential(
            minLSTM(input_size=self.minLSTM_1_input_size, hidden_size=self.minLSTM_hidden_size),
            nn.Dropout(self.dropout_rate),
            *[nn.Sequential(
                minLSTM(input_size=self.minLSTM_hidden_size, hidden_size=self.minLSTM_hidden_size), 
                nn.Dropout(self.dropout_rate)) for _ in range(self.minLSTM_num_layers - 1)]
        )
        # output layer
        self.output_layer = nn.Linear(in_features=self.minLSTM_hidden_size, out_features=self.forecasting_horizon)

    def forward(self, X):
        # X: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        x_encoded_h = self.historic_minLSTM(X['x_h'])

        x_decoded = self.output_layer(x_encoded_h[:,-1,:]).unsqueeze(-1)
        # print('x_decoded.shape')
        # print(x_decoded.shape)
        return x_decoded

# class minLSTMEncoderDecoder(nn.Module):
#     def __init__(self, config):
#         # lstm
#         self.minLSTM_hidden_size = config['minLSTM_hidden_size']
#         self.minLSTM_num_layers = config['minLSTM_num_layers']
#         self.dropout_rate = config['dropout_rate']
#         # self.encoder_downscale_layer_size = config['encoder_downscale_layer_size']
#         self.minLSTM_1_input_size = len(config['historic_cols']) # number of features in the input data
#         self.minLSTM_2_input_size = len(config['future_cols']) + self.encoder_downscale_layer_size # number of features in the future data

#         # minLSTM_network = nn.Sequential(minLSTM(input_size, hidden_size), nn.Dropout(0.1),
#         #       *[nn.Sequential(minLSTM(hidden_size, hidden_size), nn.Dropout(0.1)) for _ in range(num_layers)]

#         self.encoder_minLSTM = nn.Sequential(
#             minLSTM(input_size=self.minLSTM_1_input_size, hidden_size=self.minLSTM_hidden_size),
#             nn.Dropout(self.dropout_rate),
#             *[nn.Sequential(
#                 minLSTM(input_size=self.minLSTM_hidden_size, hidden_size=self.minLSTM_hidden_size), 
#                 nn.Dropout(self.dropout_rate)) for _ in range(self.minLSTM_num_layers - 1)]
#         )

#         self.decoder_minLSTM = nn.Sequential(
#             minLSTM(input_size=self.minLSTM_2_input_size, hidden_size=self.minLSTM_hidden_size),
#             nn.Dropout(self.dropout_rate),
#             *[nn.Sequential(
#                 minLSTM(input_size=self.minLSTM_hidden_size, hidden_size=self.minLSTM_hidden_size), 
#                 nn.Dropout(self.dropout_rate)) for _ in range(self.minLSTM_num_layers - 1)]
#         )

#         # output layer
#         self.output_layer = nn.Linear(in_features=self.minLSTM_hidden_size, out_features=1)


        







# class __main__:
#     # Test the minLSTM class
#     input_size = 10
#     hidden_size = 20
#     batch_size = 5
#     seq_len = 15

#     model = minLSTM(input_size, hidden_size)
#     x = torch.randn(batch_size, seq_len, input_size)
#     # h_0 = torch.randn(batch_size, 1, hidden_size)

#     output = model(x)
#     print(output.shape)  # Should be (batch_size, seq_len, hidden_size)