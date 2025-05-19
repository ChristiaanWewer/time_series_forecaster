import torch
import torch.nn as nn

class SSM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    @staticmethod
    def g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    @staticmethod
    def log_g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, (torch.relu(x) + 0.5).log(), -torch.softplus(-x))