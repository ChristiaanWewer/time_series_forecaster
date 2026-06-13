import torch
from torch.utils.data import TensorDataset
from typing import List
import pandas as pd


class TimeSeriesDataPreprocessor:
    def __init__(self,
                 historic_input_sequence_length: int,
                 forecasting_horizon: int,
                 historic_cols: List[str],
                 future_cols: List[str],
                 target_col: str,
                 dtype: torch.dtype = torch.float32) -> None:

        self.historic_input_sequence_length = historic_input_sequence_length
        self.forecasting_horizon = forecasting_horizon
        self.historic_cols = historic_cols
        self.future_cols = future_cols
        self.target_col = target_col
        self.all_cols = historic_cols + future_cols + [target_col]
        self.dtype = dtype

    def input_dfs_to_tensordataset(self,
                                   df_list: List[pd.DataFrame]) -> TensorDataset:
        x_h_list, x_f_list, y_list = [], [], []

        for df in df_list:
            df = df[list(dict.fromkeys(self.all_cols))]

            x_h = (torch.tensor(df[self.historic_cols].iloc[:-self.forecasting_horizon].values, dtype=self.dtype)
                   .unfold(0, self.historic_input_sequence_length, 1)
                   .permute(0, 2, 1))

            x_f = (torch.tensor(df[self.future_cols].iloc[self.historic_input_sequence_length:].values, dtype=self.dtype)
                   .unfold(0, self.forecasting_horizon, 1)
                   .permute(0, 2, 1))

            y = (torch.tensor(df[[self.target_col]].iloc[self.historic_input_sequence_length:].values, dtype=self.dtype)
                 .unfold(0, self.forecasting_horizon, 1)
                 .permute(0, 2, 1))

            valid = (
                ~x_h.isnan().any(dim=(1, 2)) &
                ~x_f.isnan().any(dim=(1, 2))
            )
            x_h_list.append(x_h[valid])
            x_f_list.append(x_f[valid])
            y_list.append(y[valid])

        return TensorDataset(
            torch.cat(x_h_list, dim=0),
            torch.cat(x_f_list, dim=0),
            torch.cat(y_list, dim=0),
        )
