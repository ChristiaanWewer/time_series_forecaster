import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import os
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple


class TimeSeriesDataPreprocessor:
    def __init__(self, 
                 historic_input_sequence_length: int, 
                 forecasting_horizon: int, 
                 historic_cols: List[str], 
                 future_cols: List[str], 
                 static_cols: List[str],
                 target_col: str, 
                 mean_training_set: pd.Series, 
                 std_training_set: pd.Series, 
                 max_nan_pct: float = 0.0, 
                 device: str = 'cpu', 
                 dtype: torch.dtype = torch.float32
                 ) -> None: 
        
        self.historic_input_sequence_length = historic_input_sequence_length
        self.forecasting_horizon = forecasting_horizon
        self.historic_cols = historic_cols
        self.future_cols = future_cols
        self.static_cols = static_cols
        self.target_col = target_col
        self.all_cols = historic_cols + future_cols + [target_col]
        self.mean_training_set = mean_training_set
        self.std_training_set = std_training_set
        self.max_nan_pct = max_nan_pct
        self.device = device
        self.dtype = dtype

    # def __handle_nans(self, 
    #                   historic_input_sequences: torch.Tensor, 
    #                   future_input_sequences: torch.Tensor, 
    #                   target_sequences: torch.Tensor,
    #                   add_nan_values_as_features: bool = False
    #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #     # find nans
    #     historic_nans = historic_input_sequences.isnan().to(self.dtype).mean(dim=1).max(dim=1).values
    #     future_nans = future_input_sequences.isnan().to(self.dtype).mean(dim=1).max(dim=1).values

    #     # make array that takes the max of the two
    #     max_nans_per_input = torch.max(historic_nans, future_nans)

    #     # select nans to keep
    #     nans_to_keep = max_nans_per_input <= self.max_nan_pct
    #     historic_input_sequences = historic_input_sequences[nans_to_keep]
    #     future_input_sequences = future_input_sequences[nans_to_keep]
    #     target_sequences = target_sequences[nans_to_keep]

    #     # fill nans with 0s in the historic input sequences and in the future input sequences
    #     historic_input_sequences = torch.nan_to_num(historic_input_sequences, nan=0.0).to(self.dtype)
    #     future_input_sequences = torch.nan_to_num(future_input_sequences, nan=0.0).to(self.dtype)

    #     return (historic_input_sequences, future_input_sequences, target_sequences)

        # if add_nan_values_as_features:
            
            # add the features that 
    def static_df_to_tensordataset(self, 
                                   df_static: pd.DataFrame) -> TensorDataset:
        

        # select the static columns
        df_static = df_static[self.static_cols]

        # scale the data if its not a one-hot encoded categorical variable
        # figure out which of the cols only have a 0 and 1, we do not need to scale those
        cols_to_scale = [col for col in self.static_cols if not (1.0 in df_static[col].unique() and 0.0 in df_static[col].unique() and df_static[col].nunique() == 2)]

        df_static_mean = self.mean_training_set[cols_to_scale]
        df_static_std = self.std_training_set[cols_to_scale]

        df_static[cols_to_scale] = (df_static[cols_to_scale] - df_static_mean) / df_static_std

        # convert to tensor
        static_tensor = torch.tensor(df_static.values, device=self.device, dtype=self.dtype)

        # create a TensorDataset
        dataset = TensorDataset(static_tensor)

        return dataset

            
    def input_dfs_to_tensordataset(self, 
                              df_list_time_series: 
                              List[pd.DataFrame]) -> TensorDataset: 

        historic_input_sequences_list = []
        future_input_sequences_list = []
        target_sequences_list = []
        
        for df in df_list_time_series:

            df = df[self.all_cols]

            # scale the data
            df = (df - self.mean_training_set[self.all_cols]) / self.std_training_set[self.all_cols]

            # use torch to create a rolling window of the data, with a window size of 'historic_input_sequence_length' days
            historic_input_sequences = torch.tensor(df[self.historic_cols].iloc[:-self.forecasting_horizon].values).unfold(0, self.historic_input_sequence_length, 1).permute(0, 2, 1)
            future_input_sequences = torch.tensor(df[self.future_cols].iloc[self.historic_input_sequence_length:].values).unfold(0, self.forecasting_horizon, 1).permute(0, 2, 1)
            target_sequences = torch.tensor(df[[self.target_col]].iloc[self.historic_input_sequence_length:].values).unfold(0, self.forecasting_horizon, 1).permute(0, 2, 1)

            # add the sequences to the lists
            historic_input_sequences_list.append(historic_input_sequences)
            future_input_sequences_list.append(future_input_sequences)
            target_sequences_list.append(target_sequences)
        
        # concatenate the sequences
        historic_input_sequences = torch.cat(historic_input_sequences_list, dim=0).to(device=self.device, dtype=self.dtype)
        future_input_sequences = torch.cat(future_input_sequences_list, dim=0).to(device=self.device, dtype=self.dtype)
        target_sequences = torch.cat(target_sequences_list, dim=0).to(device=self.device, dtype=self.dtype)
        
        # turn into TensorDataset
        dataset = TensorDataset(historic_input_sequences, future_input_sequences, target_sequences)

        return dataset


# def dfs_to_jax(
#     self,
#     dfs: List[pd.DataFrame]
# ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     """
#     Convert a list of pandas DataFrames into JAX arrays for historic inputs,
#     future inputs, and targets.
#     """
#     hist_seqs = []
#     fut_seqs  = []
#     tgt_seqs  = []

#     for df in dfs:
#         # normalize
#         arr = (df - self.mean_training_set) / self.std_training_set

#         # numpy views
#         hist_arr = arr[self.historic_cols]\
#             .iloc[:-self.forecasting_horizon]\
#             .values
#         fut_arr  = arr[self.future_cols]\
#             .iloc[self.historic_input_sequence_length:]\
#             .values
#         tgt_arr  = arr[[self.target_col]]\
#             .iloc[self.historic_input_sequence_length:]\
#             .values

#         # rolling windows
#         hist_win = sliding_window_view(
#             hist_arr,
#             window_shape=self.historic_input_sequence_length,
#             axis=0
#         )
#         fut_win = sliding_window_view(
#             fut_arr,
#             window_shape=self.forecasting_horizon,
#             axis=0
#         )
#         tgt_win = sliding_window_view(
#             tgt_arr,
#             window_shape=self.forecasting_horizon,
#             axis=0
#         )

#         # to JAX and reorder to (batch, features, time)
#         hist_jax = jnp.transpose(jnp.array(hist_win), (0, 2, 1))
#         fut_jax  = jnp.transpose(jnp.array(fut_win),  (0, 2, 1))
#         tgt_jax  = jnp.transpose(jnp.array(tgt_win),  (0, 2, 1))

#         # NaN filtering
#         hist_nans = jnp.max(jnp.mean(jnp.isnan(hist_jax), axis=1), axis=1)
#         fut_nans  = jnp.max(jnp.mean(jnp.isnan(fut_jax),  axis=1), axis=1)
#         mask = jnp.maximum(hist_nans, fut_nans) <= self.max_nan_pct

#         hist_jax = hist_jax[mask]
#         fut_jax  = fut_jax[mask]
#         tgt_jax  = tgt_jax[mask]

#         # optional missing-value scaling
#         if self.scale_inputs_with_missing_values:
#             hist_frac = jnp.mean(jnp.isnan(hist_jax), axis=2, keepdims=True)
#             fut_frac  = jnp.mean(jnp.isnan(fut_jax),  axis=2, keepdims=True)
#             hist_jax = hist_jax / hist_frac
#             fut_jax  = fut_jax  / fut_frac

#         # fill NaNs
#         hist_jax = jnp.nan_to_num(hist_jax, nan=0.0)
#         fut_jax  = jnp.nan_to_num(fut_jax,  nan=0.0)

#         hist_seqs.append(hist_jax)
#         fut_seqs.append(fut_jax)
#         tgt_seqs.append(tgt_jax)

#     # concatenate and move to device
#     historic_input_sequences = device_put(jnp.concatenate(hist_seqs, axis=0))
#     future_input_sequences   = device_put(jnp.concatenate(fut_seqs,  axis=0))
#     target_sequences         = device_put(jnp.concatenate(tgt_seqs,  axis=0))

#     return historic_input_sequences, future_input_sequences, target_sequences