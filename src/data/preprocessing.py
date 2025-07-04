import jax
import numpy as np
import pandas as pd
import torch
from jax import device_put
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import os
from torch.utils.data import TensorDataset
import pandas as pd
import jax.numpy as jnp
from jax import device_put
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple


class DataPreprocessor:
    """
    A simple DataPreprocessor that turns pandas DataFrames into JAX arrays
    for historic inputs, future inputs, and targets.
    it handles all sorts of preprocessing
    """

    def dfs_to_torch(self, dfs):

        historic_input_sequences_list = []
        future_input_sequences_list = []
        target_sequences_list = []
        
        for df in dfs:

            # scale the data
            df = (df - self.mean_training_set) / self.std_training_set

            # use torch to create a rolling window of the data, with a window size of 'historic_input_sequence_length' days
            historic_input_sequences = torch.tensor(df[self.historic_cols].iloc[:-self.forecasting_horizon].values).unfold(0, self.historic_input_sequence_length, 1).permute(0, 2, 1)
            future_input_sequences = torch.tensor(df[self.future_cols].iloc[self.historic_input_sequence_length:].values).unfold(0, self.forecasting_horizon, 1).permute(0, 2, 1)
            target_sequences = torch.tensor(df[[self.target_col]].iloc[self.historic_input_sequence_length:].values).unfold(0, self.forecasting_horizon, 1).permute(0, 2, 1)

            # remove sequences with NaNs
            historic_nans = historic_input_sequences.isnan().to(self.dtype).mean(dim=1).max(dim=1).values
            future_nans = future_input_sequences.isnan().to(self.dtype).mean(dim=1).max(dim=1).values

            # make array that takes the max of the two
            max_nans_per_input = torch.max(historic_nans, future_nans)

            # select nans to keep
            nans_to_keep = max_nans_per_input <= self.max_nan_pct
            historic_input_sequences = historic_input_sequences[nans_to_keep]
            future_input_sequences = future_input_sequences[nans_to_keep]
            target_sequences = target_sequences[nans_to_keep]

            # if one row has a nan, make sure it is scaled with the missing value to make sure the next hidden state 
            # is in the same range
            # we scale it pct wise, so lets say if we have at row t 4 rows, e.g. the rows shape is [12000, 365, 4],
            # and we have 2 nans, then we scale the inputs with 4/2
            if self.scale_inputs_with_missing_values:
                
                # compute the number of nans in the data
                nr_of_nans_in_historic_data = historic_input_sequences.isnan().to(self.dtype).mean(dim=2).unsqueeze(-1)
                nr_of_nans_in_future_data = future_input_sequences.isnan().to(self.dtype).mean(dim=2).unsqueeze(-1)

                # scale the inputs with the number of nans in the data
                historic_input_sequences = historic_input_sequences / nr_of_nans_in_historic_data
                future_input_sequences = future_input_sequences / nr_of_nans_in_future_data

            # fill nans with 0s in the historic input sequences and in the future input sequences
            historic_input_sequences = torch.nan_to_num(historic_input_sequences, nan=0.0).to(self.dtype)
            future_input_sequences = torch.nan_to_num(future_input_sequences, nan=0.0).to(self.dtype)

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