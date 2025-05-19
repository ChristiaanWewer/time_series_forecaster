import torch
import torch.nn as nn
import numpy as np
import time
import copy
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

import operator

import jax.numpy as jnp
from jax import device_put
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple
import optax
from optax import Adam

class Forecaster():
    def __init__(
            self, 
            model, 
            model_config,
            name,
            dfs_training_sets,
            dfs_validation_sets,
            historic_cols = [],
            future_cols = [],
            forecasting_horizon = 7,
            historic_input_sequence_length = 365,
            target_col = 'target',
            save_path = '/', 
            loss_function = nn.L1Loss(),
            save_criterion = nn.L1Loss(),
            save_aggregation_criterion = 'mean',
            higher_or_lower_is_better = 'lower',
            validation_logging_criteria = [nn.MSELoss()],
            number_of_epochs=100,
            learning_rate=0.001,
            batch_size=512,
            dtype=torch.float32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            max_nan_pct=0,
            scale_inputs_with_missing_values=False,
            seed=42,
            ):
        """
        Make the class that defines the forecaster.

        Input Arguments:
        Model: Class: of the Neural Network

        """
        
        # set variables to run
        self.device = device
        self.seed = seed
        self.torch_seed = torch.manual_seed(self.seed)
        self.name = name
        self.model_config = copy.deepcopy(model_config)

        # add elements to model_config
        self.model_config['historic_cols'] = historic_cols
        self.model_config['future_cols'] = future_cols
        self.model_config['forecasting_horizon'] = forecasting_horizon
        self.model_config['historic_input_sequence_length'] = historic_input_sequence_length
        

        self.model_uninitialized = model

        # save the NN class to be used for the model and compile it for faster training
        self.model = self.model_uninitialized(self.model_config)
        self.batch_size = batch_size
        self.dtype = dtype
        self.learning_rate = learning_rate

        # send the model already to our GPU if available
        self.model.to(device=self.device, dtype=self.dtype)

        # set save path
        self.save_path = save_path

        # set info about forecasts such as input variables, target variables and forecasting horizon
        self.historic_cols = historic_cols
        self.future_cols = future_cols
        self.forecasting_horizon = forecasting_horizon
        self.historic_input_sequence_length = historic_input_sequence_length
        self.target_col = target_col
        self.max_nan_pct = max_nan_pct
        self.scale_inputs_with_missing_values = scale_inputs_with_missing_values
        self.number_of_epochs = number_of_epochs
        self.aggregation_criterion = save_aggregation_criterion
        self.save_operator = operator.lt if higher_or_lower_is_better == 'lower' else operator.gt




        # make the training and validation sets
        # note we have multiple validation and training sets
        self.dfs_training_sets = dfs_training_sets
        self.dfs_validation_sets = dfs_validation_sets
        dfs_training_sets_concat = pd.concat(dfs_training_sets, axis=0)

        # compute the mean and std of the training set
        self.mean_training_set = dfs_training_sets_concat.mean()
        self.std_training_set = dfs_training_sets_concat.std()

        # 1) best criterion initialization
        self.best_logged_criterion = device_put(
            jnp.array(jnp.inf, dtype=self.dtype)
        ) if higher_or_lower_is_better == 'lower' else device_put(
            jnp.array(-jnp.inf, dtype=self.dtype)
        )

        # 2) build JAX datasets
        training_dataset = self.__dfs_to_jax(dfs_training_sets)
        validation_datasets = [self.__dfs_to_jax([df]) for df in self.dfs_validation_sets]

        # 3) create DataLoader equivalents
        self.training_loader = DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.validation_loaders = [
            DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            for ds in validation_datasets
        ]

        # 4) compute batch counts
        self.N_validation_sets = len(self.validation_loaders)
        self.N_batches_per_validation_set = [
            device_put(jnp.array(len(loader), dtype=self.dtype))
            for loader in self.validation_loaders
        ]
        self.N_batches_per_training_set = len(self.training_loader)

        # 5) optimizer setup (using Optax)
        self.optimizer = optax.adam(self.learning_rate)
        # assume you have a PyTree of params in `self.model_params`
        self.opt_state = self.optimizer.init(self.model_params)

        # 6) loss & logging criteria
        self.loss_function = loss_function
        self.validation_logging_criteria = (
            [self.loss_function, save_criterion] + validation_logging_criteria
        )
        self.N_criteria = len(self.validation_logging_criteria)

        # 7) initialize logs
        self.loss_criterion_training_set_log_tensor = jnp.zeros(
            (self.number_of_epochs,), dtype=self.dtype
        )
        self.results_criteria_on_validation_sets_log_tensor = jnp.zeros(
            (self.number_of_epochs, self.N_validation_sets, self.N_criteria),
            dtype=self.dtype
        )
        self.logging_criteria_aggregated = {
            'mean':   jnp.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
            'max':    jnp.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
            'min':    jnp.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
            'median': jnp.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
        }

          # make placeholder for the train time
        self.train_time = 0

        # all sorts of settings that speak for themselves
        self.trained = False

        # make directories if we want to save the model
        if self.save_path != False:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.save_path = os.path.join(self.save_path, name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            self.save_path_model = os.path.join(self.save_path, f'model_{name}.pt')

    def __dfs_to_torch(self, dfs):

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

    def __dfs_to_jax(
        self,
        dfs: List[pd.DataFrame]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Convert a list of pandas DataFrames into JAX arrays for historic inputs,
        future inputs, and targets.
        """
        hist_seqs = []
        fut_seqs  = []
        tgt_seqs  = []

        for df in dfs:
            # normalize
            arr = (df - self.mean_training_set) / self.std_training_set

            # numpy views
            hist_arr = arr[self.historic_cols]\
                .iloc[:-self.forecasting_horizon]\
                .values
            fut_arr  = arr[self.future_cols]\
                .iloc[self.historic_input_sequence_length:]\
                .values
            tgt_arr  = arr[[self.target_col]]\
                .iloc[self.historic_input_sequence_length:]\
                .values

            # rolling windows
            hist_win = sliding_window_view(
                hist_arr,
                window_shape=self.historic_input_sequence_length,
                axis=0
            )
            fut_win = sliding_window_view(
                fut_arr,
                window_shape=self.forecasting_horizon,
                axis=0
            )
            tgt_win = sliding_window_view(
                tgt_arr,
                window_shape=self.forecasting_horizon,
                axis=0
            )

            # to JAX and reorder to (batch, features, time)
            hist_jax = jnp.transpose(jnp.array(hist_win), (0, 2, 1))
            fut_jax  = jnp.transpose(jnp.array(fut_win),  (0, 2, 1))
            tgt_jax  = jnp.transpose(jnp.array(tgt_win),  (0, 2, 1))

            # NaN filtering
            hist_nans = jnp.max(jnp.mean(jnp.isnan(hist_jax), axis=1), axis=1)
            fut_nans  = jnp.max(jnp.mean(jnp.isnan(fut_jax),  axis=1), axis=1)
            mask = jnp.maximum(hist_nans, fut_nans) <= self.max_nan_pct

            hist_jax = hist_jax[mask]
            fut_jax  = fut_jax[mask]
            tgt_jax  = tgt_jax[mask]

            # optional missing-value scaling
            if self.scale_inputs_with_missing_values:
                hist_frac = jnp.mean(jnp.isnan(hist_jax), axis=2, keepdims=True)
                fut_frac  = jnp.mean(jnp.isnan(fut_jax),  axis=2, keepdims=True)
                hist_jax = hist_jax / hist_frac
                fut_jax  = fut_jax  / fut_frac

            # fill NaNs
            hist_jax = jnp.nan_to_num(hist_jax, nan=0.0)
            fut_jax  = jnp.nan_to_num(fut_jax,  nan=0.0)

            hist_seqs.append(hist_jax)
            fut_seqs.append(fut_jax)
            tgt_seqs.append(tgt_jax)

        # concatenate and move to device
        historic_input_sequences = device_put(jnp.concatenate(hist_seqs, axis=0))
        future_input_sequences   = device_put(jnp.concatenate(fut_seqs,  axis=0))
        target_sequences         = device_put(jnp.concatenate(tgt_seqs,  axis=0))

        return historic_input_sequences, future_input_sequences, target_sequences


    def compute_nummber_of_parameters(self):
        """
        simple function that computes the numbers of parameters

        """

        n_params = sum(p.numel() for p in self.model.parameters())
        return n_params
    
    # @torch.compile(backend="inductor")
    @torch.no_grad()
    def __evaluation_step(self, X, y, metrics):

        y_pred = self.model(X)
        
        # log desired metrics
        return torch.stack([criterion(y_pred, y) for criterion in metrics])

    # @torch.compile(backend="inductor")
    def __training_step(self, X, y):
        y_pred = self.model(X)

        # compute the loss
        loss = self.loss_function(y_pred, y)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def __evaluate_model(self, epoch):
        """
        Hidden function that evaluates the model on each of the validation sets and computes the averages of these

        Input Arguments:
        epoch: int: current epoch

        """

        self.model.eval()
        # obtain the results of each of the validation sets
        for i, validation_loader_i in enumerate(self.validation_loaders):

            for x_h, x_f, y in validation_loader_i:
                X = {'x_h': x_h, 'x_f': x_f}

                logging_tensor = self.__evaluation_step(X, y, self.validation_logging_criteria)

            logging_tensor = logging_tensor / self.N_batches_per_validation_set[i]
            self.results_criteria_on_validation_sets_log_tensor[epoch, i, :] = logging_tensor

        # compute the mean of the results
        self.logging_criteria_aggregated['mean'][epoch, :] = torch.mean(self.results_criteria_on_validation_sets_log_tensor[epoch, :, :], dim=0)
        self.logging_criteria_aggregated['max'][epoch, :] = torch.max(self.results_criteria_on_validation_sets_log_tensor[epoch, :, :], dim=0).values
        self.logging_criteria_aggregated['min'][epoch, :] = torch.min(self.results_criteria_on_validation_sets_log_tensor[epoch, :, :], dim=0).values   
        self.logging_criteria_aggregated['median'][epoch, :] = torch.median(self.results_criteria_on_validation_sets_log_tensor[epoch, :, :], dim=0).values

    def fit(self):
        """
        Function that fits the model to the training data
        This is the function we use for fitting the model to the training data
        """

        for epoch in range(self.number_of_epochs):

            self.__train_model(epoch)
            self.__evaluate_model(epoch)

            # make sure the model is saved when the validation metric of choice is maximized or minimized
            logged_criterion = self.logging_criteria_aggregated[self.aggregation_criterion][epoch][1]

            if self.save_operator(logged_criterion, self.best_logged_criterion):

                # best logged criterion is the one we want to save
                self.best_logged_criterion = logged_criterion

                # save the model
                self.model_best = copy.deepcopy(self.model)

                # save the model
                if self.save_path != False:
                    torch.save(self.model_best.state_dict(), self.save_path_model)
                    print(f'Model saved to {self.save_path_model}')        

            print(f'Epoch {epoch + 1}/{self.number_of_epochs} - Training Loss: {self.loss_criterion_training_set_log_tensor[epoch].item()} - Validation Criterion: {self.logging_criteria_aggregated[self.aggregation_criterion][epoch][1]} - Best Validation Criterion: {self.best_logged_criterion.item()}')
            


    def predict(self, x_test, batch_size=None, return_tensor=True):
        """
        Function that predicts the output of the model

        Input Arguments:
        x_test: torch.Tensor: input data, x

        batch_size: int: batch size of the data

        return_tensor: bool: if we want to return the output as a tensor or as a numpy array

        """
        if batch_size is None:
            batch_size = self.batch_size

    pass
            


    def __train_model(self, epoch):
        """
        Hidden function that trains the model

        """

        self.model.train()
        loss = torch.tensor(0).to(device=self.device, dtype=self.dtype)
        for x_h, x_f, y in self.training_loader:
            X = {'x_h': x_h, 'x_f': x_f}
            loss += self.__training_step(X, y)

        # make sure we have the average loss
        loss = loss / self.N_batches_per_training_set

        # log the loss
        self.loss_criterion_training_set_log_tensor[epoch] = loss


