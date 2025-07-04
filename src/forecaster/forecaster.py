import torch
import torch.nn as nn
import numpy as np
import time
import copy
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

import operator

# import jax.numpy as jnp
# from jax import device_put
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple
# import optax
# from optax import Adam
from src.data.preprocessing import DataPreprocessor

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
            optimizer = 'adam',
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
        self.best_logged_criterion = np.array(np.inf, dtype=self.dtype) \
            if higher_or_lower_is_better == 'lower' \
            else np.array(-np.inf, dtype=self.dtype)
    
        # 2) build TORCH datasets
        training_dataset = DataPreprocessor.__dfs_to_torch(dfs_training_sets)
        validation_datasets = [DataPreprocessor.__dfs_to_torch([df]) for df in self.dfs_validation_sets]
        
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
        self.N_batches_per_validation_set = [len(loader)
            for loader in self.validation_loaders
        ]
        self.N_batches_per_training_set = len(self.training_loader)

        # 5) optimizer setup, call optimizer string ('adam' e.g.) on the torch.optim module
        # getattr(torch.optim, optimizer) is used to get the optimizer class
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.opt_state = self.optimizer.init(self.model_params)

        # 6) loss & logging criteria
        self.loss_function = loss_function
        self.validation_logging_criteria = (
            [self.loss_function, save_criterion] + validation_logging_criteria
        )
        self.N_criteria = len(self.validation_logging_criteria)

        # 7) initialize logs
        self.loss_criterion_training_set_log_tensor = np.zeros(
            (self.number_of_epochs,), dtype=self.dtype
        )
        self.results_criteria_on_validation_sets_log_tensor = np.zeros(
            (self.number_of_epochs, self.N_validation_sets, self.N_criteria),
            dtype=self.dtype
        )
        self.logging_criteria_aggregated = {
            'mean':   np.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
            'max':    np.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
            'min':    np.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
            'median': np.zeros((self.number_of_epochs, self.N_criteria), dtype=self.dtype),
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
        self.logging_criteria_aggregated['mean'][epoch, :] = torch.mean(self.results_criteria_on_validation_sets_log_tensor[epoch, :, :], dim=0).values
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

        # make data loader with the test data
        test_dataset = 

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


