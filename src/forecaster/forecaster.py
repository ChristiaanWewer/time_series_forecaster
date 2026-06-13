import copy
import operator
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import TimeSeriesDataPreprocessor
from src.utils.scores_and_losses import MAE, DILATE, assert_differentiable, resolve_metric


class Forecaster:
    def __init__(
            self,
            model,
            model_config,
            name,
            dfs_training_sets,
            dfs_validation_sets,
            historic_cols=[],
            future_cols=[],
            forecasting_horizon=7,
            historic_input_sequence_length=365,
            target_col='target',
            save_path='/',
            loss_function='mae',
            validation_score='mae',
            minimize_validation_score=True,
            save_aggregation_criterion='mean',
            validation_logging_criteria=[],
            number_of_epochs=100,
            learning_rate=0.001,
            batch_size=512,
            dtype=torch.float32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            optimizer='adam',
            seed=42,
            use_torch_compile=False,
            save_weights_every_n_epochs=1,
    ):
        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)
        self.name = name
        self.dtype = dtype
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.aggregation_criterion = save_aggregation_criterion
        self.historic_cols = historic_cols
        self.future_cols = future_cols
        self.forecasting_horizon = forecasting_horizon
        self.historic_input_sequence_length = historic_input_sequence_length
        self.target_col = target_col
        self.save_path = save_path

        # resolve string metric names
        loss_function = resolve_metric(loss_function)
        self.validation_score = resolve_metric(validation_score)
        validation_logging_criteria = [resolve_metric(m) for m in validation_logging_criteria]
        assert_differentiable(loss_function)

        # compile DILATE kernel once and share across all DILATE instances
        if use_torch_compile:
            from src.utils.DILATE.dilate_loss import dilate_loss as _fn
            _compiled_dilate = torch.compile(_fn, dynamic=True)
            for _m in [loss_function, self.validation_score] + validation_logging_criteria:
                if isinstance(_m, DILATE):
                    _m._dilate_fn = _compiled_dilate

        self.save_operator = operator.lt if minimize_validation_score else operator.gt
        self.best_logged_criterion = (
            np.array(np.inf, dtype=np.float32) if minimize_validation_score
            else np.array(-np.inf, dtype=np.float32)
        )

        # model
        self.model_config = copy.deepcopy(model_config)
        self.model_config['historic_cols'] = historic_cols
        self.model_config['future_cols'] = future_cols
        self.model_config['forecasting_horizon'] = forecasting_horizon
        self.model_config['historic_input_sequence_length'] = historic_input_sequence_length
        self.model = model(self.model_config)
        self.model.to(device=self.device, dtype=self.dtype)
        # self.model stays raw (deepcopy/save); _model_forward is used for all forward passes
        self._model_forward = torch.compile(self.model, dynamic=True) if use_torch_compile else self.model

        # datasets (raw, un-normalized)
        preprocessor = TimeSeriesDataPreprocessor(
            historic_input_sequence_length=historic_input_sequence_length,
            forecasting_horizon=forecasting_horizon,
            historic_cols=historic_cols,
            future_cols=future_cols,
            target_col=target_col,
            dtype=dtype,
        )
        training_dataset_raw = preprocessor.input_dfs_to_tensordataset(dfs_training_sets)
        validation_datasets_raw = [preprocessor.input_dfs_to_tensordataset([df]) for df in dfs_validation_sets]

        # per-column normalization over ALL datasets (train + val) so stats cover the full range
        _all_xh = torch.cat([training_dataset_raw.tensors[0]] + [ds.tensors[0] for ds in validation_datasets_raw])
        _all_xf = torch.cat([training_dataset_raw.tensors[1]] + [ds.tensors[1] for ds in validation_datasets_raw])
        _all_y  = torch.cat([training_dataset_raw.tensors[2]] + [ds.tensors[2] for ds in validation_datasets_raw])
        _xh_mean = _all_xh.mean(dim=(0, 1))
        _xh_std  = _all_xh.std(dim=(0, 1)).clamp(min=1e-8)
        _xf_mean = _all_xf.mean(dim=(0, 1))
        _xf_std  = _all_xf.std(dim=(0, 1)).clamp(min=1e-8)
        _y_valid = _all_y[~_all_y.isnan()]
        self.target_mean = float(_y_valid.mean())
        self.target_std  = float(_y_valid.std().clamp(min=1e-8))

        def _normalize(ds):
            xh, xf, y = ds.tensors
            return TensorDataset(
                (xh - _xh_mean) / _xh_std,
                (xf - _xf_mean) / _xf_std,
                (y - self.target_mean) / self.target_std,
            )

        training_dataset = _normalize(training_dataset_raw)
        validation_datasets = [_normalize(ds) for ds in validation_datasets_raw]

        self.training_loader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loaders = [DataLoader(ds, batch_size=self.batch_size, shuffle=False) for ds in validation_datasets]

        self.N_validation_sets = len(self.validation_loaders)
        self.N_batches_per_validation_set = [len(l) for l in self.validation_loaders]
        self.N_batches_per_training_set = len(self.training_loader)

        # optimizer
        self.optimizer = getattr(torch.optim, optimizer.capitalize())(
            self.model.parameters(), lr=self.learning_rate
        )

        # loss & logging criteria
        self.loss_function = loss_function
        self.validation_logging_criteria = [self.loss_function, self.validation_score] + validation_logging_criteria
        self.N_criteria = len(self.validation_logging_criteria)

        # logs
        self.loss_log = np.zeros((self.number_of_epochs,), dtype=np.float32)
        self.val_log = np.zeros((self.number_of_epochs, self.N_validation_sets, self.N_criteria), dtype=np.float32)
        self.val_log_aggregated = {
            k: np.zeros((self.number_of_epochs, self.N_criteria), dtype=np.float32)
            for k in ('mean', 'max', 'min', 'median')
        }

        self.train_time = 0
        self.trained = False
        self.save_weights_every_n_epochs = save_weights_every_n_epochs

        if self.save_path:
            os.makedirs(os.path.join(self.save_path, name), exist_ok=True)
            self.save_path = os.path.join(self.save_path, name)
            self.save_path_best = os.path.join(self.save_path, f'model_{name}_best.pt')
            self.save_path_epoch = os.path.join(self.save_path, f'model_{name}_epoch_{{epoch}}.pt')

    # ── helpers ──────────────────────────────────────────────────────────────

    def compute_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def load_weights(self, path=None):
        if path is None:
            path = self.save_path_best
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        # checkpoints from a compiled model have keys prefixed with '_orig_mod.' — strip them
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        # re-wrap compiled forward if this forecaster uses torch.compile
        if self._model_forward is not self.model:
            self._model_forward = torch.compile(self.model, dynamic=True)

    # ── training / evaluation ────────────────────────────────────────────────

    @torch.no_grad()
    def __evaluation_step(self, X, y, metrics):
        y_pred = self._model_forward(X)
        valid = ~y.isnan().any(dim=(1, 2))
        y_pred, y = y_pred[valid], y[valid]
        loss_val = metrics[0](y_pred, y)
        y_pred_d = y_pred * self.target_std + self.target_mean
        y_d      = y      * self.target_std + self.target_mean
        return torch.stack([loss_val] + [m(y_pred_d, y_d) for m in metrics[1:]])

    def __training_step(self, X, y):
        y_pred = self._model_forward(X)
        valid = ~y.isnan().any(dim=(1, 2))
        loss = self.loss_function(y_pred[valid], y[valid])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def __train_model(self, epoch):
        self.model.train()
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for x_h, x_f, y in self.training_loader:
            X = {'x_h': x_h, 'x_f': x_f}
            loss += self.__training_step(X, y)
        self.loss_log[epoch] = (loss / self.N_batches_per_training_set).cpu().item()

    def __evaluate_model(self, epoch):
        self.model.eval()
        for i, loader in enumerate(self.validation_loaders):
            log = torch.zeros(self.N_criteria, device=self.device, dtype=self.dtype)
            for x_h, x_f, y in loader:
                X = {'x_h': x_h, 'x_f': x_f}
                log += self.__evaluation_step(X, y, self.validation_logging_criteria)
            self.val_log[epoch, i, :] = (log / self.N_batches_per_validation_set[i]).detach().cpu().numpy()

        self.val_log_aggregated['mean'][epoch] = np.mean(self.val_log[epoch], axis=0)
        self.val_log_aggregated['max'][epoch] = np.max(self.val_log[epoch], axis=0)
        self.val_log_aggregated['min'][epoch] = np.min(self.val_log[epoch], axis=0)
        self.val_log_aggregated['median'][epoch] = np.median(self.val_log[epoch], axis=0)

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self):
        for epoch in range(self.number_of_epochs):
            self.__train_model(epoch)
            self.__evaluate_model(epoch)

            logged = self.val_log_aggregated[self.aggregation_criterion][epoch][1]
            if self.save_operator(logged, self.best_logged_criterion):
                self.best_logged_criterion = logged
                self.model_best = copy.deepcopy(self.model)
                if self.save_path:
                    torch.save(self.model_best.state_dict(), self.save_path_best)
                    print(f'Best model saved → {self.save_path_best}')

            if self.save_path and self.save_weights_every_n_epochs and (epoch + 1) % self.save_weights_every_n_epochs == 0:
                epoch_path = self.save_path_epoch.format(epoch=epoch + 1)
                torch.save(self.model.state_dict(), epoch_path)
                print(f'Checkpoint saved  → {epoch_path}')

            val_loss = self.val_log_aggregated[self.aggregation_criterion][epoch][0]
            print(
                f'Epoch {epoch + 1}/{self.number_of_epochs}'
                f'  train_loss={self.loss_log[epoch]:.4f}'
                f'  val_loss={val_loss:.4f}'
                f'  val_score={logged:.4f}'
                f'  best={self.best_logged_criterion:.4f}'
            )

    def predict(self, x_test, batch_size=None, denormalize=True):
        batch_size = batch_size or self.batch_size
        loader = DataLoader(x_test, batch_size=batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                x_h, x_f = batch[0], batch[1]
                X = {'x_h': x_h, 'x_f': x_f}
                preds.append(self._model_forward(X))
        preds = torch.cat(preds, dim=0)
        if denormalize:
            preds = preds * self.target_std + self.target_mean
        return preds
