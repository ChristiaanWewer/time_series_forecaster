import copy
import operator
import os

import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import DataLoader

from src.data.datasource import DataSource
from src.data.normalization import compute_norm_stats
from src.data.timeseries_dataset import TimeSeriesDataset, _collate_fn
from src.utils.scores_and_losses import DILATE, assert_differentiable, metric_to_spec, resolve_metric


def _criteria_field_names(criteria: list) -> list[str]:
    """Structured-array field names for the logging criteria: 'loss' and 'val_score' for
    the first two (fixed slots), then each remaining criterion's class name, deduplicated
    with a numeric suffix if the same metric type appears more than once.
    """
    names = ['loss', 'val_score']
    seen: dict[str, int] = {}
    for m in criteria[2:]:
        base = type(m).__name__
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f'{base}_{seen[base]}')
    return names


class Forecaster:
    def __init__(
            self,
            model,
            model_config,
            name,
            training_datasets: list[DataSource],
            validation_datasets: list[DataSource],
            historic_cols=None,
            future_cols=None,
            target_col='y',
            forecasting_horizon=7,
            historic_input_sequence_length=365,
            save_path='/',
            loss_function='mae',
            validation_score='mae',
            minimize_validation_score=True,
            save_aggregation_criterion='mean',
            validation_logging_criteria=None,
            number_of_epochs=100,
            learning_rate=0.001,
            batch_size=512,
            num_workers=0,
            prefetch_factor=2,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
            dtype=torch.float32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            optimizer='adam',
            seed=42,
            use_torch_compile=False,
            save_weights_every_n_epochs=1,
            norm_stats=None,
    ):
        if historic_cols is None:
            historic_cols = []
        if future_cols is None:
            future_cols = []
        if validation_logging_criteria is None:
            validation_logging_criteria = ['mae']
        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)
        self.name = name
        self.dtype = dtype
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.aggregation_criterion = save_aggregation_criterion
        self.minimize_validation_score = minimize_validation_score
        self.historic_cols = historic_cols
        self.future_cols = future_cols
        self.target_col = target_col
        self.forecasting_horizon = forecasting_horizon
        self.historic_input_sequence_length = historic_input_sequence_length
        self.save_path = save_path
        self._start_epoch = 0

        # resolve string metric names
        loss_function = resolve_metric(loss_function)
        self.validation_score = resolve_metric(validation_score)
        validation_logging_criteria = [resolve_metric(m) for m in validation_logging_criteria]
        assert_differentiable(loss_function)

        # reconstructable specs for checkpointing — not the live objects, since a compiled
        # DILATE holds a torch.compile closure that isn't reliably picklable across processes
        self.loss_function_spec = metric_to_spec(loss_function)
        self.validation_score_spec = metric_to_spec(self.validation_score)
        self.validation_logging_criteria_specs = [metric_to_spec(m) for m in validation_logging_criteria]

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
        self._model_forward = torch.compile(self.model, dynamic=True) if use_torch_compile else self.model

        # normalization stats computed on training data only; applied to both train and val.
        # Pass norm_stats explicitly (e.g. when resuming) to reuse the exact prior stats
        # instead of recomputing them, so the datasets built below stay consistent with
        # what the restored model/optimizer were actually trained against.
        if norm_stats is None:
            norm_stats = compute_norm_stats(
                sources=training_datasets,
                historic_cols=historic_cols,
                future_cols=future_cols,
                target_col=target_col,
            )
        self.norm_stats = norm_stats
        self.target_mean = norm_stats.get('target_mean')
        self.target_std = norm_stats.get('target_std')

        # datasets
        _ds_kwargs = dict(
            seq_len=historic_input_sequence_length,
            horizon=forecasting_horizon,
            historic_cols=historic_cols,
            future_cols=future_cols,
            norm_stats=norm_stats,
            target_col=target_col,
            dtype=dtype,
        )
        training_dataset = TimeSeriesDataset(sources=training_datasets, **_ds_kwargs)
        val_datasets = [TimeSeriesDataset(sources=[src], **_ds_kwargs) for src in validation_datasets]

        # DataLoader settings
        _loader_kwargs = dict(
            collate_fn=_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if num_workers > 0:
            _loader_kwargs['num_workers'] = num_workers
            _loader_kwargs['prefetch_factor'] = prefetch_factor

        self.training_loader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=shuffle, **_loader_kwargs
        )
        self.validation_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs)
            for ds in val_datasets
        ]

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
        self._criteria_names = _criteria_field_names(self.validation_logging_criteria)

        # logs — structured arrays: named fields ('loss', 'val_score', ...) instead of
        # positional indices
        log_dtype = [(field, np.float32) for field in self._criteria_names]
        self.loss_log = np.zeros((self.number_of_epochs,), dtype=np.float32)
        self.val_log = np.zeros((self.number_of_epochs, self.N_validation_sets), dtype=log_dtype)
        self.val_log_aggregated = {
            k: np.zeros((self.number_of_epochs,), dtype=log_dtype)
            for k in ('mean', 'max', 'min', 'median')
        }

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
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt['state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        if self._model_forward is not self.model:
            self._model_forward = torch.compile(self.model, dynamic=True)

    def _checkpoint_dict(self, epoch: int) -> dict:
        return {
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_logged_criterion': self.best_logged_criterion,
            'norm_stats': self.norm_stats,
            'model_config': self.model_config,
            'historic_cols': self.historic_cols,
            'future_cols': self.future_cols,
            'target_col': self.target_col,
            'forecasting_horizon': self.forecasting_horizon,
            'historic_input_sequence_length': self.historic_input_sequence_length,
            'loss_function_spec': self.loss_function_spec,
            'validation_score_spec': self.validation_score_spec,
            'validation_logging_criteria_specs': self.validation_logging_criteria_specs,
            'minimize_validation_score': self.minimize_validation_score,
            'save_aggregation_criterion': self.aggregation_criterion,
            'criteria_names': self._criteria_names,
            'loss_log': self.loss_log,
            'val_log': self.val_log,
            'val_log_aggregated': self.val_log_aggregated,
        }

    @classmethod
    def load_model(
            cls,
            checkpoint_path: str,
            model,
            resume: bool,
            training_datasets: list[DataSource] | None = None,
            validation_datasets: list[DataSource] | None = None,
            device='cpu',
            batch_size=512,
            num_workers=0,
            pin_memory=False,
            use_torch_compile=False,
            learning_rate=None,
            **forecaster_kwargs,
    ) -> 'Forecaster':
        """Load a Forecaster from a checkpoint saved by fit().

        Three modes:
        - training_datasets=None: inference only. Builds just the model and loads its
          weights; no optimizer, loaders, or logs. Works on a machine that never had the
          training data.
        - resume=True (requires training_datasets): continues the same run. Restores
          weights, optimizer state (momentum/Adam moments), norm_stats, the loss/score
          config, and the epoch counter and log history — all pulled from the checkpoint,
          not re-specified, so a resumed run cannot silently diverge from the original.
          `learning_rate` optionally overrides just the restored optimizer's LR (e.g. for
          decaying it on a fine-tuning continuation) without discarding its momentum state.
        - resume=False + training_datasets given: a fresh run warm-started from these
          weights as an initialization — fresh optimizer, fresh norm_stats (computed from
          whatever training_datasets are passed now), fresh logs starting at epoch 0.
          model_config/historic_cols/etc. must be supplied via forecaster_kwargs here,
          the same as a normal Forecaster(...) call.

        use_torch_compile is applied after weights are loaded (matching load_weights),
        not before — compiling is a runtime choice for wherever you're loading, not a
        fact saved in the checkpoint.
        """
        if resume and training_datasets is None:
            raise ValueError("resume=True requires training_datasets to continue training.")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if training_datasets is None:
            fc = cls.__new__(cls)
            fc.device = device
            fc.model_config = ckpt['model_config']
            fc.model = model(fc.model_config)
            fc.model.load_state_dict(ckpt['state_dict'])
            fc.model.to(device=device)
            fc.dtype = next(fc.model.parameters()).dtype
            fc._model_forward = torch.compile(fc.model, dynamic=True) if use_torch_compile else fc.model
            fc.norm_stats = ckpt['norm_stats']
            fc.target_mean = ckpt['norm_stats'].get('target_mean')
            fc.target_std = ckpt['norm_stats'].get('target_std')
            fc.historic_cols = ckpt['historic_cols']
            fc.future_cols = ckpt['future_cols']
            fc.target_col = ckpt['target_col']
            fc.forecasting_horizon = ckpt['forecasting_horizon']
            fc.historic_input_sequence_length = ckpt['historic_input_sequence_length']
            fc.batch_size = batch_size
            return fc

        if resume:
            fc = cls(
                model=model,
                model_config=ckpt['model_config'],
                training_datasets=training_datasets,
                validation_datasets=validation_datasets or [],
                historic_cols=ckpt['historic_cols'],
                future_cols=ckpt['future_cols'],
                target_col=ckpt['target_col'],
                forecasting_horizon=ckpt['forecasting_horizon'],
                historic_input_sequence_length=ckpt['historic_input_sequence_length'],
                loss_function=ckpt['loss_function_spec'],
                validation_score=ckpt['validation_score_spec'],
                validation_logging_criteria=ckpt['validation_logging_criteria_specs'],
                minimize_validation_score=ckpt['minimize_validation_score'],
                save_aggregation_criterion=ckpt['save_aggregation_criterion'],
                norm_stats=ckpt['norm_stats'],
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                use_torch_compile=False,
                **forecaster_kwargs,
            )
            fc.model.load_state_dict(ckpt['state_dict'])
            fc.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if learning_rate is not None:
                for group in fc.optimizer.param_groups:
                    group['lr'] = learning_rate
            fc.best_logged_criterion = ckpt['best_logged_criterion']

            start_epoch = ckpt['epoch'] + 1
            if fc.number_of_epochs <= start_epoch:
                raise ValueError(
                    f"number_of_epochs ({fc.number_of_epochs}) must be greater than the "
                    f"checkpoint's completed epoch ({start_epoch}) to resume."
                )
            fc._start_epoch = start_epoch
            fc.loss_log[:start_epoch] = ckpt['loss_log'][:start_epoch]
            fc.val_log[:start_epoch] = ckpt['val_log'][:start_epoch]
            for key in fc.val_log_aggregated:
                fc.val_log_aggregated[key][:start_epoch] = ckpt['val_log_aggregated'][key][:start_epoch]
        else:
            fc = cls(
                model=model,
                training_datasets=training_datasets,
                validation_datasets=validation_datasets or [],
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                use_torch_compile=False,
                **forecaster_kwargs,
            )
            fc.model.load_state_dict(ckpt['state_dict'])

        if use_torch_compile:
            fc._model_forward = torch.compile(fc.model, dynamic=True)
        return fc

    # ── training / evaluation ────────────────────────────────────────────────

    @torch.no_grad()
    def __evaluation_step(self, X, y, metrics):
        y_pred = self._model_forward(X)
        loss_val = metrics[0](y_pred, y)
        y_pred_d = y_pred * self.target_std + self.target_mean
        y_d      = y      * self.target_std + self.target_mean
        extra = [
            m(y_pred, y) if getattr(m, 'eval_on_normalized', False) else m(y_pred_d, y_d)
            for m in metrics[1:]
        ]
        return torch.stack([loss_val] + extra)

    def __training_step(self, X, y):
        y_pred = self._model_forward(X)
        loss = self.loss_function(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def __train_model(self, epoch):
        self.model.train()
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        n = 0
        for batch in self.training_loader:
            y = batch['y'].to(self.device)
            valid = ~y.isnan().any(dim=(1, 2))
            if not valid.any():
                continue
            X = {k: v.to(self.device)[valid] for k, v in batch.items() if k != 'y'}
            loss += self.__training_step(X, y[valid])
            n += 1
        self.loss_log[epoch] = (loss / n).cpu().item() if n else 0.0

    def __evaluate_model(self, epoch):
        self.model.eval()
        for i, loader in enumerate(self.validation_loaders):
            log = torch.zeros(self.N_criteria, device=self.device, dtype=self.dtype)
            n = 0
            for batch in loader:
                y = batch['y'].to(self.device)
                valid = ~y.isnan().any(dim=(1, 2))
                if not valid.any():
                    continue
                X = {k: v.to(self.device)[valid] for k, v in batch.items() if k != 'y'}
                log += self.__evaluation_step(X, y[valid], self.validation_logging_criteria)
                n += 1
            row = (log / n).detach().cpu().tolist() if n else [np.nan] * self.N_criteria
            self.val_log[epoch, i] = tuple(row)

        unstructured = rfn.structured_to_unstructured(self.val_log[epoch])
        for key, reduce_fn in (('mean', np.mean), ('max', np.max), ('min', np.min), ('median', np.median)):
            self.val_log_aggregated[key][epoch] = tuple(reduce_fn(unstructured, axis=0).tolist())

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self):
        for epoch in range(self._start_epoch, self.number_of_epochs):
            self.__train_model(epoch)
            self.__evaluate_model(epoch)

            logged = self.val_log_aggregated[self.aggregation_criterion][epoch]['val_score']
            if self.save_operator(logged, self.best_logged_criterion):
                self.best_logged_criterion = logged
                if self.save_path:
                    torch.save(self._checkpoint_dict(epoch), self.save_path_best)
                    print(f'Best model saved → {self.save_path_best}')

            if self.save_path and self.save_weights_every_n_epochs and (epoch + 1) % self.save_weights_every_n_epochs == 0:
                epoch_path = self.save_path_epoch.format(epoch=epoch + 1)
                torch.save(self._checkpoint_dict(epoch), epoch_path)
                print(f'Checkpoint saved  → {epoch_path}')

            val_loss = self.val_log_aggregated[self.aggregation_criterion][epoch]['loss']
            print(
                f'Epoch {epoch + 1}/{self.number_of_epochs}'
                f'  train_loss={self.loss_log[epoch]:.4f}'
                f'  val_loss={val_loss:.4f}'
                f'  val_score={logged:.4f}'
                f'  best={self.best_logged_criterion:.4f}'
            )

    def predict(self, x_test: DataSource | list[DataSource], batch_size=None, denormalize=True, device=None, num_workers=0, pin_memory=False):
        batch_size = batch_size or self.batch_size
        device = device or self.device
        if device != self.device:
            self.model.to(device)
            self.device = device

        if isinstance(x_test, DataSource):
            x_test = [x_test]
        dataset = TimeSeriesDataset(
            sources=x_test,
            seq_len=self.historic_input_sequence_length,
            horizon=self.forecasting_horizon,
            historic_cols=self.historic_cols,
            future_cols=self.future_cols,
            norm_stats=self.norm_stats,
            target_col=None,
            dtype=self.dtype,
        )

        _loader_kwargs = dict(collate_fn=_collate_fn, pin_memory=pin_memory)
        if num_workers > 0:
            _loader_kwargs['num_workers'] = num_workers
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **_loader_kwargs)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                X = {k: v.to(device) for k, v in batch.items() if k != 'y'}
                preds.append(self._model_forward(X))
        preds = torch.cat(preds, dim=0)
        if denormalize:
            preds = preds * self.target_std + self.target_mean
        return preds
