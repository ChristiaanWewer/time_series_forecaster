import operator
import os

import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import DataLoader

from src.data.normalization import norm_stats_from
from src.data.timeseries_data import TimeSeriesData
from src.data.timeseries_dataset import TimeSeriesDataset
from src.forecaster.prefetcher import _make_prefetcher
from src.optim import Adam
from src.utils.scores_and_losses import MAE, assert_differentiable


def _identity_collate(batch: dict) -> dict:
    """collate_fn for TimeSeriesDataset: __getitems__ already returns an assembled batch.
    A module-level function rather than a lambda, so it pickles into spawned workers."""
    return batch


def _make_loader(dataset, batch_size, shuffle, device, num_workers=0, pin_memory=True,
                 drop_last=False, prefetch_factor=2, persistent_workers=True) -> DataLoader:
    """One DataLoader configured the same way at every site (training, validation,
    predict): identity collate, pinned memory when the device is CUDA, and persistent
    workers when num_workers > 0."""
    kwargs = dict(
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=_identity_collate,
        pin_memory=pin_memory and torch.device(device).type == 'cuda',
    )
    if num_workers > 0:
        kwargs.update(num_workers=num_workers, prefetch_factor=prefetch_factor,
                      persistent_workers=persistent_workers)
    return DataLoader(dataset, **kwargs)


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
            name,
            training_data: TimeSeriesData,
            validation_data: TimeSeriesData | None,
            loss,
            embeddings: list | None = None,
            validation_score=None,
            validation_logging_criteria: list | None = None,
            optimizer=None,
            embedding_regularization=None,
            model_regularizer=None,
            historic_cols=None,
            future_cols=None,
            target_col='y',
            forecasting_horizon=7,
            historic_input_sequence_length=365,
            save_path='/',
            minimize_validation_score=True,
            save_aggregation_criterion='mean',
            number_of_epochs=100,
            batch_size=512,
            num_workers=0,
            validation_num_workers=0,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            dtype=torch.float32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            seed=42,
            use_torch_compile=False,
            save_weights_every_n_epochs=1,
            norm_stats=None,
    ):
        if historic_cols is None:
            historic_cols = []
        if future_cols is None:
            future_cols = []
        if embeddings is None:
            embeddings = []
        if validation_score is None:
            validation_score = MAE()
        if validation_logging_criteria is None:
            validation_logging_criteria = [MAE()]
        if optimizer is None:
            optimizer = Adam(lr=0.001)

        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)
        self.name = name
        self.dtype = dtype
        self.batch_size = batch_size
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

        assert_differentiable(loss)

        # probabilistic output head: the loss alone determines how many output
        # channels the model needs (n_outputs=1 / output_kind='point' for ordinary
        # losses) — see src/models/heads.py and the model.bind() call below, which
        # sizes the head from these.
        self.output_kind = getattr(loss, 'output_kind', 'point')
        self.output_levels = getattr(loss, 'output_levels', None)
        self.n_outputs = getattr(loss, 'n_outputs', 1)

        self.save_operator = operator.lt if minimize_validation_score else operator.gt
        self.best_logged_criterion = (
            np.array(np.inf, dtype=np.float32) if minimize_validation_score
            else np.array(-np.inf, dtype=np.float32)
        )

        # normalization stats computed on training data only; applied to both train and
        # val. Pass norm_stats explicitly (e.g. when resuming) to reuse the exact prior
        # stats instead of recomputing them.
        if norm_stats is None:
            norm_stats = norm_stats_from(
                training_data,
                historic_cols=historic_cols,
                future_cols=future_cols,
                target_col=target_col,
            )
        self.norm_stats = norm_stats
        self.target_mean = norm_stats.get('target_mean')
        self.target_std = norm_stats.get('target_std')

        # group-reweighted losses (reweight=True) need each training source's own raw
        # target std, looked up by its stable identity (DataSource.group_key) rather
        # than list position, so a resumed run with reordered training_data sources still
        # lines samples up with the right series — a missing key here means a series
        # present now wasn't present when norm_stats was computed, deliberately
        # uncaught.
        if getattr(loss, 'needs_group_idx', False):
            group_target_std = norm_stats['group_target_std']
            group_std = torch.tensor(
                [group_target_std[ds.group_key] for ds in training_data.sources], dtype=dtype
            )
            loss.bind(group_std)
            loss.to(self.device)

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
        training_dataset = TimeSeriesDataset(training_data, **_ds_kwargs)
        # one dataset per validation source over the shared TimeSeriesData, so metrics
        # stay per series (one val_log row each)
        val_datasets = [
            TimeSeriesDataset(validation_data, source_positions=[i], **_ds_kwargs)
            for i in range(len(validation_data))
        ] if validation_data is not None else []

        # validation workers are a separate knob without persistent workers: with N
        # validation loaders, persistent pools would keep N × workers processes alive
        _loader_kwargs = dict(device=device, pin_memory=pin_memory, drop_last=drop_last,
                              prefetch_factor=prefetch_factor)
        self.training_loader = _make_loader(
            training_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, **_loader_kwargs
        )
        self.validation_loaders = [
            _make_loader(ds, batch_size, shuffle=False, num_workers=validation_num_workers,
                         persistent_workers=False, **_loader_kwargs)
            for ds in val_datasets
        ]

        self.N_validation_sets = len(self.validation_loaders)
        self.N_batches_per_validation_set = [len(l) for l in self.validation_loaders]
        self.N_batches_per_training_set = len(self.training_loader)

        # embeddings + model: bind embeddings to the actual column layout, combine
        # them (+ passthrough columns) into the historic/future feature pipeline, then
        # bind the model to the resulting input sizes and the loss-derived output shape.
        self._bind_embeddings_and_model(
            embeddings, model, historic_cols, future_cols,
            forecasting_horizon, self.n_outputs, self.output_kind, self.output_levels,
        )
        self.model.to(device=self.device, dtype=self.dtype)
        for net in self.embeddings:
            net.to(device=self.device, dtype=self.dtype)
        self._model_forward = torch.compile(self.model, dynamic=True) if use_torch_compile else self.model

        # optimizer — params split into five groups: embedding weights/biases, backbone
        # weights/biases, and the output head (dim>1 vs dim<=1, same as before; the
        # output head is identified by the 'output_head.' name prefix and is always
        # exempt from weight_decay and from regularization, same reasoning as biases).
        embedding_weights, embedding_biases = [], []
        for net in self.embeddings:
            for p in net.parameters():
                (embedding_weights if p.dim() > 1 else embedding_biases).append(p)

        backbone_weights, backbone_biases, head_params = [], [], []
        for param_name, p in self.model.named_parameters():
            if param_name.startswith('output_head.'):
                head_params.append(p)
            elif p.dim() > 1:
                backbone_weights.append(p)
            else:
                backbone_biases.append(p)

        self._embedding_weights = embedding_weights
        self._backbone_weights = backbone_weights
        param_groups = [
            {'params': embedding_weights},
            {'params': embedding_biases, 'weight_decay': 0.0},
            {'params': backbone_weights},
            {'params': backbone_biases, 'weight_decay': 0.0},
            {'params': head_params, 'weight_decay': 0.0},
        ]
        self.optimizer = optimizer.bind(param_groups)
        self.learning_rate = optimizer.lr

        self.embedding_regularizer = embedding_regularization
        if self.embedding_regularizer is not None:
            self.embedding_regularizer.bind(embedding_weights, self.learning_rate)
        self.model_regularizer = model_regularizer
        if self.model_regularizer is not None:
            self.model_regularizer.bind(backbone_weights, self.learning_rate)

        # loss & logging criteria
        self.loss = loss
        self.validation_score = validation_score
        self.validation_logging_criteria = [self.loss, self.validation_score] + validation_logging_criteria
        self.N_criteria = len(self.validation_logging_criteria)
        self._criteria_names = _criteria_field_names(self.validation_logging_criteria)

        # bind every criterion to the loss's own output_kind/output_levels — a no-op for
        # metrics that never reference self._level_index, and what lets quantile-specific
        # scores (WinklerScore/CRPS/PICP/PINAW) and the point-metric median auto-select
        # (MAE/NSE/.../_select_point) find the right channel(s).
        for m in self.validation_logging_criteria:
            m.bind_levels(self.output_kind, self.output_levels)

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

    # ── embeddings ───────────────────────────────────────────────────────────

    def _bind_embeddings_and_model(self, embeddings, model, historic_cols, future_cols,
                                    forecasting_horizon, n_outputs, output_kind, output_levels):
        names = [net.name for net in embeddings]
        if len(set(names)) != len(names):
            raise ValueError(f"embedding names must be unique, got: {names}")
        for net in embeddings:
            net.bind(historic_cols, future_cols)
        self.embeddings = embeddings

        self._embed_h_nets, self._embed_h_passthrough, h_size = self._prepare_side(
            embeddings, historic_cols, '_historic_idx')
        self._embed_f_nets, self._embed_f_passthrough, f_size = self._prepare_side(
            embeddings, future_cols, '_future_idx')

        model.bind(
            historic_input_size=h_size, future_input_size=f_size,
            forecasting_horizon=forecasting_horizon,
            n_outputs=n_outputs, output_kind=output_kind, output_levels=output_levels,
        )
        self.model = model

    @staticmethod
    def _prepare_side(embeddings, cols, idx_attr):
        applicable = [(net, getattr(net, idx_attr)) for net in embeddings if getattr(net, idx_attr) is not None]
        claimed = set()
        for _, idx in applicable:
            claimed.update(idx)
        passthrough = [i for i in range(len(cols)) if i not in claimed]
        output_size = sum(net.embedding_dim for net, _ in applicable) + len(passthrough)
        return applicable, passthrough, output_size

    @staticmethod
    def _embed_side(x, nets, passthrough_idx):
        parts = [net(x[..., idx]) for net, idx in nets]
        if passthrough_idx:
            parts.append(x[..., passthrough_idx])
        return torch.cat(parts, dim=-1) if parts else x

    def _embed(self, x_h, x_f):
        return (
            self._embed_side(x_h, self._embed_h_nets, self._embed_h_passthrough),
            self._embed_side(x_f, self._embed_f_nets, self._embed_f_passthrough),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def compute_number_of_parameters(self):
        total = sum(p.numel() for net in self.embeddings for p in net.parameters())
        return total + sum(p.numel() for p in self.model.parameters())

    def print_parameter_breakdown(self):
        embed_rows = [(net.name, sum(p.numel() for p in net.parameters())) for net in self.embeddings]
        model_rows = self.model.parameter_breakdown()
        embedding_total = sum(count for _, count in embed_rows)
        model_total = model_rows[-1][1]
        rows = embed_rows + model_rows[:-1] + [('total', embedding_total + model_total)]

        counts = [f'{count:,}' for _, count in rows]
        label_width = max(len(label) for label, _ in rows)
        count_width = max(len(c) for c in counts)
        rule = '-' * (label_width + count_width + 2)

        print(rule)
        for (label, _), count in zip(rows[:-1], counts[:-1]):
            print(f'{label:<{label_width}}  {count:>{count_width}}')
        print(rule)
        label, count = rows[-1][0], counts[-1]
        print(f'{label:<{label_width}}  {count:>{count_width}}')

    def _split_model_state_dict(self):
        full = self.model.state_dict()
        head = {k[len('output_head.'):]: v for k, v in full.items() if k.startswith('output_head.')}
        backbone = {k: v for k, v in full.items() if not k.startswith('output_head.')}
        return backbone, head

    def _load_model_and_embeddings_state(self, ckpt):
        backbone, head = ckpt['model']['backbone'], ckpt['model']['head']
        full = {**backbone, **{f'output_head.{k}': v for k, v in head.items()}}
        self.model.load_state_dict(full)
        embeddings_by_name = {net.name: net for net in self.embeddings}
        for name, state in ckpt['embeddings'].items():
            embeddings_by_name[name].load_state_dict(state)

    def load_weights(self, path=None):
        if path is None:
            path = self.save_path_best
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._load_model_and_embeddings_state(ckpt)
        if self._model_forward is not self.model:
            self._model_forward = torch.compile(self.model, dynamic=True)

    def _checkpoint_dict(self, epoch: int) -> dict:
        backbone, head = self._split_model_state_dict()
        return {
            'model': {'backbone': backbone, 'head': head},
            'embeddings': {net.name: net.state_dict() for net in self.embeddings},
            'norm_stats': self.norm_stats,
            'historic_cols': self.historic_cols,
            'future_cols': self.future_cols,
            'target_col': self.target_col,
            'forecasting_horizon': self.forecasting_horizon,
            'historic_input_sequence_length': self.historic_input_sequence_length,
            'n_outputs': self.n_outputs,
            'output_kind': self.output_kind,
            'output_levels': self.output_levels,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'embedding_regularizer_state': self.embedding_regularizer.get_state() if self.embedding_regularizer else None,
            'model_regularizer_state': self.model_regularizer.get_state() if self.model_regularizer else None,
            'best_logged_criterion': self.best_logged_criterion,
            'epoch': epoch,
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
            embeddings: list | None = None,
            resume: bool = False,
            loss=None,
            validation_score=None,
            validation_logging_criteria: list | None = None,
            optimizer=None,
            embedding_regularization=None,
            model_regularizer=None,
            training_data: TimeSeriesData | None = None,
            validation_data: TimeSeriesData | None = None,
            device='cpu',
            batch_size=512,
            num_workers=0,
            pin_memory=True,
            use_torch_compile=False,
            learning_rate=None,
            **forecaster_kwargs,
    ) -> 'Forecaster':
        """Load a Forecaster from a checkpoint saved by fit(). Checkpoints hold only
        learned state (weights, optimizer momentum, regularizer state, logs) — never
        reconstructable hyperparameters — so `model`/`embeddings` (and, for resume,
        every other component: loss/validation_score/optimizer/regularizers) must be
        the same unbound objects you'd hand to a fresh Forecaster() call.

        Three modes:
        - training_data=None: inference only. Binds model/embeddings using the
          schema and output shape saved in the checkpoint, then loads weights. No
          optimizer, loaders, or regularizers.
        - resume=True (requires training_data and every other component):
          continues the same run — a full Forecaster() construction, then weights,
          optimizer state, regularizer state, and log history are loaded on top from
          the checkpoint. `learning_rate`, if given, overwrites just the restored
          optimizer's LR (keeps momentum/Adam moment state).
        - resume=False + training_data given: a fresh run warm-started from these
          weights as an initialization — fresh optimizer, fresh norm_stats, fresh
          logs at epoch 0; only weights are loaded.
        """
        if resume and training_data is None:
            raise ValueError("resume=True requires training_data to continue training.")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        embeddings = embeddings or []

        if training_data is None:
            fc = cls.__new__(cls)
            fc.device = device
            fc.historic_cols = ckpt['historic_cols']
            fc.future_cols = ckpt['future_cols']
            fc.target_col = ckpt['target_col']
            fc.forecasting_horizon = ckpt['forecasting_horizon']
            fc.historic_input_sequence_length = ckpt['historic_input_sequence_length']
            fc.norm_stats = ckpt['norm_stats']
            fc.target_mean = ckpt['norm_stats'].get('target_mean')
            fc.target_std = ckpt['norm_stats'].get('target_std')
            fc.output_kind = ckpt['output_kind']
            fc.output_levels = ckpt['output_levels']
            fc.n_outputs = ckpt['n_outputs']

            fc._bind_embeddings_and_model(
                embeddings, model, fc.historic_cols, fc.future_cols,
                fc.forecasting_horizon, fc.n_outputs, fc.output_kind, fc.output_levels,
            )
            fc._load_model_and_embeddings_state(ckpt)
            fc.model.to(device=device)
            for net in fc.embeddings:
                net.to(device=device)
            fc.dtype = next(fc.model.parameters()).dtype
            fc._model_forward = torch.compile(fc.model, dynamic=True) if use_torch_compile else fc.model
            fc.batch_size = batch_size
            fc.embedding_regularizer = None
            fc.model_regularizer = None
            # training history, so a loaded-for-inference Forecaster still supports the
            # same post-hoc plotting (loss/val_score curves) a freshly-trained one does
            fc.loss_log = ckpt['loss_log']
            fc.val_log = ckpt['val_log']
            fc.val_log_aggregated = ckpt['val_log_aggregated']
            fc._criteria_names = ckpt['criteria_names']
            return fc

        if resume:
            fc = cls(
                model=model, embeddings=embeddings, loss=loss, validation_score=validation_score,
                validation_logging_criteria=validation_logging_criteria,
                optimizer=optimizer, embedding_regularization=embedding_regularization,
                model_regularizer=model_regularizer,
                training_data=training_data, validation_data=validation_data,
                historic_cols=ckpt['historic_cols'], future_cols=ckpt['future_cols'],
                target_col=ckpt['target_col'], forecasting_horizon=ckpt['forecasting_horizon'],
                historic_input_sequence_length=ckpt['historic_input_sequence_length'],
                norm_stats=ckpt['norm_stats'],
                device=device, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                use_torch_compile=False,
                **forecaster_kwargs,
            )
            fc._load_model_and_embeddings_state(ckpt)
            fc.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if fc.embedding_regularizer is not None:
                fc.embedding_regularizer.load_state(ckpt['embedding_regularizer_state'])
            if fc.model_regularizer is not None:
                fc.model_regularizer.load_state(ckpt['model_regularizer_state'])
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
                model=model, embeddings=embeddings, loss=loss, validation_score=validation_score,
                validation_logging_criteria=validation_logging_criteria,
                optimizer=optimizer, embedding_regularization=embedding_regularization,
                model_regularizer=model_regularizer,
                training_data=training_data, validation_data=validation_data,
                device=device, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                use_torch_compile=False,
                **forecaster_kwargs,
            )
            fc._load_model_and_embeddings_state(ckpt)

        if use_torch_compile:
            fc._model_forward = torch.compile(fc.model, dynamic=True)
        return fc

    # ── training / evaluation ────────────────────────────────────────────────

    def __training_step(self, x_h, x_f, y, group_idx=None):
        y_pred = self._model_forward(x_h, x_f)
        if getattr(self.loss, 'needs_group_idx', False):
            loss = self.loss(y_pred, y, group_idx)
        else:
            loss = self.loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        if self.embedding_regularizer is not None:
            self.embedding_regularizer.step(self._embedding_weights)
        if self.model_regularizer is not None:
            self.model_regularizer.step(self._backbone_weights)
        self.optimizer.step()
        return loss.detach()

    def __train_model(self, epoch):
        self.model.train()
        for net in self.embeddings:
            net.train()
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        n = 0
        for batch in _make_prefetcher(self.training_loader, self.device):
            y = batch['y']
            valid = ~y.isnan().any(dim=(1, 2))
            if not valid.any():
                continue
            x_h = batch['x_h'][valid]
            x_f = batch['x_f'][valid]
            x_h, x_f = self._embed(x_h, x_f)
            group_idx = batch['group_idx'][valid] if 'group_idx' in batch else None
            loss += self.__training_step(x_h, x_f, y[valid], group_idx)
            n += 1
        self.loss_log[epoch] = (loss / n).cpu().item() if n else 0.0

    @torch.no_grad()
    def __evaluate_model(self, epoch):
        self.model.eval()
        for net in self.embeddings:
            net.eval()
        metrics = self.validation_logging_criteria
        poolable = [getattr(m, 'poolable', False) for m in metrics]
        for i, loader in enumerate(self.validation_loaders):
            log = torch.zeros(self.N_criteria, device=self.device, dtype=self.dtype)
            accs = [m.new_accumulators() if p else None for m, p in zip(metrics, poolable)]
            n = 0
            for batch in _make_prefetcher(loader, self.device):
                y = batch['y']
                valid = ~y.isnan().any(dim=(1, 2))
                if not valid.any():
                    continue
                x_h = batch['x_h'][valid]
                x_f = batch['x_f'][valid]
                x_h, x_f = self._embed(x_h, x_f)
                y_valid = y[valid]
                y_pred = self._model_forward(x_h, x_f)
                y_pred_d = y_pred * self.target_std + self.target_mean
                y_d      = y_valid * self.target_std + self.target_mean
                for idx, m in enumerate(metrics):
                    # index 0 (the loss) always sees normalized values, matching
                    # __training_step; the rest see denormalized unless eval_on_normalized
                    use_normalized = idx == 0 or getattr(m, 'eval_on_normalized', False)
                    yp, yt = (y_pred, y_valid) if use_normalized else (y_pred_d, y_d)
                    if poolable[idx]:
                        m.accumulate(accs[idx], yp, yt)
                    else:
                        log[idx] += m(yp, yt)
                n += 1
            row = (
                [m.pooled_value(accs[idx]) if poolable[idx] else (log[idx] / n).item()
                 for idx, m in enumerate(metrics)]
                if n else [np.nan] * self.N_criteria
            )
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

    def predict(self, x_test: TimeSeriesData, batch_size=None, denormalize=True, device=None, num_workers=0, pin_memory=True, epoch=None):
        batch_size = batch_size or self.batch_size
        device = device or self.device
        if device != self.device:
            self.model.to(device)
            for net in self.embeddings:
                net.to(device)
            self.device = device

        if epoch is not None:
            # loads that epoch's saved checkpoint into self.model/self.embeddings in
            # place (same mechanism as load_weights, which this delegates to) — a
            # permanent weight swap, not a scoped one. self.save_path_epoch (and the
            # checkpoint file itself) may not exist — deliberately uncaught, let it crash.
            self.load_weights(path=self.save_path_epoch.format(epoch=epoch))

        dataset = TimeSeriesDataset(
            x_test,
            seq_len=self.historic_input_sequence_length,
            horizon=self.forecasting_horizon,
            historic_cols=self.historic_cols,
            future_cols=self.future_cols,
            norm_stats=self.norm_stats,
            target_col=None,
            dtype=self.dtype,
        )

        loader = _make_loader(dataset, batch_size, shuffle=False, device=device, num_workers=num_workers,
                              pin_memory=pin_memory, persistent_workers=False)

        self.model.eval()
        for net in self.embeddings:
            net.eval()
        preds = []
        with torch.no_grad():
            for batch in _make_prefetcher(loader, device):
                x_h, x_f = self._embed(batch['x_h'], batch['x_f'])
                preds.append(self._model_forward(x_h, x_f))
        preds = torch.cat(preds, dim=0)
        if denormalize:
            preds = preds * self.target_std + self.target_mean
        return preds.detach().cpu().numpy()
