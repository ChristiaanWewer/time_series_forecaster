import numpy as np
import pandas as pd
import pytest
import torch

from src.data import DataSource, TimeSeriesData
from src.forecaster import Forecaster
from src.models import LSTMEncoderDecoder, LSTMHistoric, EmbeddingNetwork
from src.optim import AdamW
from src.utils.regularization import L1Regularizer, RLN
from src.utils.scores_and_losses import CRPS, MAE, QuantileLoss


@pytest.fixture
def csv_path(tmp_path):
    n = 400
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'temp': rng.normal(10, 3, n),
        'temp_fcst': rng.normal(10, 3, n),
        'precip': rng.exponential(2, n),
        'landuse_code': rng.normal(0, 1, n),
        'y': rng.normal(5, 2, n),
    }, index=dates)
    path = tmp_path / 'data.csv'
    df.to_csv(path)
    return str(path)


def test_fit_predict_checkpoint_roundtrip(tmp_path, csv_path):
    train_data = TimeSeriesData([DataSource(csv=csv_path, start='2020-01-01', end='2020-10-31')])
    val_data = TimeSeriesData([DataSource(csv=csv_path, start='2020-11-01', end='2020-12-31')])

    landuse = EmbeddingNetwork(name='landuse', variables=['landuse_code'], embedding_dim=4, historic=True, future=False)
    model = LSTMHistoric(hidden_size=8, num_layers=1, dropout_rate=0.0)

    fc = Forecaster(
        model=model, name='e2e', embeddings=[landuse], loss=MAE(),
        optimizer=AdamW(lr=0.01), model_regularizer=L1Regularizer(lambda_=0.001),
        training_data=train_data, validation_data=val_data,
        historic_cols=['temp', 'precip', 'landuse_code'], future_cols=[],
        target_col='y', forecasting_horizon=3, historic_input_sequence_length=10,
        number_of_epochs=2, batch_size=16, save_path=str(tmp_path), device=torch.device('cpu'),
    )
    fc.fit()

    assert np.all(np.isfinite(fc.loss_log))
    assert np.all(np.isfinite(fc.val_log['loss']))

    preds = fc.predict(val_data)
    assert preds.ndim == 3 and preds.shape[-1] == 1
    assert np.all(np.isfinite(preds))

    # resume continues training and restores regularizer/optimizer state — load from
    # the last-epoch checkpoint (not save_path_best, which may lag behind on noisy
    # synthetic data) so resuming from epoch 2 is deterministic.
    ckpt_path = fc.save_path_epoch.format(epoch=2)
    fc2 = Forecaster.load_model(
        ckpt_path,
        model=LSTMHistoric(hidden_size=8, num_layers=1, dropout_rate=0.0),
        embeddings=[EmbeddingNetwork(name='landuse', variables=['landuse_code'], embedding_dim=4, historic=True, future=False)],
        resume=True, loss=MAE(), optimizer=AdamW(lr=0.01),
        model_regularizer=L1Regularizer(lambda_=0.001),
        training_data=train_data, validation_data=val_data,
        number_of_epochs=3, save_path=str(tmp_path), name='e2e', device='cpu',
    )
    assert fc2._start_epoch == 2
    fc2.fit()
    assert np.all(np.isfinite(fc2.loss_log))

    # inference-only load needs no loss/optimizer/training data
    fc3 = Forecaster.load_model(
        ckpt_path,
        model=LSTMHistoric(hidden_size=8, num_layers=1, dropout_rate=0.0),
        embeddings=[EmbeddingNetwork(name='landuse', variables=['landuse_code'], embedding_dim=4, historic=True, future=False)],
        device='cpu',
    )
    preds3 = fc3.predict(val_data)
    assert preds3.shape == preds.shape


def test_quantile_head_and_shared_embedding_and_rln(tmp_path, csv_path):
    train_data = TimeSeriesData([DataSource(csv=csv_path, start='2020-01-01', end='2020-12-31')])

    elev = EmbeddingNetwork(name='elev_like', variables=['temp'], embedding_dim=2, historic=True, future=True)
    model = LSTMEncoderDecoder(hidden_size=8, num_layers=1, dropout_rate=0.0, encoder_downscale_layer_size=4)
    levels = [0.1, 0.5, 0.9]

    fc = Forecaster(
        model=model, name='qtest', embeddings=[elev], loss=QuantileLoss(q=levels),
        validation_score=CRPS(), optimizer=AdamW(lr=0.01), model_regularizer=RLN(theta=-6.0),
        training_data=train_data, validation_data=train_data,
        historic_cols=['temp'], future_cols=['temp_fcst', 'temp'],
        target_col='y', forecasting_horizon=3, historic_input_sequence_length=10,
        number_of_epochs=1, batch_size=16, save_path=str(tmp_path), device=torch.device('cpu'),
    )
    fc.fit()
    preds = fc.predict(train_data)
    assert preds.shape[-1] == len(levels)
    assert np.all(preds[..., 0] <= preds[..., 1])
    assert np.all(preds[..., 1] <= preds[..., 2])


def test_duplicate_embedding_names_raise(tmp_path, csv_path):
    train_data = TimeSeriesData([DataSource(csv=csv_path, start='2020-01-01', end='2020-12-31')])
    net_a = EmbeddingNetwork(name='dup', variables=['temp'], embedding_dim=2, historic=True, future=False)
    net_b = EmbeddingNetwork(name='dup', variables=['precip'], embedding_dim=2, historic=True, future=False)
    model = LSTMHistoric(hidden_size=4, num_layers=1, dropout_rate=0.0)
    with pytest.raises(ValueError):
        Forecaster(
            model=model, name='dupfail', embeddings=[net_a, net_b], loss=MAE(),
            training_data=train_data, validation_data=train_data,
            historic_cols=['temp', 'precip'], future_cols=[],
            target_col='y', forecasting_horizon=2, historic_input_sequence_length=5,
            number_of_epochs=1, save_path=str(tmp_path), device=torch.device('cpu'),
        )
