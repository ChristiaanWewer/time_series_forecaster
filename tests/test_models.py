import pytest
import torch

from src.models import LSTMHistoric, LSTMEncoderDecoder


def test_lstm_historic_point_output():
    model = LSTMHistoric(hidden_size=8, num_layers=1, dropout_rate=0.0)
    model.bind(historic_input_size=3, future_input_size=0, forecasting_horizon=5,
               n_outputs=1, output_kind='point', output_levels=None)
    x_h = torch.randn(4, 10, 3)
    x_f = torch.randn(4, 5, 0)
    out = model(x_h, x_f)
    assert out.shape == (4, 5, 1)


def test_lstm_historic_quantile_output_is_monotonic():
    levels = [0.1, 0.5, 0.9]
    model = LSTMHistoric(hidden_size=8, num_layers=1, dropout_rate=0.0)
    model.bind(historic_input_size=3, future_input_size=0, forecasting_horizon=5,
               n_outputs=len(levels), output_kind='quantile', output_levels=levels)
    x_h = torch.randn(4, 10, 3)
    x_f = torch.randn(4, 5, 0)
    out = model(x_h, x_f)
    assert out.shape == (4, 5, 3)
    assert torch.all(out[..., 0] <= out[..., 1])
    assert torch.all(out[..., 1] <= out[..., 2])


def test_lstm_encoder_decoder_sequence_output():
    model = LSTMEncoderDecoder(hidden_size=6, num_layers=1, dropout_rate=0.0, encoder_downscale_layer_size=4)
    model.bind(historic_input_size=2, future_input_size=1, forecasting_horizon=3,
               n_outputs=1, output_kind='point', output_levels=None)
    x_h = torch.randn(2, 10, 2)
    x_f = torch.randn(2, 3, 1)
    out = model(x_h, x_f)
    assert out.shape == (2, 3, 1)


def test_double_bind_raises():
    model = LSTMHistoric(hidden_size=4, num_layers=1, dropout_rate=0.0)
    model.bind(historic_input_size=2, future_input_size=0, forecasting_horizon=2,
               n_outputs=1, output_kind='point', output_levels=None)
    with pytest.raises(RuntimeError):
        model.bind(historic_input_size=2, future_input_size=0, forecasting_horizon=2,
                    n_outputs=1, output_kind='point', output_levels=None)


def test_forward_before_bind_raises():
    model = LSTMHistoric(hidden_size=4, num_layers=1, dropout_rate=0.0)
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 5, 3), torch.randn(2, 2, 0))


def test_parameter_breakdown_shape():
    model = LSTMHistoric(hidden_size=4, num_layers=1, dropout_rate=0.0)
    model.bind(historic_input_size=2, future_input_size=0, forecasting_horizon=2,
               n_outputs=1, output_kind='point', output_levels=None)
    rows = model.parameter_breakdown()
    assert rows[-1][0] == 'total'
    assert rows[-1][1] == sum(p.numel() for p in model.parameters())
    assert rows[-1][1] == sum(count for _, count in rows[:-1])
