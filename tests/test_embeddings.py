import pytest
import torch

from src.forecaster.forecaster import Forecaster
from src.models.embeddings import EmbeddingNetwork


def test_missing_variable_raises():
    net = EmbeddingNetwork(name='bad', variables=['nope'], embedding_dim=2, historic=True, future=False)
    with pytest.raises(ValueError, match='nope'):
        net.bind(historic_cols=['temp'], future_cols=[])


def test_missing_side_flag_raises():
    with pytest.raises(ValueError):
        EmbeddingNetwork(name='bad', variables=['temp'], embedding_dim=2, historic=False, future=False)


def test_forward_before_bind_raises():
    net = EmbeddingNetwork(name='unbound', variables=['temp'], embedding_dim=2, historic=True, future=False)
    with pytest.raises(RuntimeError):
        net(torch.randn(2, 3, 1))


def test_double_bind_raises():
    net = EmbeddingNetwork(name='ok', variables=['temp'], embedding_dim=2, historic=True, future=False)
    net.bind(historic_cols=['temp'], future_cols=[])
    with pytest.raises(RuntimeError):
        net.bind(historic_cols=['temp'], future_cols=[])


def test_historic_only_forward_shape():
    net = EmbeddingNetwork(name='h', variables=['temp'], embedding_dim=4, historic=True, future=False)
    net.bind(historic_cols=['temp', 'precip'], future_cols=[])
    x = torch.randn(3, 5, 1)  # already sliced to this net's one claimed column
    out = net(x)
    assert out.shape == (3, 5, 4)
    assert net._historic_idx == [0]
    assert net._future_idx is None


def test_shared_historic_and_future_uses_one_instance():
    net = EmbeddingNetwork(name='shared', variables=['elev'], embedding_dim=3, historic=True, future=True)
    net.bind(historic_cols=['temp', 'elev'], future_cols=['fcst', 'elev'])
    assert net._historic_idx == [1]
    assert net._future_idx == [1]
    x_h = torch.randn(2, 5, 1)
    x_f = torch.randn(2, 3, 1)
    out_h = net(x_h)
    out_f = net(x_f)
    assert out_h.shape == (2, 5, 3)
    assert out_f.shape == (2, 3, 3)
    # one shared nn.Sequential — same weights used both times
    assert list(net.parameters())


def test_multilayer_with_output_activation():
    net = EmbeddingNetwork(
        name='mlp', variables=['a', 'b'], embedding_dim=5, historic=True, future=False,
        num_layers=2, hidden_size=8, output_activation='gelu',
    )
    net.bind(historic_cols=['a', 'b', 'c'], future_cols=[])
    out = net(torch.randn(4, 7, 2))
    assert out.shape == (4, 7, 5)


def test_unclaimed_columns_pass_through_and_concatenate_after_embedded_output():
    # embed only 'landuse_code' (index 2); temp/precip/flow (indices 0, 1, 3) are
    # untouched passthrough columns
    historic_cols = ['temp', 'precip', 'landuse_code', 'flow']
    net = EmbeddingNetwork(name='landuse', variables=['landuse_code'], embedding_dim=3,
                            historic=True, future=False)
    net.bind(historic_cols=historic_cols, future_cols=[])

    applicable, passthrough_idx, output_size = Forecaster._prepare_side(
        [net], historic_cols, '_historic_idx'
    )
    assert passthrough_idx == [0, 1, 3]
    assert output_size == 3 + len(passthrough_idx)  # embedding_dim + passthrough count

    x = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    out = Forecaster._embed_side(x, applicable, passthrough_idx)
    assert out.shape == (2, 5, output_size)

    # embedded output comes first (config order), passthrough columns follow,
    # untouched and in their original relative order
    expected_embedded = net(x[..., [2]])
    expected_passthrough = x[..., [0, 1, 3]]
    assert torch.allclose(out[..., :3], expected_embedded)
    assert torch.allclose(out[..., 3:], expected_passthrough)


def test_no_embeddings_is_pure_passthrough():
    historic_cols = ['temp', 'precip', 'flow']
    applicable, passthrough_idx, output_size = Forecaster._prepare_side(
        [], historic_cols, '_historic_idx'
    )
    assert applicable == []
    assert passthrough_idx == [0, 1, 2]
    assert output_size == len(historic_cols)

    x = torch.randn(2, 5, 3)
    out = Forecaster._embed_side(x, applicable, passthrough_idx)
    assert torch.equal(out, x)
