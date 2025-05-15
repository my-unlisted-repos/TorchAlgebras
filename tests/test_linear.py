import torch
from torchalgebras.algebras import BasicAlgebra
from torchalgebras.base import Algebra, AlgebraicTensor
from torchalgebras.layers import AlgebraicLinear


def _assert_allclose(x, y):
    assert x.shape == y.shape, f'{x.shape = }, {y.shape = }'
    assert torch.allclose(x, y, rtol=1e-5,atol=1e-5), f'{(x - y).abs().mean() = }'

@torch.no_grad
def _assert_layers(W,b):
    in_channels, out_channels = W.shape[1], W.shape[0]
    torch_linear = torch.nn.Linear(in_channels, out_channels)
    torch_linear.weight.set_(W)
    torch_linear.bias.set_(b)

    semiring_linear = AlgebraicLinear(BasicAlgebra(), in_channels, out_channels)
    semiring_linear.weight.set_(W)
    semiring_linear.bias.set_(b) # type:ignore

    # vec
    x = torch.randn(in_channels)
    _assert_allclose(torch_linear(x), semiring_linear(x))

    # batched1
    x = torch.randn(32, in_channels)
    _assert_allclose(torch_linear(x), semiring_linear(x))

    # batched2
    x = torch.randn(32, 3, in_channels)
    _assert_allclose(torch_linear(x), semiring_linear(x))

def test_semiring_linear():
    W = torch.randn(12, 1)
    b = torch.randn(12)
    _assert_layers(W, b)

    W = torch.randn(1, 12)
    b = torch.randn(1)
    _assert_layers(W, b)

    W = torch.randn(64, 32)
    b = torch.randn(64)
    _assert_layers(W, b)