import torch
from torchalgebras.base import Algebra, AlgebraicTensor
from torchalgebras.algebras import BasicAlgebra

def _assert_allclose(x, y):
    real = x @ y
    sem = (AlgebraicTensor(x, BasicAlgebra()) @ AlgebraicTensor(y, BasicAlgebra())).data
    assert real.shape == sem.shape, f'{real.shape = }, {sem.shape = }'
    assert torch.allclose(real, sem, rtol=1e-5,atol=1e-5), f'{(real - sem).abs().mean() = }'


def test_semiring():
    # vv
    x = torch.randn(10); y = torch.randn(10)
    _assert_allclose(x, y)

    # outer
    x = torch.randn(10, 1); y = torch.randn(1, 4)
    _assert_allclose(x, y)

    # vM
    x = torch.randn(10); y = torch.randn(10, 3)
    _assert_allclose(x, y)

    # Mv
    x = torch.randn(10, 3); y = torch.randn(3)
    _assert_allclose(x, y)

    # MM
    x = torch.randn(12, 5); y = torch.randn(5, 8)
    _assert_allclose(x, y)

    # batched + broadcasting
    x = torch.randn(4, 3, 12, 5); y = torch.randn(3, 5, 8)
    _assert_allclose(x, y)

    # broadcasting2
    x = torch.randn(4, 3, 12, 5); y = torch.randn(1, 5, 8)
    _assert_allclose(x, y)

    # Mv broadcasting
    x = torch.randn(4,3,12,5); y = torch.randn(5)
    _assert_allclose(x, y)

    # vM broadcasting
    x = torch.randn(12); y = torch.randn(4,3,12,5)
    _assert_allclose(x, y)
