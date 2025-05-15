# pylint: disable=abstract-method
import math

from typing import Any, Literal, cast
import torch

ALGEBRAS: "dict[str, Algebra]" = dict()

def register_(algebra, *keys):
    for k in keys:
        if k in ALGEBRAS: raise KeyError(f"key {k} already exists")
        ALGEBRAS[k] = algebra

class Algebra:
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def neg(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sub(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.add(x, self.neg(y))

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def div(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mul(x, self.reciprocal(_ensure_tensor(y, x)))

    def reciprocal(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pow(self, base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sum(self, x: torch.Tensor, dim: int | None = None, keepdim=False) -> torch.Tensor:
        raise NotImplementedError

    def prod(self, x: torch.Tensor, dim: int | None = None, keepdim=False) -> torch.Tensor:
        raise NotImplementedError

    def min(self, x: torch.Tensor, dim: int | None = None, keepdim=False) -> torch.Tensor:
        if dim is None: return torch.min(x)
        return x.amin(dim, keepdim)

    def max(self, x: torch.Tensor, dim: int | None = None, keepdim=False) -> torch.Tensor:
        if dim is None: return torch.max(x)
        return x.amax(dim, keepdim)

    def mm(self, x: torch.Tensor, y: torch.Tensor):
        # this imlements matmul by calling mul and sum

        x_squeeze = False
        y_squeeze = False

        if x.ndim == 1:
            x_squeeze = True
            x = x.unsqueeze(0)

        if y.ndim == 1:
            y_squeeze = True
            y = y.unsqueeze(1)

        res = self.sum(self.mul(x.unsqueeze(-1), y.unsqueeze(-3)), dim = -2)

        if x_squeeze: res = res.squeeze(-2)
        if y_squeeze: res = res.squeeze(-1)

        return res

    def matmul(self, x:torch.Tensor, y:torch.Tensor):
        return self.mm(x, y)

def get_algebra(algebra: str | Algebra) -> Algebra:
    if isinstance(algebra, str): return ALGEBRAS[algebra]
    if isinstance(algebra, Algebra): return algebra
    raise TypeError(type(algebra))

def _ensure_tensor(x, ref: "torch.Tensor | AlgebraicTensor") -> torch.Tensor:
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, AlgebraicTensor): return x.data
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)

AlgebraType = Algebra | str

class AlgebraicTensor:
    def __init__(self, data, algebra: AlgebraType):
        super().__init__()
        algebra = get_algebra(algebra)
        if not isinstance(data, (torch.Tensor, AlgebraicTensor)): data = torch.as_tensor(data)
        self.data: torch.Tensor = cast(torch.Tensor, data)
        self.algebra = algebra
        self._handled_funcs = {v for v in dir(algebra) if not v.startswith('_')}

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        algebra = None
        handled_funcs = None
        all_args = list(args) + list(kwargs.values())
        for arg in all_args:
            if isinstance(arg, AlgebraicTensor):
                algebra = arg.algebra
                handled_funcs = arg._handled_funcs
        assert algebra is not None
        assert handled_funcs is not None

        args = [a.data if isinstance(a, AlgebraicTensor) else a for a in args]
        kwargs = {k: v.data if isinstance(v, AlgebraicTensor) else v for k, v in kwargs.items()}

        if func.__name__ in handled_funcs:
            func = getattr(algebra, func.__name__)

        ret = func(*args, **kwargs)
        return cls(ret, algebra=algebra)

    def __repr__(self):
        r: str = self.as_tensor().__repr__()
        op = r.find('(')
        cl = r.find(')')
        if cl == -1: cl = len(r)
        r = r[op+1:cl]
        return f"{self.algebra.__class__.__name__}Tensor({r})"

    def as_tensor(self):
        return torch.as_tensor(self.data)

    def as_subclass(self, cls):
        if cls == AlgebraicTensor: return cls(self.data, self.algebra)
        return self.data.as_subclass(cls)

    @property
    def device(self): return self.data.device
    @property
    def dtype(self): return self.data.dtype
    @property
    def requires_grad(self): return self.data.requires_grad
    @property
    def grad(self): return self.__class__(self.data.grad, self.algebra)

    def ndimension(self): return self.data.ndimension()
    @property
    def ndim(self): return self.data.ndim

    def size(self, dim=None): return self.data.size(dim)
    @property
    def shape(self): return self.data.shape

    def numel(self): return self.data.numel()

    def requires_grad_(self, mode=True):
        return self.__class__(self.data.requires_grad_(mode), self.algebra)

    def to(self, *args, **kwargs):
        return self.__class__(self.data.to(*args, **kwargs), self.algebra)

    def cpu(self, memory_format=torch.preserve_format):
        return self.__class__(self.data.cpu(memory_format=memory_format), self.algebra)

    def cuda(self, memory_format=torch.preserve_format):
        return self.__class__(self.data.cuda(memory_format=memory_format), self.algebra)

    def clone(self):
        return self.__class__(self.data.clone(), self.algebra)

    def detach(self):
        return self.__class__(self.data.detach(), self.algebra)

    def add(self, other: "NumberOrTensor"):
        return self.__class__(self.algebra.add(self.data, _ensure_tensor(other, self)), self.algebra)
    def radd(self, other: "NumberOrTensor"):
        return self.__class__(self.algebra.add(_ensure_tensor(other, self), self.data), self.algebra)
    def __add__(self, other): return self.add(other)
    def __radd__(self, other): return self.radd(other)

    def sub(self, other):
        return self.__class__(self.algebra.sub(self.data, _ensure_tensor(other, self)), self.algebra)
    def rsub(self, other):
        return self.__class__(self.algebra.sub(_ensure_tensor(other, self), self.data), self.algebra)
    def __sub__(self, other): return self.sub(other)
    def __rsub__(self, other): return self.rsub(other)

    def neg(self):
        return self.__class__(self.algebra.neg(self.data), self.algebra)
    def __neg__(self): return self.neg()

    def mul(self, other):
        return self.__class__(self.algebra.mul(self.data, _ensure_tensor(other, self)), self.algebra)
    def rmul(self, other):
        return self.__class__(self.algebra.mul(_ensure_tensor(other, self), self.data), self.algebra)
    def __mul__(self, other): return self.mul(other)
    def __rmul__(self, other): return self.rmul(other)

    def div(self, other):
        return self.__class__(self.algebra.div(self.data, _ensure_tensor(other, self)), self.algebra)
    def rdiv(self, other):
        return self.__class__(self.algebra.div(_ensure_tensor(other, self), self.data), self.algebra)
    def __div__(self, other): return self.div(other)
    def __rdiv__(self, other): return self.rdiv(other)

    def reciprocal(self):
        return self.__class__(self.algebra.reciprocal(self.data), self.algebra)

    def pow(self, other):
        return self.__class__(self.algebra.pow(self.data, _ensure_tensor(other, self)), self.algebra)
    def rpow(self, other):
        return self.__class__(self.algebra.pow(_ensure_tensor(other, self), self.data), self.algebra)
    def __pow__(self, other): return self.pow(other)
    def __rpow__(self, other): return self.rpow(other)

    def sum(self, dim: int | None, keepdim = False):
        return self.__class__(self.algebra.sum(self.data, dim, keepdim), self.algebra)

    def min(self, dim: int | None, keepdim = False):
        return self.__class__(self.algebra.min(self.data, dim, keepdim), self.algebra)

    def max(self, dim: int | None, keepdim = False):
        return self.__class__(self.algebra.max(self.data, dim, keepdim), self.algebra)

    def mm(self, other): return self.__class__(self.algebra.mm(self.data, _ensure_tensor(other, self)), self.algebra)
    def rmm(self, other): return self.__class__(self.algebra.mm(_ensure_tensor(other, self), self.data), self.algebra)
    def matmul(self, other): return self.mm(other)
    def __matmul__(self, other): return self.mm(other)
    def __rmatmul__(self, other): return self.rmm(other)

    # -------------------------------- other funcs ------------------------------- #
    # i add random functions as I need to use them
    def diagonal(self, offset: int = 0, dim1: int = 0, dim2: int = 1):
        return self.__class__(self.data.diagonal(offset, dim1, dim2), self.algebra)

def algebraic_tensor(data, algebra: Algebra):
    t = AlgebraicTensor(data, algebra)
    return cast(torch.Tensor, t)

def totorch(x):
    if isinstance(x, AlgebraicTensor): return x.data
    if isinstance(x, torch.Tensor): return x
    raise TypeError(x)

NumberOrTensor = int | float | torch.Tensor# | AlgebraicTensor # since it is not a subclass, it is just casted as tensor
