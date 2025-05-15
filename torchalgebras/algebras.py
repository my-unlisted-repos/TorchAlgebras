# pylint: disable=abstract-method

from typing import Literal

import torch

from .base import Algebra, register_


class BasicAlgebra(Algebra):
    def add(self, x, y): return x + y
    def neg(self, x): return -x
    def mul(self, x, y): return x * y
    def div(self, x, y): return x / y
    def reciprocal(self, x): return x.reciprocal()
    def pow(self, base, exponent): return base ** exponent
    def sum(self, x, dim=None, keepdim=False): return x.sum(dim, keepdim)
    def prod(self, x, dim=None, keepdim=False): return x.prod(dim, keepdim)

class ElementaryAlgebra(BasicAlgebra):
    """uses pytorch matmul which is faster"""
    def sub(self,x,y): return x-y
    def rsub(self,x,y): return y-x
    def div(self, x, y): return x/y
    def rdiv(self, x, y): return y/x
    def mm(self,x,y): return x @ y
    def rmm(self,x,y): return y @ x
register_(ElementaryAlgebra(), 'elementary')

class TropicalSemiring(Algebra):
    def __init__(self, type: Literal['min', 'max'] = 'min'):
        self.type = type

    def add(self, x, y):
        if self.type == 'min': return torch.minimum(x, y)
        if self.type == 'max': return torch.maximum(x, y)
        raise ValueError(self.type)

    def sum(self, x, dim = None, keepdim = False):
        if self.type == 'min': return x.amin(dim, keepdim) # type:ignore
        if self.type == 'max': return x.amax(dim, keepdim) # type:ignore
        raise ValueError(self.type)

    def sub(self, x, y):
        if self.type == 'min': raise RuntimeError
        if self.type == 'max':
            # x - y is equivalent to solving max(a, y) = x for a
            # if y == x, then a <= y
            # if y < x then a = x
            # if y > x then there is no solution lets relax into a=y
            # so it becomes max(x, y)
            return torch.maximum(x, y)
        raise ValueError(self.type)

    def mul(self, x, y): return x + y
    def pow(self, base, exponent): return base*exponent
    def div(self, x, y): return x - y
    def reciprocal(self, x): return -x

    def prod(self, x, dim=None, keepdim=False): return torch.sum(x, dim, keepdim)
register_(TropicalSemiring('min'), 'tropical', 'tropical_min')
register_(TropicalSemiring('max'), 'tropical_max')

class FuzzySemiring(Algebra):
    def add(self, x, y): return torch.maximum(x, y)
    def sum(self, x, dim=None, keepdim=False): return x.amax(dim, keepdim) # type:ignore
    def mul(self, x, y): return torch.minimum(x, y)
register_(FuzzySemiring(), 'fuzzy')

class LukasiewiczSemiring(Algebra):
    def add(self, x, y): return torch.maximum(x, y)
    def sum(self, x, dim=None, keepdim=False): return x.amax(dim, keepdim) # type:ignore
    def mul(self, x, y): return torch.clip(x+y-1, min=0)
    def reciprocal(self, x): return 2-x
    def pow(self, base, exponent): return torch.clip(exponent * base - (exponent - 1), min=0)
register_(LukasiewiczSemiring(), 'lukasiewicz')

class ViterbiSemiring(Algebra):
    def __init__(self, type: Literal['min', 'max'] = 'max'):
        self.type = type
    def add(self, x, y):
        if self.type == 'max': return torch.maximum(x, y)
        if self.type == 'min': return torch.minimum(x, y)
        raise ValueError(self.type)

    def sum(self, x, dim=None, keepdim=False):
        if self.type == 'max': return x.amax(dim, keepdim) # type:ignore
        if self.type == 'min': return x.amin(dim, keepdim) # type:ignore
        raise ValueError(self.type)

    def mul(self, x, y): return x * y
    def neg(self, x): raise NotImplementedError # not defined
    def reciprocal(self, x): return x.reciprocal()
    def pow(self, base, exponent): return base ** exponent
register_(ViterbiSemiring('max'), 'viterbi', 'viterbi_max')
register_(ViterbiSemiring('min'), 'viterbi_min')

class LogSemiring(Algebra):
    def add(self, x, y): return torch.logaddexp(x, y)
    def mul(self, x, y): return x + y

    def sum(self, x, dim=None, keepdim=False):
        if dim is None: return torch.logsumexp(input=x, dim=None) # type:ignore
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    def reciprocal(self, x): return -x
    def pow(self, base, exponent): return base * exponent
    def neg(self, x): raise NotImplementedError # not defined
register_(LogSemiring(), 'log')


class ProbabilisticLogicSemiring(Algebra):
    def add(self, x, y): return x + y - x * y
    def mul(self, x, y): return x * y

    def sum(self, x, dim=None, keepdim=False):
        return 1 - torch.prod(1 - x, dim=dim, keepdim=keepdim)

    def neg(self, x):
        eps = torch.finfo(x.dtype).eps
        denom = (1 - x)
        denom = torch.where(denom.abs() <= eps, eps, denom)
        return x / denom

    def reciprocal(self, x): return 1/x
    def pow(self, base, exponent): return base * exponent
register_(ProbabilisticLogicSemiring(), 'probabilistic', 'prob')


class ModuloRing(Algebra):
    def __init__(self, N: float):
        self.N = N

    def add(self, x, y): return torch.remainder(x + y, self.N)
    def neg(self, x): return torch.remainder(-x, self.N)
    def mul(self, x, y): return torch.remainder(x * y, self.N)
    def sum(self, x, dim=None, keepdim=False) -> torch.Tensor: return torch.remainder(torch.sum(x, dim=dim, keepdim=keepdim), self.N)
    def reciprocal(self, x): return torch.remainder(torch.reciprocal(x), self.N) # as proxy
    def pow(self, base, exponent) -> torch.Tensor: return torch.remainder(torch.pow(base, exponent), self.N)
register_(ModuloRing(1), 'modulo1')
register_(ModuloRing(5), 'modulo5')

