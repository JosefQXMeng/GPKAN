
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import functional as F

from .utils import MaternExpectation, SquaredExponentialExpectation



class RBF(Module, ABC):

	def __init__(self, dim0: int, dim1: int, num_comp: int):
		Module.__init__(self)

		self.num_comp = num_comp  # K
		# [D, Q, K]
		self.weight = Parameter(torch.zeros(dim0, dim1, num_comp))
		self.center = Parameter(torch.rand(dim0, dim1, num_comp).mul(2).sub(1))
		self._lengthscale = Parameter(torch.zeros(dim0, dim1, num_comp))

	@property
	def lengthscale(self) -> Tensor:
		return F.softplus(self._lengthscale) + 1e-6

	@abstractmethod
	def expectation(self) -> Tensor:
		...

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		x_mean & x_var ~ [B, Q]
		->
		E[m(x)] ~ [B, D, Q]
		"""
		dist = x_mean.unsqueeze(-1).sub(self.center)
		if x_var is not None:
			x_var = x_var.unsqueeze(-1)
		return self.expectation(self.lengthscale, dist, x_var).mul(self.weight).sum(-1)
	
	def add_comp(self, num: int) -> None:
		weight = self.weight.data
		self.weight = Parameter(torch.cat([weight, torch.zeros(*weight.size()[:-1], num)], dim=-1))
		center = self.center.data
		self.center = Parameter(torch.cat([center, torch.rand(*center.size()[:-1], num).mul(2).sub(1)], dim=-1))
		_lengthscale = self._lengthscale.data
		self._lengthscale = Parameter(torch.cat([_lengthscale, torch.zeros(*_lengthscale.size()[:-1], num)], dim=-1))
		self.num_comp += num

	def extra_repr(self) -> str:
		return (bool(self.num_comp)*f"num_comp={self.num_comp}")


class GaussianRBF(RBF):

	def __init__(self, dim0: int, dim1: int, num_comp: int):
		RBF.__init__(self, dim0, dim1, num_comp)

	def expectation(self, lengthscale: Tensor, mu: Tensor, sigma_sq: Optional[Tensor] = None) -> Tensor:
		return SquaredExponentialExpectation(lengthscale, mu, sigma_sq)


class AbsoluteExponentialRBF(RBF):

	def __init__(self, dim0: int, dim1: int, num_comp: int):
		RBF.__init__(self, dim0, dim1, num_comp)

	def expectation(self, lengthscale: Tensor, mu: Tensor, sigma_sq: Optional[Tensor] = None) -> Tensor:
		return MaternExpectation(0.5, lengthscale, mu, sigma_sq)


