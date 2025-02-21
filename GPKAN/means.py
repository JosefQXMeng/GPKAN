
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import functional as F

from .utils import StandardNormalCDF, SquaredExponentialExpectation



class GaussianRBF(Module):
	
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

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		x_mean & x_var & [B, 1, Q]
		->
		E[m(x)] ~ [B, D, Q]
		"""
		dist = x_mean.unsqueeze(-1).sub(self.center)
		if x_var is not None:
			x_var = x_var.unsqueeze(-1)
		return SquaredExponentialExpectation(self.lengthscale, dist, x_var).mul(self.weight).sum(-1)

	def extra_repr(self) -> str:
		return (f"num_comp={self.num_comp}")


# TODO: smoother?
class AbsoluteExponentialRBF(Module):

	def __init__(self, dim0: int, dim1: int, num_comp: int):
		Module.__init__(self)

		self.num_comp = num_comp  # K
		# [D, Q, K]
		self.w = Parameter(torch.zeros(dim0, dim1, num_comp))
		self.c = Parameter(torch.randn(dim0, dim1, num_comp))
		self._l = Parameter(torch.zeros(dim0, dim1, num_comp))
	
	@property
	def l(self) -> Tensor:
		return F.softplus(self._l) + 1e-6
	
	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		x_mean & x_var & [B, 1, Q]
		->
		E[m(x)] ~ [B, D, Q]
		"""
		l = self.l
		mu = x_mean.unsqueeze(-1).sub(self.c).div(l)
		if x_var is None:
			res = mu.abs().mul(-1).exp()
		else:
			sigma = x_var.unsqueeze(-1).sqrt().div(l)
			res = StandardNormalCDF(mu.div(sigma).sub(sigma)).mul(mu.mul(-1).exp()).add(
				StandardNormalCDF(mu.mul(-1).div(sigma).sub(sigma)).mul(mu.exp())
			).mul(sigma.pow(2).div(2).exp())
		return res.mul(self.w).sum(-1)

	def extra_repr(self) -> str:
		return (f"num_comp={self.num_comp}")


