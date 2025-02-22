
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.linalg import cholesky_ex, solve_triangular
from torch.nn import Module, Parameter
from torch.nn import functional as F

from .utils import MaternExpectation, SquaredExponentialExpectation



class Kernel(Module, ABC):

	def __init__(self, dim0: int, dim1: int):
		Module.__init__(self)

		# [D, Q]
		self._outputscale = Parameter(torch.zeros(dim0, dim1))
		self._lengthscale = Parameter(torch.zeros(dim0, dim1))

	@property
	def outputscale(self) -> Tensor:
		return F.softplus(self._outputscale) + 1e-6

	@property
	def lengthscale(self) -> Tensor:
		return F.softplus(self._lengthscale) + 1e-6
	
	@abstractmethod
	def expectation(self) -> Tensor:
		...

	def Kuu_cholesky(self, induc_loc: Tensor) -> Tensor:
		"""
		z ~ [D, Q, M]
		->
		Kuu = L @ L.mT ~ [D, Q, M, M]
		"""
		lengthscale = self.lengthscale.unsqueeze(-1).unsqueeze(-1)
		dist = induc_loc.unsqueeze(-1).sub(induc_loc.unsqueeze(-2))
		Kuu = self.expectation(lengthscale, dist)
		return cholesky_ex(Kuu + 1e-6 * torch.eye(Kuu.size(-1))).L

	def Kuu_inv(self, induc_loc: Tensor) -> Tensor:
		"""
		z ~ [D, Q, M]
		->
		Kuu_inv ~ [D, Q, M, M]
		"""
		L = self.Kuu_cholesky(induc_loc)
		L_inv = solve_triangular(L, torch.eye(induc_loc.size(-1)), upper=False)
		return L_inv.mT.matmul(L_inv)

	def Cuf(self, induc_loc: Tensor, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		z ~ [D, Q, M]
		x_mean & x_var ~ [B, Q]
		->
		Cuf ~ [B, D, Q, M]
		"""
		lengthscale = self.lengthscale.unsqueeze(-1)
		dist = x_mean.unsqueeze(-2).unsqueeze(-1).sub(induc_loc)
		if x_var is not None:
			x_var = x_var.unsqueeze(-2).unsqueeze(-1)
		return self.expectation(lengthscale, dist, x_var)

	def Cff(self, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		x_var ~ [B, Q]
		->
		Cff ~ [B, D, Q]
		"""
		sigma_sq = None if x_var is None else x_var.unsqueeze(-2).mul(2)
		return self.expectation(self.lengthscale, None, sigma_sq)


class SquaredExponentialKernel(Kernel):

	def __init__(self, dim0: int, dim1: int):
		Kernel.__init__(self, dim0, dim1)

	def expectation(self, lengthscale: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None):
		return SquaredExponentialExpectation(lengthscale, mu, sigma_sq)


class MaternKernel(Kernel):

	def __init__(self, nu: float, dim0: int, dim1: int):
		Kernel.__init__(self, dim0, dim1)

		assert nu in [0.5, 1.5, 2.5]
		self.nu = nu  # nu

	def expectation(self, lengthscale: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None):
		return MaternExpectation(self.nu, lengthscale, mu, sigma_sq)

	def extra_repr(self) -> str:
		return f"nu={self.nu}"
	

class Matern0p5Kernel(MaternKernel):

	def __init__(self, dim0: int, dim1: int):
		MaternKernel.__init__(self, 0.5, dim0, dim1)


class Matern1p5Kernel(MaternKernel):

	def __init__(self, dim0: int, dim1: int):
		MaternKernel.__init__(self, 1.5, dim0, dim1)


class Matern2p5Kernel(MaternKernel):

	def __init__(self, dim0: int, dim1: int):
		MaternKernel.__init__(self, 2.5, dim0, dim1)


