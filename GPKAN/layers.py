
import math
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from .abstr import Layer
from .means import AbsoluteExponentialRBF, GaussianRBF
from .kernels import Matern0p5Kernel, Matern1p5Kernel, Matern2p5Kernel, SquaredExponentialKernel



class FCLayer(Layer):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_induc: int,
		kernel : Optional[Union[str, Callable]] = "SquaredExponentialKernel",
		mean_func: Optional[Union[str, Callable]] = None,
		num_comp: int = 0,
		degree: int = 0,
		sep_induc: bool = True,
	):
		Layer.__init__(self, in_dim, out_dim)
		
		self.num_induc = num_induc  # M
		if isinstance(kernel, str):
			kernel = eval(kernel)
		assert kernel is None or callable(kernel)

		if kernel is not None and num_induc:
			self.kernel = kernel(out_dim, in_dim)
			s = [out_dim, in_dim, num_induc]
			# z ~ [(D), Q, M]
			self.induc_loc = Parameter(torch.rand(s if sep_induc else s[1:]).mul(2).sub(1))
			degree = degree if kernel == SquaredExponentialKernel else 0
			self.degree = degree
			self.polynomial_coef = Parameter(torch.zeros(degree+1, *s))
			self.init_moment_coef(degree)
		else:
			self.kernel = None

		if isinstance(mean_func, str):
			mean_func = eval(mean_func)
		assert mean_func is None or callable(mean_func)

		if mean_func is not None and num_comp:
			self.mean_func = mean_func(out_dim, in_dim, num_comp)
		else:
			self.mean_func = None
		
		self.l1_norm = torch.zeros(out_dim, in_dim)
		self.entropy = torch.zeros(out_dim, in_dim)
	
	def init_moment_coef(self, degree: int) -> None:
		coef_list = []
		for n in range(1, degree+1):
			sublist = []
			for k in range(n+1):
				if not k % 2:
					sublist.append(math.comb(n, k) * math.prod(range(1, k, 2)))
			coef_list.append(sublist)
		self.moment_coef = coef_list

	def normal_moment(self, mean: Tensor, var: Tensor, n: int) -> Tensor:
		assert n in range(1, self.degree+1)
		E = torch.zeros([])
		for k in range(n+1):
			if not k % 2:
				E = mean.pow(n-k).mul(var.pow(k//2)).mul(self.moment_coef[n-1][k//2]).add(E)
		return E
	
	def expected_u(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> Tensor:
		"""
		x_mean & x_var ~ [B, 1, Q]
		->
		E[u(x)] ~ [(B), D, D, M]
		"""
		expected_u = self.polynomial_coef[0]
		if self.degree:
			x_mean = x_mean.unsqueeze(-2).unsqueeze(-1)
			if x_var is not None:
				x_var = x_var.unsqueeze(-2).unsqueeze(-1)
				lengthscale = self.kernel.lengthscale.unsqueeze(-1)
				cross_prod = x_mean.mul(lengthscale).add(x_var.mul(self.induc_loc))
				var_sum = x_var.add(lengthscale)
			q_mean = x_mean if x_var is None else cross_prod.div(var_sum)
			q_var = torch.zeros([]) if x_var is None else x_var.mul(lengthscale).div(var_sum)
		for k in range(1, self.degree+1):
			expected_u = self.normal_moment(q_mean, q_var, k).mul(self.polynomial_coef[k]).add(expected_u)
		return expected_u
		
	def compute_w(
		self, x_mean: Tensor, x_var: Optional[Tensor] = None, regularize: bool = False,
	) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		x_mean & x_var ~ [B, Q]
		->
		w_mean & w_var ~ [B, D, Q]
		"""

		if self.mean_func is None:
			E_prior_mean = induc_mean = 0
		else:
			# E[m(x)] ~ [B, D, Q]
			E_prior_mean = self.mean_func(x_mean.unsqueeze(-2), None if x_var is None else x_var.unsqueeze(-2))
			if self.num_induc:
				# m(z) ~ [D, Q, M]
				induc_mean = self.mean_func(self.induc_loc.movedim(-1,0)).movedim(0,-1)
		
		w_mean = E_prior_mean
		w_var = None

		if self.num_induc:

			# Kuu_inv ~ [D, Q, M, M]
			Kuu_inv = self.kernel.Kuu_inv(self.induc_loc)
			# Cuf ~ [B, D, Q, M]
			Cuf = self.kernel.Cuf(self.induc_loc, x_mean, x_var)
			# Cff ~ [(B, D, Q)]
			Cff = self.kernel.Cff(x_var)
			# E[u(x)] ~ [(B), D, Q, M]
			E_u = self.expected_u(x_mean, x_var)

			w_mean += Kuu_inv.matmul(E_u.sub(induc_mean).unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)
			w_var = Cff.sub(Kuu_inv.matmul(Cuf.unsqueeze(-1)).squeeze(-1).mul(Cuf).sum(-1)).mul(self.kernel.outputscale)

		if regularize:
			self.l1_norm = w_mean.abs().mean(0)
			self.entropy = w_var.mul(math.pi).mul(2).log().add(1).div(2).mean(0)

		return w_mean, w_var

	def forward(
		self, x_mean: Tensor, x_var: Optional[Tensor] = None, regularize: bool = False,
	) -> tuple[Tensor, Tensor]:
		"""
		x_mean & x_var ~ [B, Q]
		->
		f_mean & f_var ~ [B, D]
		"""
		w_mean, w_var = self.compute_w(x_mean, x_var, regularize)
		f_mean = w_mean.sum(-1)
		f_var = None if w_var is None else w_var.sum(-1) + 1e-6
		return f_mean, f_var

	def extra_repr(self) -> str:
		value = f"in_dim={self.in_dim}, out_dim={self.out_dim}"
		if self.kernel is not None and self.num_induc:
			value += + bool(self.num_induc) * f", num_induc={self.num_induc}"
			if self.degree:
				value += f", degree={self.degree}"
		return value


class NormLayer(Layer):

	def __init__(self):
		Layer.__init__(self)

	def forward(self, x_mean: Tensor, x_var: Optional[Tensor] = None) -> tuple[Tensor, Union[Tensor, None]]:
		return x_mean.tanh(), None if x_var is None else x_var.tanh()


