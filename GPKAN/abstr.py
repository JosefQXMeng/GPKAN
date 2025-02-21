
from abc import ABC, abstractmethod
import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import functional as F



class Layer(Module, ABC):

	def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None):
		Module.__init__(self)

		self.in_dim = in_dim  # Q
		self.out_dim = out_dim  # D
	
	@abstractmethod
	def forward(self):
		...


class Network(Module, ABC):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
	):
		Module.__init__(self)

		self.in_dim = in_dim  # D^0
		self.out_dim = out_dim  # D^L
		if hidden_dims is None:
			hidden_dims = []
		elif isinstance(hidden_dims, int):
			hidden_dims = [hidden_dims]
		# [D^0,...,D^L]
		self.dims = [in_dim] + hidden_dims + [out_dim]

	@abstractmethod
	def forward(self):
		...


class Regr(ABC):

	def __init__(self, out_dim: int, min_noise_var: float = 1e-6):
		self._noise_var = Parameter(torch.zeros([out_dim]))
		self.min_noise_var = min_noise_var

	@property
	def noise_var(self) -> Tensor:
		return F.softplus(self._noise_var) + self.min_noise_var

	def mll(self, f_mean: Tensor, f_var: Tensor, y: Tensor) -> Tensor:
		var = f_var.add(self.noise_var)
		return y.sub(f_mean).pow(2).div(var).add(var.log()).add(math.log(2*math.pi)).div(-2)

	def ell(self, f_mean: Tensor, f_var: Tensor, y: Tensor) -> Tensor:
		noise = self.noise_var
		return y.sub(f_mean).pow(2).add(f_var).div(noise).add(noise.log()).add(math.log(2*math.pi)).div(-2)

	@abstractmethod
	def loglikelihood(self):
		...

	@abstractmethod
	def pred(self):
		...


