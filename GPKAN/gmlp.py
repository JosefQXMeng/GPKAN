
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleDict, SiLU
from torch.nn import functional as F

from .abstr import Network, Regr



class MLP(Network):
	
	def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int], activ: Optional[Module] = None):
		Network.__init__(self, in_dim, out_dim, hidden_dims)

		layers = {}
		for i in range(1, len(self.dims)):
			layers[f"fc_{i}"] = Linear(self.dims[i-1], self.dims[i])
			if i+1 < len(self.dims):
				layers[f"activ_{i}"] = SiLU() if activ is None else activ
		self.layers = ModuleDict(layers)
	
	def forward(self, x: Tensor) -> Tensor:
		for layer in self.layers.values():
			x = layer(x)
		return x


class GaussianMLP(MLP):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: list[int],
		activ: Optional[Module] = None,
	):
		MLP.__init__(self, in_dim, out_dim*2, hidden_dims, activ)

	def forward(self, x) -> tuple[Tensor, Tensor]:
		f_mean, f_var = MLP.forward(self, x).tensor_split(2, dim=-1)
		f_var = F.softplus(f_var) + 1e-6
		return f_mean, f_var


class GaussianMLPR(GaussianMLP, Regr):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: list[int],
		activ: Optional[Module] = None,
		min_noise_var: float = 1e-6,
	):
		GaussianMLP.__init__(self, in_dim, out_dim, hidden_dims, activ)
		Regr.__init__(self, out_dim, min_noise_var)
	
	def loglikelihood(self, x: Tensor, y: Tensor) -> Tensor:
		f_mean, f_var = self.forward(x)
		return self.ell(f_mean, f_var, y).sum(-1).mean()
	
	def pred(self, x: Tensor, y: Optional[Tensor] = None) -> Any:
		with torch.no_grad():
			f_mean, f_var = self.forward(x)
			mll = None if f_var is None or y is None else self.mll(f_mean, f_var, y)
			return f_mean, mll


