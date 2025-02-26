
from typing import Any, Optional, Union

from torch import Tensor
from torch.nn import ModuleDict

from .abstr import Network, Regr
from .gmlp import GaussianMLPR
from .layers import FCLayer, NormLayer



class GPKAN(Network):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
		num_induc: Union[int, list[int]],
		kernel : list[Optional[str]] = "SquaredExponentialKernel",
		mean_func: list[Optional[str]] = None,
		num_comp: int = 0,
		degree: int = 0,
		num_basis: int = 0,
	):
		Network.__init__(self, in_dim, out_dim, hidden_dims)

		if isinstance(num_induc, int):
			num_induc = [num_induc] * (len(self.dims)-1)
		assert len(num_induc) == len(self.dims)-1

		if mean_func is None or isinstance(mean_func, str):
			mean_func = [mean_func] * (len(self.dims)-1)
		assert len(mean_func) == len(self.dims)-1

		if kernel is None or isinstance(kernel, str):
			kernel = [kernel] * (len(self.dims)-1)
		assert len(kernel) == len(self.dims)-1

		layerdict = {}
		for i in range(len(self.dims)-1):
			layerdict[f"norm_{i+1}"] = NormLayer()
			layerdict[f"fc_{i+1}"] = FCLayer(
				self.dims[i], self.dims[i+1], num_induc[i], kernel[i], mean_func[i], num_comp, degree, num_basis,
			)
		self.layers = ModuleDict(layerdict)

	def forward(self, x: Tensor) -> tuple[Tensor, Union[Tensor, None]]:
		"""
		x ~ [B, D^0]
		->
		f_mean & f_var ~ [B, D^L]
		"""
		f_mean = x
		f_var = None
		for layer in self.layers.values():
			f_mean, f_var = layer.forward(f_mean, f_var)
		return f_mean, f_var
	
	def add_degree(self, num: int) -> None:
		for layer in self.layers.values():
			if isinstance(layer, FCLayer):
				layer.add_degree(num)

	def add_basis(self, num: int) -> None:
		for layer in self.layers.values():
			if isinstance(layer, FCLayer):
				layer.add_basis(num)
	
	def add_comp(self, num: int) -> None:
		for layer in self.layers.values():
			if isinstance(layer, FCLayer):
				layer.add_comp(num)


class GPKANR(GPKAN, Regr):

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		hidden_dims: Optional[Union[int, list[int]]],
		num_induc: Union[int, list[int]],
		kernel : list[Optional[str]] = "SquaredExponentialKernel",
		mean_func: list[Optional[str]] = None,
		num_comp: int = 0,
		degree: int = 0,
		num_basis: int = 0,
		min_noise_var: float = 1e-6,
	):
		GPKAN.__init__(self, in_dim, out_dim, hidden_dims, num_induc, kernel, mean_func, num_comp, degree, num_basis)
		Regr.__init__(self, out_dim, min_noise_var)

	def loglikelihood(self, x: Tensor, y: Tensor) -> Tensor:
		return GaussianMLPR.loglikelihood(self, x, y)
	
	def pred(self, x: Tensor, y: Optional[Tensor] = None) -> Any:
		return GaussianMLPR.pred(self, x, y)
	
	def loss(self, x: Tensor, y: Tensor) -> Tensor:
		"""
		negative expected log-likelihood if f_var else mean squared error
		"""
		f_mean, f_var = self.forward(x)
		if f_var is None:
			return y.sub(f_mean).pow(2).sum(-1).mean()
		else:
			return self.ell(f_mean, f_var, y).sum(-1).mean().mul(-1)


