
import math
from typing import Optional

import torch
from torch import Tensor
from torch.special import erf



def StandardNormalCDF(x: Tensor) -> Tensor:
	return erf(x.div(math.sqrt(2))).add(1).div(2)


def MaternExpectation(
	nu: float, l: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None,
) -> Tensor:
	assert nu in [0.5, 1.5, 2.5]
	mu = torch.zeros([]) if mu is None else mu
	if sigma_sq is None:
		x_abs = mu.div(l).mul(math.sqrt(nu*2)).abs()
		Exp = x_abs.mul(-1).exp()
		P = torch.ones([])
		if nu > 1:
			P = x_abs.add(P)
		if nu > 2:
			P = x_abs.pow(2).div(3).add(P)
		return P.mul(Exp)
	else:
		mean = mu.div(l).mul(math.sqrt(nu*2))
		var = sigma_sq.div(l.pow(2)).mul(nu*2)
		std = var.sqrt()
		E1 = var.div(2).add(mean).exp().mul(StandardNormalCDF(std.mul(-1).sub(mean.div(std))))
		E2 = var.div(2).sub(mean).exp().mul(StandardNormalCDF(std.mul(-1).add(mean.div(std))))
		Exp = torch.zeros([])
		coef1 = torch.ones([])
		coef2 = torch.ones([])
		if nu > 1:
			comp_exp = mean.pow(2).div(var).div(-2).exp().mul(std).div(math.sqrt(math.pi/2))
			Exp = comp_exp.add(Exp)
			comp1 = var.add(mean)
			comp2 = var.sub(mean)
			coef1 = comp1.mul(-1).add(coef1)
			coef2 = comp2.mul(-1).add(coef2)
		if nu > 2:
			Exp = comp_exp.mul(var).mul(-1).div(3).add(Exp)
			coef1 = comp1.pow(2).add(var).div(3).add(coef1)
			coef2 = comp2.pow(2).add(var).div(3).add(coef2)
		return Exp.add(E1.mul(coef1)).add(E2.mul(coef2))


def SquaredExponentialExpectation(
	l_sq: Tensor, mu: Optional[Tensor] = None, sigma_sq: Optional[Tensor] = None,
) -> Tensor:
	var_sum = l_sq if sigma_sq is None else l_sq.add(sigma_sq)
	std_quot = torch.ones([]) if sigma_sq is None else l_sq.div(var_sum).sqrt()
	exp = torch.ones([]) if mu is None else mu.pow(2).div(var_sum).div(-2).exp()
	return std_quot.mul(exp)


