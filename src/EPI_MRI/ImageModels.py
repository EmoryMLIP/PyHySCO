import torch
from abc import ABC, abstractmethod


class ImageModel(ABC):
	"""
	Defines the structure of an image model object used to offer interpolation on an image.

	Attributes
	----------
	data : torch.Tensor (size m)
		Original image data.
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	h : torch.Tensor (size # of dimensions)
		Cell size.
	device : str
		Device on which to compute operations.
	dtype : torch.dtype
		Data type for all data tensors.

	Parameters
	----------
	data : torch.Tensor (size m)
		Original image data.
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	device : str, optional
		Device on which to compute operations (default is 'cpu').
	dtype : torch.dtype, optional
		Data type for all data tensors (default is torch.float64).
	"""
	def __init__(self, data, omega, m, device='cpu', dtype=torch.float64):
		self.data = data
		self.omega = omega
		self.m = m
		self.dtype = dtype
		self.device = device
		self.h = (omega[1::2]-omega[:omega.shape[0]-1:2])/m

	@abstractmethod
	def eval(self, x, do_derivative=False):
		"""
		Evaluates the image interpolation on distorted grid x; moves data only in the distortion dimension.

		Parameters
		----------
		x : torch.Tensor (size m)
			Points on which to interpolate.
		do_derivative : bool, optional
			Flag to calculate and return the derivative of interpolation (default is False).

		Returns
		----------
		Tc : torch.Tensor (size m)
			Image interpolated on points x.
		dT : torch.Tensor (size m)
			Derivative of interpolation, only returned when do_derivative=True.
		"""
		pass


class Interp1D(ImageModel):
	"""
	Defines a model to efficiently interpolate an image at different points.

	Attributes
	----------
	data : torch.Tensor (size m)
		Original image data.
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	h : torch.Tensor (size # of dimensions)
		Cell size.
	device : str
		Device on which to compute operations.
	dtype : torch.dtype
		Data type for all data tensors.

	Parameters
	----------
	data : torch.Tensor (size m)
		Original image data.
	omega : torch.Tensor (size # of dimensions x 2)
		Image domain.
	m : torch.Tensor (size # of dimensions)
		Discretization size.
	device : str, optional
		Device on which to compute operations (default is 'cpu').
	dtype : torch.dtype, optional
		Data type for all data tensors (default is torch.float64).
	"""

	def __init__(self, data, omega, m, dtype=torch.float64, device='cpu'):
		super().__init__(data, omega, m, dtype=dtype, device=device)
		i0 = (torch.arange(0, torch.prod(m[0:-1]).item(), device=device).view(-1, 1) * (m[-1]*torch.ones(1, m[-1], device=device))).contiguous().view(-1)
		self.i0 = i0.type(torch.long)

	def eval(self, x, do_derivative=False):
		"""
		Evaluates interpolation of data on points x; only in the distortion dimension.

		Parameters
		----------
		x : torch.Tensor (size m)
			Points on which to interpolate.
		do_derivative : bool, optional
			Flag to calculate and return the derivative of interpolation (default is False).

		Returns
		----------
		Tc : torch.Tensor (size m)
			Image interpolated on points x.
		dT : torch.Tensor (size m)
			Derivative of interpolation, only returned when do_derivative=True.
		"""
		dim = int(self.omega.shape[0] / 2)
		if dim < 2 or dim > 4:
			print("dimension = 2 or 3 or 4 only supported dimensions")
			return
		T = self.data

		x = (x - self.omega[-2]) / self.h[-1] - 0.5
		x = torch.clamp(x, 0, self.m[-1]-1)
		shape_x = x.shape
		x = x.contiguous().view(-1)
		p = torch.floor(x)
		x = x - p
		p = p.type(torch.long)

		i0 = self.i0
		pl = i0+p  # left neighbor
		pr = i0+torch.clamp(p+1, 0, self.m[-1]-1)  # right neighbor
		TD = T.contiguous().view(-1)
		Tx = TD[pl] * (1 - x) + TD[pr] * x
		if do_derivative:
			dT = (TD[pr]-TD[pl])/self.h[-1]
			return Tx.view(shape_x), dT.view(shape_x)
		else:
			return Tx.view(shape_x)
