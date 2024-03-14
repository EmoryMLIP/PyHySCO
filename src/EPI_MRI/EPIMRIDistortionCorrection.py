import torch
from EPI_MRI.ImageModels import *
from EPI_MRI.InitializationMethods import *
from EPI_MRI.Preconditioners import *


class EPIMRIDistortionCorrection:
	""" Objective function and Jacobian correction for EPI-MRI.

	Defines and implements a distortion correction model and optimization problem for
	Echo-Planar MRI susceptibility artifact distortion correction.

	Find b that minimizes:

	J(b) = D(I1(b), I2(b)) + alpha S(b) + beta P(b) + rho Q(b)

	* D gives distance between corrected images I1(b) and I2(b)

	* S penalizes non-smooth field maps

	* P penalizes field maps violating intensity modulation constraint

	* Q is a proximal term used on some optimization schemes

	Parameters
	----------
	data_obj : `DataObject`
		object containing interpolation models for input images along with domain size and details
	alpha : float
		regularization parameter for the smoothness regularizer
	beta : float
		regularization parameter for the intensity regularizer
	averaging_operator : Class (subclass of :obj:`LinearOperators.LinearOperator`), optional
		class to use for the averaging operator (default is `LinearOperators.myAvg1D`)
	derivative_operator : Class (subclass of `LinearOperators.LinearOperator`), optional
		class to use for the derivative operator (default is `LinearOperators.myDiff1D`)
	regularizer : Class (subclass of `.LinearOperators.LinearOperator`), optional
		class to use for the regularizer (default is `LinearOperators.myLaplacian3D`)
	rho : float, optional
		augmentation parameter for proximal term (default is 0.0)
	initialization : Class (subclass of `InitializationMethods.InitializationMethod`), optional
		class to use for the initialization of the field map (default is `InitializationMethods.InitializeOT`)
	PC : Class (subclass of `Preconditioners.Preconditioner`), optional
		preconditioner (default is `Preconditioners.JacobiCG`)

	Attributes
	----------
	dataObj : `DataObject`
		object containing interpolation models for input images along with domain size and details
	A : `LinearOperators.LinearOperator`
		averaging operator
	D : `LinearOperators.LinearOperator`
		partial derivative operator in the phase encoding dimension
	xc : torch.Tensor (size prod(m))
		cell-centered grid in the phase encoding dimension
	S : `Regularizers.QuadRegularizer`
		defines the smoothness regularizer
	Q : `Regularizers.TikRegularizer`
		defines the proximal term, if used
	PC : `Preconditioners.Preconditioner`
		preconditioner object, if used
	initialization : `InitializationMethods.InitializationMethod`
		initialization object
	alpha : float
		regularization parameter for the smoothness regularizer
	beta : float
		regularization parameter for the intensity regularizer
	rho : float
		parameter for the proximal term augmentation parameter, if used
	device : string
		device on which to compute operations
	dtype : torch.dtype
		data type for all data tensors
	Dc : float
		most recent data fit term value
	Sc : float
		most recent smoothness regularization term value
	Pc : float
		most recent intensity regularization term value
	Qc : float
		most recent proximal term value, if used
	corr1 : torch.Tensor (size m)
		most recent corrected image from dataObj.I1
	corr2 : torch.Tensor (size m)
		most recent corrected image from dataObj.I2

	"""
	def __init__(self, data_obj, alpha, beta, averaging_operator=myAvg1D, derivative_operator=myDiff1D, regularizer=myLaplacian3D, rho=0.0, initialization=InitializeOT, PC=JacobiCG):
		self.dataObj = data_obj
		self.device = data_obj.device
		self.dtype = data_obj.dtype
		self.A = averaging_operator(self.dataObj.omega,self.dataObj.m, self.dtype, self.device)
		self.D = derivative_operator(self.dataObj.omega, self.dataObj.m, self.dtype, self.device)
		self.xc = get_cell_centered_grid(self.dataObj.omega, self.dataObj.m, device=self.device, dtype=self.dtype).reshape(tuple(self.dataObj.m))
		self.S = QuadRegularizer(regularizer(self.dataObj.omega,self.dataObj.m, self.dtype, self.device))
		self.Q = TikRegularizer(self.dataObj.omega, self.dataObj.m)
		self.alpha = alpha
		self.beta = beta
		self.rho = rho
		self.initialization = initialization()
		self.Dc = None
		self.Sc = None
		self.Pc = None
		self.Qc = None
		self.corr1 = None
		self.corr2 = None
		if PC is not None:
			self.PC = PC(self.dataObj)
		else:
			self.PC = None

	def eval(self, yc, yref=None, do_derivative=False, calc_hessian=False):
		"""
		Evaluates the objective function given the field map yc.

		J(yc) = D(I1(yc),I2(yc)) + alpha S(yc) + beta P(yc) + rho Q(yc, yref)

		If do_derivative is True, returns the gradient as well.

		If calc_hessian is True, returns the Hessian and PC as well.

		Parameters
		----------
		yc : torch.Tensor (size m_plus(m))
			a field inhomogeneity map
		yref : torch.Tensor (size m_plus(m)), optional
			reference image used in proximal term, default is None
		do_derivative : boolean, optional
			flag to compute and return the gradient, default is False
		calc_hessian : boolean, optional
			flag to construct and return Hessian mat-vec, default is False

		Returns
		-------
		J(yc) : torch.Tensor (size 1)
			objective function value
		dJ(yc) : torch.Tensor (size m_plus(m))
			gradient of the objective function, only returned when do_derivative=True
		H : Callable
			callable matrix-vector product with (approximate) Hessian, only returned when calc_hessian=True
		PC : Callable
			callable solver to apply preconditioner, only returned when calc_hessian=True
		"""
		hd = torch.prod(self.dataObj.h)
		
		# compute interpolated image and derivative
		T1c, Jac1, FI1, dFI1 = self.mp_transform(self.dataObj.I1, yc, do_derivative=do_derivative)
		T2c, Jac2, FI2, dFI2 = self.mp_transform(self.dataObj.I2, -yc, do_derivative=do_derivative)

		# compute distance measure
		Dc, dDc = self.distance(T1c, T2c)
		dD = None
		if do_derivative:
			geom = FI1 + FI2
			intensity = Jac1 * dFI1 + Jac2 * dFI2
			dD = self.D.transp_mat_mul(dDc*geom) + self.A.transp_mat_mul(dDc*intensity)
		else:
			geom = None
			intensity = None

		# smoothness regularizer
		Sc, dS, d2S = self.S.eval(yc, do_derivative=do_derivative)

		# intensity regularizer
		G, dG, d2G = self.phi_EPI(Jac1-1, do_derivative=do_derivative, calc_hessian=calc_hessian)
		Pc = torch.sum(G)
		dP = None
		if do_derivative:
			dP = self.D.transp_mat_mul(dG)

		# compute proximal term
		if self.rho >0:
			Qc,dQ,d2Q = self.Q.eval(yc, 1.0, yref, do_derivative=do_derivative)
		else:
			Qc = 0.0
			dQ = 0.0
			d2Q = None

		# save terms of objective function and corrected images
		self.Dc = Dc
		self.Sc = Sc
		self.Pc = Pc
		self.Qc = Qc
		self.corr1 = T1c
		self.corr2 = T2c

		Jc = Dc + hd*self.alpha*Sc + hd*self.beta*Pc + self.rho*Qc
		if not do_derivative:
			return Jc
		dJ = dD + hd*self.alpha*dS + hd*self.beta*dP + self.rho*dQ
		if not calc_hessian:
			return Jc, dJ
		else:
			def H(x):
				""" Matrix-vector product between Hessian and a tensor x of size m_plus(m). """
				Dx = self.D.mat_mul(x)
				dr = geom*Dx + intensity*self.A.mat_mul(x)
				dr_d2psi = dr * hd
				if self.beta == 0:  # d2P is zeros
					d2D = self.D.transp_mat_mul(dr_d2psi*geom) + self.A.transp_mat_mul(dr_d2psi*intensity)				
					return d2D + hd*self.alpha*d2S.mat_mul(x) + hd*self.rho*x
				else:
					d2D = self.D.transp_mat_mul(dr_d2psi*geom + hd*self.beta*d2G*Dx) + self.A.transp_mat_mul(dr_d2psi*intensity)
					return d2D + hd*self.alpha*d2S.mat_mul(x) + hd*self.rho*x
			if self.PC is not None:
				diagD, diagP, diagS = self.PC.getM(geom, intensity,hd, d2G, self.D, self.A, self.S.H, self.alpha, self.beta)
				self.PC.M += hd*self.rho
				M = lambda x: self.PC.eval(x)
			else:
				M = None

			return Jc, dJ, H, M

	def mp_transform(self, I, b, do_derivative=False):
		"""
		Applies the distortion correction model.

		TI(xc) = I(xc + bc) * (1 + dbc)

		If do_derivative is True, computes gradient information as well.

		Parameters
		----------
		I : ImageModels.ImageModel
			interpolating image model
		b : torch.Tensor (size m_plus(m))
			a field inhomogeneity map
		do_derivative : boolean, optional
			flag to compute and return the gradient, default is False

		Returns
		----------
		TI : torch.Tensor (size m)
			result of applying the correction model to image I using the field map b
		Jac : torch.Tensor (size m)
			mass preserving factor 1 + dbc
		FI : torch.Tensor (size m)
			result of I interpolated on xc + bc
		dFI : torch.Tensor (size m)
			derivative of applying image model interpolation, None when do_derivative=False
		"""
		bc = self.A.mat_mul(b)  # averaging matrix & translation vector
		dbc = self.D.mat_mul(b)  # derivative matrix & derivative of deformation vector
		Jac = 1+dbc  # determinant of the transform xc+bc
		xt = self.xc + bc

		if do_derivative:
			FI, dFI = I.eval(xt, do_derivative=True)  # interpolation on deformed grid
			TI = FI * Jac  # mass preserving factor

		else:
			FI = I.eval(xt)  # interpolation on deformed grid
			TI = FI * Jac  # mass preserving factor
			dFI = None

		return TI, Jac, FI, dFI

	def phi_EPI(self, x, do_derivative=False, calc_hessian=False):
		"""
		Barrier function for the intensity regularization term, applied element-wise.

		phi(x) = -x^4 / ((x^2 - 1))

		phi satisfies these important conditions:

			* phi(x) > 0, for all x

			* phi(|x| -> 1) -> infinity

			* phi(0) = 0

			* phi is convex

			* phi(x) = phi(-x)

		Parameters
		----------
		x : torch.Tensor (size m)
			partial derivative of the field map
		do_derivative : boolean, optional
			flag to compute the first derivative (default is False)
		calc_hessian : boolean, optional
			flag to compute the second derivative (default is False)

		Returns
		----------
		G : torch.Tensor (size m)
			function value
		dG : torch.Tensor (size m)
			first derivative of the function, None if do_derivative=False
		d2G : torch.Tensor (size m)
			second derivative of the function, None if calc_hessian=False
		"""
		dG, d2G = None, None
		# penalize values outside of (-1,1)
		x[torch.abs(x) >= 1] = float('inf')
		x2 = x*x
		G = torch.nan_to_num(-(x2*x2) / (x2-1))
		if do_derivative:
			dG = torch.nan_to_num(-2*(x*x2)*(x2-2) / (x2-1)**2)
		if calc_hessian:
			d2G = torch.nan_to_num(-2 * x2 * (x2 * x2 - 3 * x2 + 6) / (x2 - 1) ** 3)
		return G, dG, d2G

	def initialize(self, *args, **kwargs):
		"""
		Calls the initialization scheme.

		Parameters
		----------
		args, kwargs : Any
			arguments and keyword arguments as needed for the initialization scheme

		Returns
		----------
		B0 : torch.Tensor (size m_plus(m))
			initial guess for the field map
		"""
		return self.initialization.eval(self.dataObj, *args, **kwargs)

	def distance(self, Tc, Rc):
		"""
		Computes the sum of squared difference metric and derivatives for two images.

		Parameters
		----------
		Tc : torch.Tensor (size m)
			first image
		Rc : torch.Tensor (size m)
			second image

		Returns
		----------
		Dc : torch.Tensor (size 1)
			distance value = 1/2 * hd * rc.T * rc
		dD : torch.Tensor (size m)
			distance derivative = hd * dr
		"""
		d2psi = torch.prod(self.dataObj.h)
		rc = Tc - Rc
		Dc = 0.5 * d2psi * torch.norm(rc)**2
		dr = 1
		dD = rc*d2psi*dr
		return Dc, dD


class DataObject:
	"""
	Defines an object to store input images, interpolation models, and domain information.

	Attributes
	----------
	I1 : ImageModels.ImageModel
		interpolating image model for the first input image; has the opposite phase encoding direction as I2
	I2 : ImageModels.ImageModel
		interpolating image model for the second input image; has the opposite phase encoding direction as I1
	omega : torch.Tensor (size # of dimensions x 2)
		image domain
	m : torch.Tensor (size # of dimensions)
		discretization size
	h : torch.Tensor (size # of dimensions)
		cell size
	p : list (size # of dimensions)
		order to permute dimensions to return the image to input orientation
	im1 : torch.Tensor
		original image 1 without normalization
	im2 : torch.Tensor
		original image 2 without normalization
	device : string
		device on which to compute operations
	dtype : torch.dtype
		data type for all data tensors

	Parameters
	----------
	img1 : str
		file path of [first] input image or stacked images if in the same file
	img2: str, optional
		file path of the second input image if images are separate files (default is None)
	phase_encoding_direction: int, optional
		gives which dimension of img1 and img2 is the phase encoding dimension (i.e. 1 for the first, 2 for the second, etc.) (default is 1)
	image_model : Class (subclass of ImagesModels.ImageModel), optional
		class to use for the image interpolation model (default is ImageModels.Interp1D)
	do_normalize : boolean, optional
		flag to normalize image intensities (default is True)
	device : string, optional
		device on which to compute operations (default is 'cpu')
	dtype : torch.dtype, optional
		data type for all data tensors (default is torch.float64)
	"""
	def __init__(self, img1, img2=None, phase_encoding_direction=1, image_model=Interp1D, do_normalize=True, dtype=torch.float64, device='cpu'):
		self.device = device
		self.dtype = dtype
		rho0, rho1, self.omega, self.m, self.h, self.p = load_data(img1, im2=img2, phase_encoding_direction=phase_encoding_direction, device=self.device, dtype=self.dtype)
		self.im1 = rho0
		self.im2 = rho1
		if do_normalize:
			rho0, rho1 = normalize(rho0, rho1)
		self.I1 = image_model(rho0, self.omega, self.m, dtype=self.dtype, device=self.device)
		self.I2 = image_model(rho1, self.omega, self.m, dtype=self.dtype, device=self.device)
