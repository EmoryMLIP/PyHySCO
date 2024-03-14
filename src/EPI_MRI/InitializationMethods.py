from EPI_MRI.utils import *
from abc import ABC, abstractmethod
from EPI_MRI.LinearOperators import FFT3D, getLaplacianStencil


class InitializationMethod(ABC):
    """
    Defines initialization method to be used to initialize field map estimate.

    All children must implement an initialization evaluation method.
    """
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, data, *args, **kwargs):
        """
        Evaluates initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        args, kwargs : Any
            Particular arguments and keyword arguments for initialization method.

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial guess for field map.
        """
        pass


class InitializeOT(InitializationMethod):
    """
    Defines parallelized one-dimensional optimal transport based initialization scheme.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, blur_result=True, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        blur_result : boolean, optional
            Flag to apply Gaussian blur to `init_OT` result before returning (default is True).
        args, kwargs : Any
            Provided shift, if given (see method `init_OT`).

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        if blur_result:
            return self.blur(self.init_OT(data, *args, **kwargs).reshape(list(m_plus(data.m))), data.omega, data.m)
        else:
            return self.init_OT(data, *args, **kwargs)

    def init_OT(self, data, shift=2):
        """
        Optimal Transport based initialization scheme.

        Performs parallel 1-D optimal transport in distortion dimension to estimate field map.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        shift : float, optional
            Numeric shift to ensure smoothness of positive measure.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        device = data.device
        dtype = data.dtype

        if torch.numel(data.m) == 3:
            rho0 = data.I1.data.reshape(-1, data.m[2])
            rho1 = data.I2.data.reshape(-1, data.m[2])
        elif torch.numel(data.m) == 4:
            rho0 = data.I1.data.reshape(-1, data.m[3])
            rho1 = data.I2.data.reshape(-1, data.m[3])
        else:
            rho0 = data.I1.data
            rho1 = data.I2.data

        rho0new = torch.empty(rho0.shape, device=device, dtype=dtype)
        rho1new = torch.empty(rho1.shape, device=device, dtype=dtype)
        rho0new = torch.add(rho0, shift, out=rho0new)
        rho1new = torch.add(rho1, shift, out=rho1new)

        rho0new = torch.div(rho0new, torch.sum(rho0new, dim=1, keepdim=True), out=rho0new)
        rho1new = torch.div(rho1new, torch.sum(rho1new, dim=1, keepdim=True), out=rho1new)

        C0 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho0new, dim=1)),
            dim=1)
        C1 = torch.cat(
            (torch.zeros((rho0new.shape[0], 1), dtype=dtype, device=device), torch.cumsum(rho1new, dim=1)),
            dim=1)
        C0[:, -1] = torch.ones_like(C0[:, 1])
        C1[:, -1] = torch.ones_like(C1[:, 1])

        t = torch.linspace(0, 1, int(data.m[-1] + 1), dtype=dtype, device=device).view(1, -1).expand(
            int(torch.prod(data.m) / data.m[-1]), -1)

        # interpolations

        iC0 = interp_parallel(C0, t, t, device=device)
        iC1 = interp_parallel(C1, t, t, device=device)

        iChf = torch.empty(iC0.shape, device=device)
        iChf = torch.div(torch.add(iC0, iC1, out=iChf), 2, out=iChf)

        T0hf = interp_parallel(t, iChf, C0, device=device)
        T1hf = interp_parallel(t, iChf, C1, device=device)

        T0hf = interp_parallel(T0hf, t, t, device=device)  # invert the mapping
        T1hf = interp_parallel(T1hf, t, t, device=device)  # invert the mapping

        T0hf = (data.omega[-2] - data.omega[-1]) * (T0hf - t)
        T1hf = (data.omega[-2] - data.omega[-1]) * (T1hf - t)

        Bc = torch.reshape(0.5 * (T0hf - T1hf), list(m_plus(data.m)))

        return -1 * Bc

    def blur(self, input, omega, m, alpha=1.0):
        """
        Performs Gaussian blur to pre-smooth initial field map.

        Parameters
        ----------
        input : torch.Tensor (size m_plus(m))
            Field map from `init_OT`.
        omega : torch.Tensor
            Image domain.
        m : torch.Tensor
            Image size.
        alpha : float, optional
            Standard deviation of Gaussian kernel (default is 1.0).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map with blur.
        """
        L,_,_,_ = getLaplacianStencil(omega, m, input.dtype, input.device)
        K = FFT3D(L, m)
        return K.inv(input,1/alpha)


class InitializeRandom(InitializationMethod):
    """
    Defines random initialization scheme.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        args, kwargs : Any
            Provided seed, if given (see `rand_init`).

        Returns
        ----------
        B0 : torch.Tensor (size (m_plus(m),1))
            Initial field map.
        """
        return self.rand_init(data, *args, **kwargs)

    def rand_init(self, data, seed=None):
        """
        Random initialization scheme.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        seed : int, optional
            Seed for torch.random (for reproducibility) (default is None).

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(list(m_plus(data.m)), device=data.device, dtype=data.dtype)


class InitializeZeros(InitializationMethod):
    """
    Defines zero initialization scheme.
    """
    def __init__(self):
        super().__init__()

    def eval(self, data, *args, **kwargs):
        """
        Call initialization.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.
        args, kwargs : Any
            None for this initialization scheme.

        Returns
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial field map.
        """
        return self.zero_init(data)

    def zero_init(self, data):
        """
        Zeros initialization scheme.

        Parameters
        ----------
        data : `EPIMRIDistortionCorrection.DataObject`
            Original image data.

        Returns
        ----------
        Bc : torch.Tensor (size m_plus(m))
            Initial guess for field inhomogeneity map.
        """
        return torch.zeros(list(m_plus(data.m)), device=data.device, dtype=data.dtype)
