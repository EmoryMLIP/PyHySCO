import torch
from EPI_MRI.utils import m_plus
from abc import ABC, abstractmethod


class LinearOperator(ABC):
    """
    Abstract class to define operations of LinearOperator subclasses. A LinearOperator object must define methods for
    multiplication and transpose multiplication.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def mat_mul(self, x):
        pass

    @abstractmethod
    def transp_mat_mul(self, x):
        pass


class Conv1D(LinearOperator):
    """
    Defines a one-dimensional convolution, batched over the last dimension.

    Attributes
    ----------
    kernel : torch.Tensor
        convolution kernel
    m : torch.Tensor
        image size
    padding : int
        size of padding to use in convolution

    Parameters
    ---------
    kernel : torch.Tensor
        convolution kernel; first two dimension should be of size 1
    m : torch.Tensor
        image size
    padding : int, optional
        size of padding to use in convolution (default is 0)
    """
    def __init__(self, kernel, m, padding=0):
        # assert that kernel is a 3D torch tensor whose first two dimensions are 1
        assert kernel.shape[0] == 1 and kernel.shape[1] == 1
        self.kernel = kernel
        self.m = m
        self.padding = padding

    def mat_mul(self, x):
        """
        Performs 1D convolution using kernel on input x, batched in last dimension.

        Parameters
        ----------
        x : torch.Tensor (shape m_plus(m))
            input tensor

        Returns
        --------
        Ax : torch.Tensor
            result of convolution (shape m)
        """
        x_shape = x.shape
        x = x.contiguous().view(-1, 1, self.m[-1] + 1)
        Ax = torch.nn.functional.conv1d(x, self.kernel, padding=self.padding)
        return Ax.view(x_shape[:-1] + (-1,))

    def transp_mat_mul(self, x):
        """
        Performs 1D transpose convolution using kernel on input x, batched in last dimension.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        --------
        Atx : torch.Tensor
            result of convolution
        """
        x_shape = x.shape
        x = x.contiguous().view(torch.prod(self.m[:-1]).item(), 1, -1)
        Atx = torch.nn.functional.conv_transpose1d(x, self.kernel, padding=self.padding)
        return Atx.view(x_shape[:-1] + (-1,))

    def op_mul(self, B):
        """
        Performs operator multiplication between this convolution operator and another 1D convolution operator.

        Parameters
        ----------
        B : `Conv1D`
            second convolution operator

        Returns
        --------
        AB : `Conv1D`
            object representing operation that is the result of operator multiplication
        """
        # assert that B is a Conv1D operator
        assert isinstance(B, Conv1D)
        # assert that the kernel sizes are the same
        assert self.kernel.shape == B.kernel.shape
        # assert that all elements in self.m are equal to the corresponding elements in B.m
        assert all([self.m[i] == B.m[i] for i in range(len(self.m))])

        return Conv1D(self.kernel * B.kernel, self.m)

    def diag(self):
        """
        Returns diagonal of this operator as a matrix.

        Returns
        --------
        D : torch.Tensor
            diagonal of this operator
        """
        return self.kernel[tuple(torch.div(torch.tensor(self.kernel.shape), 2, rounding_mode='floor'))] * torch.ones(
            tuple(m_plus(self.m)), device=self.kernel.device, dtype=self.kernel.dtype)


class FFT3D(LinearOperator):
    """
    Defines a three-dimensional convolution, implemented using the diagonalization K = Q^H D Q,
    where Q is a discrete Fourier transform matrix with periodic boundary conditions, and D is diagonal with
    the eigenvalues of K as its elements.

    Attributes
    ----------
    kernel : torch.Tensor
        convolution kernel
    m : torch.Tensor
        image size
    eig : torch.Tensor
        eigenvalues of kernel

    Parameters
    ---------
    kernel : torch.Tensor
        convolution kernel
    m : torch.Tensor
        image size
    """
    def __init__(self, kernel, m):
        self.m = m
        self.kernel = kernel.squeeze(0).squeeze(0)
        self.eig = self.compute_eigs()

    def compute_eigs(self):
        """
        Computes the eigenvalues of self.kernel

        Returns
        -------
        eig : torch.Tensor
            eigenvalues of self.kernel in fourier space
        """
        k = self.kernel.shape
        m = self.m

        # put K in the center of an n[0] x n[1] x n[2] grid
        j0 = torch.div(m[0], 2, rounding_mode='floor') - torch.div(k[0] - 1, 2, rounding_mode='floor')
        j1 = torch.div(m[1], 2, rounding_mode='floor') - torch.div(k[1] - 1, 2, rounding_mode='floor')
        j2 = torch.div(m[2] + 1, 2, rounding_mode='floor') - torch.div(k[2] - 1, 2, rounding_mode='floor')

        Bp = torch.zeros(m[0], m[1], m[2] + 1, device=self.kernel.device, dtype=self.kernel.dtype)
        Bp[j0:j0 + k[0], j1:j1 + k[1], j2 + 1:j2 + k[2] + 1] = self.kernel
        Bp = torch.fft.fftshift(Bp)
        Bh = torch.fft.fftn(Bp)
        # assert torch.norm(torch.imag(Bh))/torch.norm(Bh) < 1e-4
        return torch.abs(Bh)

    def mat_mul(self, x):
        """
        Performs 3D convolution using kernel on input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        --------
        Lx : torch.Tensor
            result of convolution
        """
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x) * self.eig))

    def transp_mat_mul(self, x):
        """
        Performs 3D transpose convolution using kernel on input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        --------
        Lx : torch.Tensor
            result of transpose convolution
        """
        return self.mat_mul(x)

    def inv(self, z, rho):
        """
        Applies inverse of self.kernel to input z, scaled by rho.

        Parameters
        ----------
        z : torch.Tensor
            input tensor
        rho : float
            scaling variable

        Returns
        --------
        Liz : torch.Tensor
            result of inverse
        """
        return torch.real(torch.fft.ifftn(torch.fft.fftn(z) / (self.eig + rho)))

    def diag(self):
        """
        Returns diagonal of this operator as a matrix.

        Returns
        --------
        D : torch.Tensor
            diagonal of this operator
        """
        return self.kernel[tuple(torch.div(torch.tensor(self.kernel.shape), 2, rounding_mode='floor'))] * torch.ones(
            tuple(m_plus(self.m)), device=self.kernel.device, dtype=self.kernel.dtype)


class Conv3D(LinearOperator):
    """
    Defines a three-dimensional convolution.

    Attributes
    ----------
    kernel : torch.Tensor
        convolution kernel
    m : torch.Tensor
        image size
    padding : int
        size of padding to use in convolution
    shape : list
        shape input tensor should be before applying convolution

    Parameters
    ---------
    kernel : torch.Tensor
        convolution kernel; first two dimension should be of size 1
    m : torch.Tensor
        image size
    padding : list, optional
        size of padding to use in convolution (default is [1, 1, 1])
    """
    def __init__(self, kernel, m, padding=[1, 1, 1]):
        # assert that kernel is a 3D torch tensor whose first two dimensions are 1
        assert kernel.shape[0] == 1 and kernel.shape[1] == 1

        self.kernel = kernel
        self.m = m
        self.padding = padding
        dim = torch.numel(self.m)
        shape = [1, 1, 1, 1, 1]
        shape[-dim:] = list(m_plus(self.m))
        if dim == 4:
            shape[0] = shape[1]
            shape[1] = 1
        self.shape = shape

    def mat_mul(self, x):
        """
        Performs 3D convolution using kernel on input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        --------
        Lx : torch.Tensor
            result of convolution
        """
        return torch.nn.functional.conv3d(x.reshape(self.shape), self.kernel,
                                          padding=self.padding).contiguous().squeeze()

    def transp_mat_mul(self, x):
        """
        Performs 3D transpose convolution using kernel on input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        --------
        Ltx : torch.Tensor
            result of transpose convolution
        """
        return torch.nn.functional.conv3d(x.reshape(self.shape), self.kernel,
                                          padding=self.padding).contiguous().squeeze()

    def diag(self):
        """
        Returns diagonal of this operator as a matrix.

        Returns
        --------
        D : torch.Tensor
            diagonal of this operator
        """
        return self.kernel[tuple(torch.div(torch.tensor(self.kernel.shape), 2, rounding_mode='floor'))] * torch.ones(
            tuple(m_plus(self.m)), device=self.kernel.device, dtype=self.kernel.dtype)


class Identity(LinearOperator):
    """
    Identity operator.

    Attributes
    ----------
    rho : float
        scalar multiplier

    Parameters
    ----------
    rho : float, optional
        scalar multiplier (default is 1.0)
    """
    def __init__(self, rho=1.0):
        super().__init__()
        self.rho = rho

    def mat_mul(self, x):
        """
        Returns (scaled) identity multiplied by input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        Ax : torch.Tensor
            self.rho * x
        """
        return self.rho * x

    def transp_mat_mul(self, x):
        """
        Returns (scaled) identity multiplied by input x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        Atx : torch.Tensor
            self.rho * x
        """
        return self.rho * x

    def inv(self, z, shift):
        """
        Returns (scaled) inverse identity multiplied by input z.

        Parameters
        ----------
        z : torch.Tensor
            input tensor
        shift : float
            scalar shift

        Returns
        -------
        Liz : torch.Tensor
            z / (self.rho + shift)
        """
        return z / (self.rho + shift)


def myAvg1D(omega, m, dtype, device):
    """
    Builds and returns averaging operator as `Conv1D` object.

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    dtype : torch.dtype
        data type
    device : String
        compute device

    Returns
    -------
    A : `Conv1D`
        averaging operator object
    """
    kernel = torch.tensor([0.5, 0.5], dtype=dtype, device=device).reshape(1, 1, -1)
    return Conv1D(kernel, m)


def myDiff1D(omega, m, dtype, device):
    """
    Builds and returns derivative operator as `Conv1D` object.

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    dtype : torch.dtype
        data type
    device : String
        compute device

    Returns
    -------
    D : `Conv1D`
        derivative operator object
    """
    h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
    return Conv1D(torch.tensor([-1 / h[-1], 1 / h[-1]], dtype=dtype, device=device).reshape(1, 1, -1), m)


def myLaplacian1D(omega, m, dtype, device):
    """
    Builds and returns 1D Laplacian as `Conv1D` object.

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    dtype : torch.dtype
        data type
    device : String
        compute device

    Returns
    -------
    L : `Conv1D`
        1D Laplacian operator object
    """
    h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
    return Conv1D(torch.tensor([-1 / (h[-1] ** 2), 2 / (h[-1] ** 2), -1 / (h[-1] ** 2)], dtype=dtype, device=device).reshape(1, 1,-1), m, padding=1)


def getLaplacianStencil(omega, m, dtype, device):
    """
    Builds Laplacian stencil.

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    dtype : torch.dtype
        data type
    device : String
        compute device

    Returns
    -------
    lx + ly + lz : torch.Tensor
        full Laplacian
    lx : torch.Tensor
        Laplacian in x dimension
    ly : torch.Tensor
        Laplacian in y dimension
    lz : torch.Tensor
        Laplacian in z dimension (None if image is 2D)
    """
    dim = torch.numel(m)
    h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
    if dim == 2:
        lx = (-1 / h[-2] ** 2) * torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=dtype,
                                              device=device).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(
            dim=0)
        ly = (-1 / h[-1] ** 2) * torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=dtype,
                                              device=device).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(
            dim=0)
        return lx + ly, lx, ly, None
    else:
        lx = (-1 / h[-3] ** 2) * torch.tensor(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, -2, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=dtype, device=device).unsqueeze(dim=0).unsqueeze(
            dim=0)
        ly = (-1 / h[-2] ** 2) * torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=dtype, device=device).unsqueeze(dim=0).unsqueeze(
            dim=0)
        lz = (-1 / h[-1] ** 2) * torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=dtype, device=device).unsqueeze(dim=0).unsqueeze(
            dim=0)
        return lx + ly + lz, lx, ly, lz


def myLaplacian2D(omega, m, dtype, device):
    """
    Builds and returns 2D Laplacian in frequency encoding and slice directions.

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    dtype : torch.dtype
        data type
    device : String
        compute device

    Returns
    -------
    L : `FFT3D`
        2D Laplacian as 3D convolution implemented using FFT diagonalization.
    """
    _, Lx, Ly, _ = getLaplacianStencil(omega, m, dtype, device)
    return FFT3D(Lx + Ly, m)


def myLaplacian3D(omega, m, dtype, device):
    """
    Builds and returns 3D Laplacian.

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    dtype : torch.dtype
        data type
    device : String
        compute device

    Returns
    -------
    L : `Conv3D`
        3D Laplacian as 3D convolution.
    """
    padding = [1, 1, 1]
    L, _, _, _ = getLaplacianStencil(omega, m, dtype, device)
    return Conv3D(L, m, padding)
