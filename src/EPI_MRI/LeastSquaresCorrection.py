import math
from EPI_MRI.utils import *


class LeastSquaresCorrection:
    """
    Given a field map, produces a corrected image.

    This class provides functionality to correct distorted images using a field map.
    The method is based on Least Squares Restoration.

    Reference: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ApplytopupFurtherInformation

    Attributes
    ----------
    dataObj : `EPIMRIDistortionCorrection.DataObject`
        Contains original (distorted) data as a DataObject.
    A : `LinearOperators.LinearOperator`
        Averaging operator used for image averaging, e.g. `LinearOperators.myAvg1D`.
    device : str
        The device on which to compute operations, e.g., 'cpu' or 'cuda'.
    xc : torch.Tensor (size m)
        Cell-centered grid in the phase encoding dimension.

    Parameters
    ----------
    data : `EPIMRIDistortionCorrection.DataObject`
        Contains the original (distorted) data as a DataObject.
    A : `LinearOperators.LinearOperator`
        The averaging operator to be used for image averaging, e.g. `LinearOperators.myAvg1D`.
    """
    def __init__(self, data, A):
        self.dataObj = data
        self.A = A
        self.device = data.device
        self.xc = get_cell_centered_grid(self.dataObj.omega, self.dataObj.m, device=self.device)

    def apply_correction(self, yc):
        """
        Solves the least squares problem and returns the corrected image.

        Parameters
        ----------
        yc : torch.Tensor (size m_plus(m))
            A field inhomogeneity map.

        Returns
        ----------
        rhocorr : torch.Tensor (size m)
            The corrected image.
        """
        bc1 = self.A.mat_mul(yc).reshape(-1, 1)  # averaging matrix & translation vector
        xp1 = (self.xc + bc1).reshape(-1, self.dataObj.m[-1])
        xp2 = (self.xc - bc1).reshape(-1, self.dataObj.m[-1])
        rho0 = self.dataObj.I1.data.reshape(-1, self.dataObj.m[-1])
        rho1 = self.dataObj.I2.data.reshape(-1, self.dataObj.m[-1])

        C1 = self.get_push_forward_parallel(self.dataObj.omega[-2:], xp1.shape[1], xp1.shape[0], xp1.clone(), self.dataObj.h[-1], self.dataObj.h[-1])
        C2 = self.get_push_forward_parallel(self.dataObj.omega[-2:], xp2.shape[1], xp2.shape[0], xp2.clone(), self.dataObj.h[-1], self.dataObj.h[-1])
        C = torch.cat((C1, C2), dim=1)
        rhocorr = torch.linalg.lstsq(C, torch.hstack((rho0, rho1))).solution
        return rhocorr.reshape(list(self.dataObj.m))

    def get_push_forward_parallel(self, omega, mc, mf, xp, h, hp):
        """
        Constructs the push forward matrix for distortion correction.

        Parameters
        ----------
        omega : torch.Tensor (size 2*dim)
            The image domain.
        mc : int
            The size of the distortion dimension.
        mf : int
            The size of the non-distortion dimensions.
        xp : torch.tensor (size (-1, m[-1]))
            The distorted grid.
        h : float
            The cell-size in the distortion dimension.
        hp : float
            The cell-size in the distortion dimension.

        Returns
        ----------
        T : torch.Tensor (size mf, mc, xp.shape[1])
            The push forward matrix.
        """
        epsP = hp  # width of particles
        n_parallel = mf
        np = xp.shape[1]  # number of particles
        n = mc  # number of voxels in sampling grid
        pwidth = int((math.ceil(epsP / h)))  # upper bound for support of basis functions

        # map particle positions to the domain [0, mc]
        xp = (xp - omega[0]) / h

        # get cell index of particles center of mass
        P = torch.ceil(xp)
        w = xp - (P - 1)

        B = (self.int1D_parallel(w, pwidth, epsP, h, hp, n_parallel)).reshape(n_parallel, -1, 1)
        J = torch.arange(0, np, device=self.device).repeat(n_parallel, 2 * pwidth + 1, 1).reshape(n_parallel, -1, 1)
        i0 = torch.repeat_interleave(n * torch.arange(0, n_parallel, device=self.device).view(-1, 1), 3 * n).reshape(n_parallel, 3 * n, 1)
        I = (P.unsqueeze(dim=1).repeat(1, 3, 1) + torch.arange(-pwidth - 1, pwidth, device=self.device).unsqueeze(dim=1).expand(P.shape[0], 3, P.shape[1])).reshape(n_parallel, -1, 1)
        valid = torch.logical_and(I >= 0, I < mc)
        I = I[valid].long()
        J = J[valid].long()
        B = B[valid]
        I = I + i0[valid]

        T = torch.zeros(n_parallel * n, np, dtype=B.dtype, device=self.device)
        T[I, J] = B

        return T.reshape(n_parallel, n, np)

    def int1D_parallel(self, w, pwidth, eps, h, hp, n_parallel):
        """
        One-dimensional interpolation for distortion correction.

        Parameters
        ----------
        w : torch.Tensor
            Input data.
        pwidth : int
            Upper bound for the support of basis functions.
        eps : float
            Particle width.
        h : float
            Cell-size in the distortion dimension.
        hp : float
            Cell-size in the distortion dimension.
        n_parallel : int
            Size of the non-distortion dimensions.

        Returns
        ----------
        Bij : torch.Tensor
            Interpolated data.
        """
        Bij = torch.zeros(n_parallel, 2 * pwidth + 1, w.shape[1], dtype=w.dtype, device=self.device)
        Bleft = self.B_parallel(-pwidth - w, eps, h, n_parallel)
        for p in range(-pwidth, pwidth + 1):
            Bright = self.B_parallel(1 + p - w, eps, h, n_parallel)
            Bij[:, p + pwidth, :] = hp * (Bright - Bleft).squeeze()
            Bleft = Bright
        return Bij

    def B_parallel(self, x, eps, h, n_parallel):
        """
        Indexing and combination for one-dimensional interpolation.

        Parameters
        ----------
        x : torch.Tensor
            input data
        eps : float
            particle width
        h : float
            cell-size in distortion dimension
        n_parallel : int
            size of non-distortion dimensions

        Returns
        ----------
        Bij : torch.Tensor
            interpolated data
        """
        Bij = torch.zeros(n_parallel, x.shape[1], dtype=x.dtype, device=self.device)
        ind1 = (-eps / h <= x) & (x <= 0)
        ind2 = (0 < x) & (x <= eps / h)
        ind3 = (eps / h < x)
        Bij[ind1] = x[ind1] + 1 / (2 * eps / h) * x[ind1] ** 2 + eps / (h * 2)
        Bij[ind2] = x[ind2] - 1 / (2 * eps / h) * x[ind2] ** 2 + eps / (h * 2)
        Bij[ind3] = eps / h
        return Bij / eps
