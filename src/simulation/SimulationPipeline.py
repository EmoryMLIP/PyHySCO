"""
This file contains the methods and script to simulate distortions from a fieldmap and T2w image.
See __main__ below for details.
"""

from fsl.wrappers import flirt, fsl_prepare_fieldmap, bet, fslsplit
import sys
from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import math


def get_nodal_grid(omega, m, device='cpu'):
    """
    Generate the nodal grid of size m over domain omega.

    Parameters
    ----------
    omega : torch.Tensor (size # of dimensions x 2)
        Image domain.
    m : torch.Tensor (size # of dimensions)
        Discretization size.
    device : str, optional
        Device on which to compute operations (default is 'cpu').

    Returns
    ----------
    x : torch.Tensor (size (prod(m),1)
       Nodal grid in the distortion dimension
    """
    def nu(i):
        return torch.linspace(omega[2*i-2], omega[2*i-1], m[i-1], device=device)
    x = None
    if torch.numel(m) == 2:
        x = torch.meshgrid(nu(1), nu(2), indexing='ij')[1]
        x = x.reshape(-1, 1)
    elif torch.numel(m) == 3:
        x = torch.meshgrid(nu(1), nu(2), nu(3), indexing='ij')[2]
        x = x.reshape(-1, 1)
    elif torch.numel(m) == 4:
        x = torch.meshgrid(nu(1), nu(2), nu(3), nu(4), indexing='ij')[3]
        x = x.reshape(-1, 1)
    return x


class DistortionSimulation:
    """
    Given a field map and original image, produces a distorted image.

    Attributes
    ----------
    dataObj : DataObject
        Contains original (distorted) data as a DataObject.
    device : str
        The device on which to compute operations, e.g., 'cpu' or 'cuda'.
    xc : torch.Tensor (size prod(m))
       nodal grid in the phase encoding dimension.

    Parameters
    ----------
    data : `EPIMRIDistortionCorrection.DataObject`
        Contains the original (undistorted) data as a DataObject.
    """
    def __init__(self, data):
        self.dataObj = data
        self.device = data.device
        self.xc = get_nodal_grid(self.dataObj.omega, self.dataObj.m, device=self.device)

    def simulate_distortion(self, I, yc):
        """
        Simulates distortion of I using field map yc.

        Parameters
        ---------
        I : torch.Tensor
            undistorted image
        yc : torch.Tensor
            field map

        Returns
        -------
        CI : torch.Tensor
            distorted image
        """
        yc = self.average_to_cell_centers(yc)
        xp1 = (self.xc + yc).reshape(-1, self.dataObj.m[-1])
        C1 = self.get_push_forward_parallel(self.dataObj.omega[-2:], xp1.shape[1], xp1.shape[0], xp1.clone(), self.dataObj.h[-1], self.dataObj.h[-1])
        return C1@I.reshape(-1, self.dataObj.m[-1], 1)

    def average_to_cell_centers(self, x):
        """
        Averages field map in cell centers.

        Parameters
        ----------
        x : torch.Tensor
            field map

        Returns
        --------
        x : torch.Tensor
            field map averaged in cell centers

        """
        a = torch.tensor([0.5, 0.5], dtype=self.dataObj.dtype, device=self.device).reshape(1, 1, -1)
        x = x.contiguous().view(-1, 1, self.dataObj.m[-1])
        Ax = torch.nn.functional.conv1d(x, a, padding=[1], dilation=2)
        return Ax.view(-1, 1)

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
        bc1 = self.average_to_cell_centers(yc)
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


if __name__ == "__main__":
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    data = sys.argv[1]
    data_path = "simulation/" + data + "/"
    severity = sys.argv[2]
    # determines threshold of pixels violating intensity constraint; allowing more => more distortion
    if severity == "low":
        percent = 0.005
    elif severity == "medium":
        percent = 0.01
    elif severity == "high":
        percent = 0.1
    else:
        print("improper severity use low/medium/high")
        exit(1)

    # 1. split magnitude image
    print("splitting magnitude image")
    fslsplit(data_path + data+"_3T_FieldMap_Magnitude.nii.gz", out=data_path + "Magnitude")

    # 2. brain extract T2w
    print("brain extract T2w")
    bet(data_path+data+"_3T_T2w_SPC1.nii.gz", data_path+"T2w_mask.nii.gz")

    # 3. register magnitude and phase to T2w
    print("register magnitude and phase to T2w")
    flirt(data_path+"Magnitude0000.nii.gz", data_path+"T2w_mask.nii.gz", out=data_path+"MagnitudeRegistered.nii.gz", omat=data_path+"MagnitudeRegistered.mat")
    flirt(data_path+data+"_3T_FieldMap_Phase.nii.gz", data_path+"T2w_mask.nii.gz", out=data_path+"PhaseRegistered.nii.gz", applyxfm=True, init=data_path+"MagnitudeRegistered.mat")

    # 4. brain extract magnitude
    print("brain extract magnitude")
    bet(data_path+"MagnitudeRegistered.nii.gz", data_path+"MagnitudeRegistered_brain.nii.gz")

    # 5. get field map (fsl_prepare_fieldmap is a wrapper command for prelude specific to SIEMENS scans)
    print("get field map")
    fsl_prepare_fieldmap(data_path+"PhaseRegistered.nii.gz", data_path+"MagnitudeRegistered_brain.nii.gz", data_path+"FieldMap.nii.gz", 4.92)

    # 6. visualize results
    t2w = torch.tensor(np.asarray(nib.load(data_path+data+'_3T_T2w_SPC1.nii.gz').dataobj))
    print(t2w.shape)
    x = int(t2w.shape[0]/2)
    y = int(t2w.shape[1]/2)
    z = int(t2w.shape[2]/2)
    plt.figure()
    plt.title("T2w")
    plt.subplot(7,3,1)
    plt.imshow(t2w[:,:,z], cmap='gray', origin='lower')
    plt.subplot(7,3,2)
    plt.imshow(t2w[:,y,:], cmap='gray', origin='lower')
    plt.subplot(7,3,3)
    plt.imshow(t2w[x,:,:], cmap='gray', origin='lower')
    t2w_mask = torch.tensor(np.asarray(nib.load(data_path+'T2w_mask.nii.gz').dataobj))
    print(t2w_mask.shape)
    plt.subplot(7,3,4)
    plt.imshow(t2w_mask[:,:,z], cmap='gray', origin='lower')
    plt.ylabel("t2w mask")
    plt.subplot(7,3,5)
    plt.imshow(t2w_mask[:,y,:], cmap='gray', origin='lower')
    plt.subplot(7,3,6)
    plt.imshow(t2w_mask[x,:,:], cmap='gray', origin='lower')
    fieldmag = torch.tensor(np.asarray(nib.load(data_path+'Magnitude0000.nii.gz').dataobj))
    print(fieldmag.shape)
    xf = int(fieldmag.shape[0]/2)
    yf = int(fieldmag.shape[1]/2)
    zf = int(fieldmag.shape[2]/2)
    plt.subplot(7,3,7)
    plt.imshow(fieldmag[:,:,zf], cmap='gray', origin='lower')
    plt.ylabel("magnitude")
    plt.subplot(7,3,8)
    plt.imshow(fieldmag[:,yf,:], cmap='gray', origin='lower')
    plt.subplot(7,3,9)
    plt.imshow(fieldmag[xf,:,:], cmap='gray', origin='lower')
    mag_registered = torch.tensor(np.asarray(nib.load(data_path+'MagnitudeRegistered_brain.nii.gz').dataobj))
    print(mag_registered.shape)
    plt.subplot(7,3,10)
    plt.imshow(mag_registered[:,:,z], cmap='gray', origin='lower')
    plt.ylabel("magnitude registered")
    plt.subplot(7,3,11)
    plt.imshow(mag_registered[:,y,:], cmap='gray', origin='lower')
    plt.subplot(7,3,12)
    plt.imshow(mag_registered[x,:,:], cmap='gray', origin='lower')
    fieldphase = torch.tensor(np.asarray(nib.load(data_path+data+'_3T_FieldMap_Phase.nii.gz').dataobj))
    print(fieldphase.shape)
    plt.subplot(7,3,13)
    plt.imshow(fieldphase[:,:,zf], cmap='gray', origin='lower')
    plt.ylabel("phase")
    plt.subplot(7,3,14)
    plt.imshow(fieldphase[:,yf,:], cmap='gray', origin='lower')
    plt.subplot(7,3,15)
    plt.imshow(fieldphase[xf,:,:], cmap='gray', origin='lower')
    phase_registered = torch.tensor(np.asarray(nib.load(data_path+'PhaseRegistered.nii.gz').dataobj))
    print(phase_registered.shape)
    plt.subplot(7,3,16)
    plt.imshow(phase_registered[:,:,z], cmap='gray', origin='lower')
    plt.ylabel("phase registered")
    plt.subplot(7,3,17)
    plt.imshow(phase_registered[:,y,:], cmap='gray', origin='lower')
    plt.subplot(7,3,18)
    plt.imshow(phase_registered[x,:,:], cmap='gray', origin='lower')
    fieldmap = torch.tensor(np.asarray(nib.load(data_path+'FieldMap.nii.gz').dataobj), dtype=torch.float64, device=device)
    print(fieldmap.shape)
    plt.subplot(7,3,19)
    plt.imshow(fieldmap[:,:,z].cpu(), cmap='gray', origin='lower')
    plt.ylabel("fieldmap")
    plt.subplot(7,3,20)
    plt.imshow(fieldmap[:,y,:].cpu(), cmap='gray', origin='lower')
    plt.subplot(7,3,21)
    plt.imshow(fieldmap[x,:,:].cpu(), cmap='gray', origin='lower')
    plt.show()

    # 7. scale fieldmap
    print("scale fieldmap")
    dataobj = DataObject(data_path+'T2w_mask.nii.gz', data_path+'T2w_mask.nii.gz', 2, device=device)
    p = [2, 0, 1]
    fieldmap_p = fieldmap.permute(p)
    scale = 1
    dvb = 1 / dataobj.h[-1] * torch.diff((fieldmap_p / scale).reshape(list((dataobj.m))), dim=-1)
    violating = torch.nonzero(torch.abs(dvb) > 1).shape[0]
    while violating / torch.prod(torch.tensor(dvb.shape)) > percent:
        scale += 10
        dvb = 1 / dataobj.h[-1] * torch.diff((fieldmap_p / scale).reshape(list((dataobj.m))), dim=-1)
        violating = torch.nonzero(torch.abs(dvb) > 1).shape[0]
    print(violating / torch.prod(torch.tensor(dvb.shape)))

    # 8. simulate distortion
    print("simulate distortion")
    sim = DistortionSimulation(dataobj)
    im1 = sim.simulate_distortion(dataobj.I1.data, (fieldmap_p / scale).reshape(-1, 1)).reshape(list(dataobj.m))
    im2 = sim.simulate_distortion(dataobj.I1.data, -1 * (fieldmap_p / scale).reshape(-1, 1)).reshape(list(dataobj.m))
    save_data(im1.permute(dataobj.p), data_path + severity + "_+v.nii.gz")
    save_data(im2.permute(dataobj.p), data_path + severity + "_-v.nii.gz")
    save_data(fieldmap_p/scale, data_path + severity + "_fieldmap.nii.gz")

    # 9. visualize distortion
    plt.subplot(7, 3, 1)
    plt.imshow(fieldmap[:, :, z].cpu() / scale, cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("fieldmap")
    plt.subplot(7, 3, 2)
    plt.imshow(fieldmap[:, y, :].cpu() / scale, cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 3)
    plt.imshow(fieldmap[x, :, :].cpu() / scale, cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 4)
    plt.imshow(im1.permute(dataobj.p)[:, :, z].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("im1")
    plt.subplot(7, 3, 5)
    plt.imshow(im1.permute(dataobj.p)[:, y, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 6)
    plt.imshow(im1.permute(dataobj.p)[x, :, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 7)
    plt.imshow(im2.permute(dataobj.p)[:, :, z].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("im2")
    plt.subplot(7, 3, 8)
    plt.imshow(im2.permute(dataobj.p)[:, y, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 9)
    plt.imshow(im2.permute(dataobj.p)[x, :, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 10)
    plt.imshow(im1.permute(dataobj.p)[:, :, z].cpu() - im2.permute(dataobj.p)[:, :, z].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("diff")
    plt.subplot(7, 3, 11)
    plt.imshow(im1.permute(dataobj.p)[:, y, :].cpu() - im2.permute(dataobj.p)[:, y, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 12)
    plt.imshow(im1.permute(dataobj.p)[x, :, :].cpu() - im2.permute(dataobj.p)[x, :, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 13)
    plt.imshow(dataobj.I1.data.permute(dataobj.p)[:, :, z].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("t2w")
    plt.subplot(7, 3, 14)
    plt.imshow(dataobj.I1.data.permute(dataobj.p)[:, y, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 15)
    plt.imshow(dataobj.I1.data.permute(dataobj.p)[x, :, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    original = dataobj.I1.data
    sim.dataObj.I1 = im1
    sim.dataObj.I2 = im2
    corrected_im = sim.apply_correction((fieldmap_p/scale).reshape(-1, 1))
    print(torch.norm(corrected_im - original))
    plt.subplot(7, 3, 16)
    plt.imshow(corrected_im.permute(dataobj.p)[:, :, z].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("corrected")
    plt.subplot(7, 3, 17)
    plt.imshow(corrected_im.permute(dataobj.p)[:, y, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 18)
    plt.imshow(corrected_im.permute(dataobj.p)[x, :, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 19)
    plt.imshow(original.permute(dataobj.p)[:, :, z].cpu()-corrected_im.permute(dataobj.p)[:, :, z].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.ylabel("diff")
    plt.subplot(7, 3, 20)
    plt.imshow(original.permute(dataobj.p)[:, y, :].cpu()-corrected_im.permute(dataobj.p)[:, y, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.subplot(7, 3, 21)
    plt.imshow(original.permute(dataobj.p)[x, :, :].cpu()-corrected_im.permute(dataobj.p)[x, :, :].cpu(), cmap='gray', origin='lower')
    plt.colorbar()
    plt.show()
