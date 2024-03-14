from EPI_MRI.utils import *
from EPI_MRI.LeastSquaresCorrection import LeastSquaresCorrection
from EPI_MRI.Regularizers import TikRegularizer, QuadRegularizer
from EPI_MRI.LinearOperators import *
import matplotlib.pyplot as plt
import torch
from optimization.OptimizationLogger import *


class EPIOptimize:
    """
    General superclass for optimizers for EPI-MRI Distortion Correction.

    Attributes
    ----------
    corr_obj : EPIMRIDistortionCorrection
        Contains the optimization problem objective function, data, and parameters.
    verbose : bool
        Flag to print details of optimization.
    B0 : torch.Tensor
        Initial guess for the field map.
    Bc : torch.Tensor
        Optimal field map.
    max_iter : int
        Maximum number of iterations for optimization.
    path : str
        Filepath for log file, results, and saved images.
    log : OptimizationLogger
        Class to handle logging optimization history and information.

    Parameters
    ----------
    corr_obj : `EPIMRIDistortionCorrection`
        contains optimization problem objective function, data, and parameters
    verbose : boolean, optional
        flag to print details of optimization (default is False)
    max_iter : int, optional
        maximum number of iterations (default is 200)
    path: String, optional
        filepath for log file, results, and saved images (default is '')
    """

    def __init__(self, corr_obj, max_iter=200, path='', verbose=False):
        self.verbose = verbose
        self.corr_obj = corr_obj
        self.max_iter = max_iter
        self.B0 = None
        self.Bc = None
        self.path = path
        self.log = OptimizationLogger(path, verbose)

    def visualize(self, slice_num=None, diffusion_num=None, img_min=0, img_max=50, diff_min=-50, diff_max=0):
        """
        Visualize and save results.

        Parameters
        ----------
        slice_num : int, optional
            for 3D and 4D the slice to visualize (default is None)
        diffusion_num : int, optional
            for 4D the diffusion direction to visualize (default is None)
        img_min : int, optional
            vmin value for images (default is 0)
        img_max : int, optional
            vmax value for images (default is 50)
        diff_min : int, optional
            vmin value for (inverted) difference images (default is -50)
        diff_max : int, optional
            vmax value for (inverted) difference images (default is 0)
        """
        if torch.numel(self.corr_obj.dataObj.m) == 2:
            im1 = self.corr_obj.dataObj.I1.data.cpu()
            im2 = self.corr_obj.dataObj.I2.data.cpu()
            Bopt = self.Bc.reshape(list(m_plus(self.corr_obj.dataObj.m))).cpu()
            im1corr = self.corr_obj.corr1.reshape(list(self.corr_obj.dataObj.m)).cpu()
            im2corr = self.corr_obj.corr2.reshape(list(self.corr_obj.dataObj.m)).cpu()
        elif torch.numel(self.corr_obj.dataObj.m) == 3:
            if slice_num is None:
                slice_num = int(self.corr_obj.dataObj.m[0] / 2)
            im1 = self.corr_obj.dataObj.I1.data[slice_num, :, :].cpu()
            im2 = self.corr_obj.dataObj.I2.data[slice_num, :, :].cpu()
            Bopt = self.Bc.reshape(list(m_plus(self.corr_obj.dataObj.m)))[slice_num, :, :].cpu()
            im1corr = self.corr_obj.corr1.reshape(list(self.corr_obj.dataObj.m))[slice_num, :, :].cpu()
            im2corr = self.corr_obj.corr2.reshape(list(self.corr_obj.dataObj.m))[slice_num, :, :].cpu()
        else:  # dim==4
            if slice_num is None:
                slice_num = int(self.corr_obj.dataObj.m[1] / 2)
            if diffusion_num is None:
                diffusion_num = 0
            im1 = self.corr_obj.dataObj.I1.data[diffusion_num, slice_num, :, :].cpu()
            im2 = self.corr_obj.dataObj.I2.data[diffusion_num, slice_num, :, :].cpu()
            Bopt = self.Bc.reshape(list(m_plus(self.corr_obj.dataObj.m)))[diffusion_num, slice_num, :, :].cpu()
            im1corr = self.corr_obj.corr1.reshape(list(self.corr_obj.dataObj.m))[diffusion_num, slice_num, :, :].cpu()
            im2corr = self.corr_obj.corr2.reshape(list(self.corr_obj.dataObj.m))[diffusion_num, slice_num, :, :].cpu()

        dist_init = self.corr_obj.distance(self.corr_obj.dataObj.im1.reshape(-1, 1),
                                           self.corr_obj.dataObj.im2.reshape(-1, 1))[0].item()
        S = QuadRegularizer(myLaplacian3D(self.corr_obj.dataObj.omega,self.corr_obj.dataObj.m,  self.Bc.dtype, self.Bc.device))
        smooth = S.eval(self.Bc)[0].item()

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 4, 2)
        plt.imshow(im1, cmap='gray', origin='lower', vmin=img_min, vmax=img_max)
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$I_{+v}$')
        plt.subplot(2, 4, 3)
        plt.imshow(im2, cmap='gray', origin='lower', vmin=img_min, vmax=img_max)
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$I_{-v}$')
        plt.subplot(2, 4, 4)
        plt.imshow(1 - torch.abs(im1 - im2), cmap='gray', origin='lower', vmin=diff_min, vmax=diff_max)
        plt.xticks([])
        plt.yticks([])
        plt.title("difference=%1.2f%%" % (dist_init / dist_init * 100))
        plt.subplot(2, 4, 5)
        plt.imshow(Bopt, cmap='gray', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title(r"S(b)=%1.4e" % smooth)
        plt.subplot(2, 4, 6)
        plt.imshow(im1corr, cmap='gray', origin='lower', vmin=img_min, vmax=img_max)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 7)
        plt.imshow(im2corr, cmap='gray', origin='lower', vmin=img_min, vmax=img_max)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 8)
        plt.imshow(1 - torch.abs(im1corr - im2corr), cmap='gray', origin='lower', vmin=diff_min, vmax=diff_max)
        plt.xticks([])
        plt.yticks([])
        plt.title("difference=%1.2f%%" % (self.corr_obj.Dc / dist_init * 100))
        plt.savefig(self.path + 'EPIOptimize.png', bbox_inches='tight')
        plt.show()

    def evaluate_optimization(self):
        """
        Evaluates initialization and optimization results.

        Returns
        -------
        dist : torch.Tensor
            corrected image distance
        reg : torch.Tensor
            smoothness regularizer value
        """
        if self.B0 is None or self.Bc is None or self.corr_obj.corr1 is None or self.corr_obj.corr2 is None:
            raise ValueError("need to run optimization first!")
        dist_init = \
            self.corr_obj.distance(self.corr_obj.dataObj.im1.reshape(-1, 1),
                                   self.corr_obj.dataObj.im2.reshape(-1, 1))[
                0].item()
        S = TikRegularizer(myLaplacian3D(self.corr_obj.dataObj.m, self.corr_obj.dataObj.omega, self.dtype, self.device))
        reg_init = S.eval(self.B0, 1.0)[0].item()
        if torch.norm(self.B0) < 1e-5:
            reg_init = 1.0
        dist = self.corr_obj.Dc
        reg = S.eval(self.Bc, 1.0)[0].item()
        res_string = "input dist val:\t%1.4e\t opt dist val:\t%1.4e\t improvement:\t%1.4f\tOT reg val:\t%1.4e\t opt " \
                     "reg val:\t%1.4e\t improvement:\t%1.4f" % (
                         dist_init, dist, (dist_init - dist) / dist_init * 100, reg_init, reg,
                         (reg_init - reg) / reg_init * 100)
        if self.verbose:
            print(res_string)
        self.log.log_message(res_string)
        return dist, reg

    def run_correction(self, B0):
        """
        Runs correction.
        Must be implemented by subclass and store optimal field map in self.Bc.

        Parameters
        ----------
        B0 : torch.Tensor (size mplus(m))
            initial guess for field map
        """
        raise NotImplementedError("Please implement this method to optimize the field map and store in self.Bc")

    def apply_correction(self, method='jac'):
        """
        Apply optimal field map to correct inputs. Saves resulting images and optimal field map as NIFTI files.

        Parameters
        ----------
        method: String, optional
            correction method, either 'jac' for Jacobian modulation (default) or 'lstsq' for least squares restoration

        Returns
        --------
        corr1(, corr2) : torch.Tensor
            corrected image(s)
        """
        self.Bc = self.Bc.detach()
        self.B0 = self.B0.detach()

        save_data(self.Bc.reshape(list(m_plus(self.corr_obj.dataObj.m))).permute(self.corr_obj.dataObj.p),
                  self.path + '-EstFieldMap.nii.gz')

        if method == 'jac':
            corr1 = self.corr_obj.corr1.reshape(list(self.corr_obj.dataObj.m)).permute(self.corr_obj.dataObj.p)
            corr2 = self.corr_obj.corr2.reshape(list(self.corr_obj.dataObj.m)).permute(self.corr_obj.dataObj.p)
            save_data(corr1, self.path + '-im1Corrected.nii.gz')
            save_data(corr2, self.path + '-im2Corrected.nii.gz')
            return corr1, corr2
        elif method == 'lstsq':
            lstsq_corr = LeastSquaresCorrection(self.corr_obj.dataObj, self.corr_obj.A)
            corr = lstsq_corr.apply_correction(self.Bc).permute(self.corr_obj.dataObj.p)
            save_data(corr, self.path + '-imCorrected.nii.gz')
            return corr
        else:
            raise NotImplementedError('correction method not supported')
