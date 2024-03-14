from optimization.EPIOptimize import EPIOptimize
from optimization.ADMM import ADMM
import torch
import time
from EPI_MRI.utils import save_data, m_plus
import matplotlib.pyplot as plt


class LCurve(EPIOptimize):
    """
    EPI-MRI Distortion Correction, choosing alpha parameter with LCurve.

    Attributes
    ----------
    dataObj : `DataObject`
        Original data and parameters.
    loss_func : `EPIMRIDistortionCorrection`
        Contains optimization problem objective function, data, and parameters.
    opt : `EPIOptimize`
        Optimizer to use.
    verbose : bool
        Flag to print details of optimization.
    max_iter : int
        Maximum number of LBFGS iterations.
    Bc : torch.Tensor
        Final field map.
    alpha_min : float
        Smallest alpha to try.
    alpha_max : float
        Largest alpha to try.

    Parameters
    ----------
    data : `DataObject`
        Image data.
    loss_func : `EPIMRIDistortionCorrection`
        Contains optimization problem objective function, data, and parameters.
    opt : `EPIOptimize`
        Optimizer.
    alpha_min: float
        Smallest alpha to try.
    alpha_max: float
        Largest alpha to try.
    num_points: int, optional
        Number of alphas on LCurve (default is 10).
    verbose : bool, optional
        Flag to print details of optimization (default is False).
    path: str, optional
        Filepath for log file, results, and saved images (default is '').
    """
    def __init__(self, data, loss_func, opt, alpha_min, alpha_max, num_points=10, verbose=False, path=''):
        super().__init__(loss_func, max_iter=opt.max_iter, verbose=verbose, path=path)
        self.dataObj = data
        self.loss_func = loss_func
        self.opt = opt
        if alpha_min > alpha_max:
            raise ValueError("alpha_min must be less than alpha_max")
        self.alpha_min = torch.log10(torch.tensor(alpha_min)).item()
        self.alpha_max = torch.log10(torch.tensor(alpha_max)).item()
        self.alphas = torch.logspace(self.alpha_min, self.alpha_max, num_points, device=data.device,
                                     dtype=data.dtype)
        self.max_iter = opt.max_iter
        self.log = opt.log
        self.im = None
        self.dist_list = None
        self.D = None
        self.reg_list = None
        self.S = None
        self.kappa = None
        self.B_list = None
        self.corr1_list = None
        self.corr2_list = None

    def run_correction(self, pre_smooth=True, return_all=False):
        """
        Run optimization (using LCurve).

        Parameters
        ----------
        pre_smooth : bool, optional
            Use pre-smoothing on OT initialization (default is True).
        return_all : bool, optional
            Flag to return all field maps for all alphas (default is False).

        Returns
        ----------
        dict
            for 'min_alpha', 'max_alpha', and 'max_curvature' list of alpha, field map, first corrected image, and
                second corrected image
            when return_all is True, also includes for 'all': [max curvature index, list of alphas,
                list of field maps, list of first corrected images, list of second corrected images,
                list of distances, list of smoothness values, list of curvature values]
        """
        start = time.time()
        alphas = torch.flip(self.alphas, dims=(0,))
        dist_list = []
        reg_list = []
        B_list = []
        corr1_list = []
        corr2_list = []
        B0 = self.loss_func.initialize(blur_result=pre_smooth)
        B0_OT = B0.clone()
        for alpha in alphas:
            self.log.log_message('---------------------------------------------------------------')
            self.log.log_message('alpha='+str(alpha.item()))
            if isinstance(self.opt, ADMM):
                self.loss_func.alpha = [alpha.item(), alpha.item()]
            else:
                self.loss_func.alpha = alpha.item()
            self.opt.run_correction(B0)
            dist_list.append(self.loss_func.Dc)
            reg_list.append(self.loss_func.Sc)
            B_list.append(self.opt.Bc.cpu())
            corr1_list.append(self.loss_func.corr1.reshape(list(self.dataObj.m)).cpu())
            corr2_list.append(self.loss_func.corr2.reshape(list(self.dataObj.m)).cpu())
            B0 = self.opt.Bc

        # calculate curvature information
        D = torch.tensor(dist_list, dtype=self.loss_func.dtype, device=self.loss_func.device)
        S = torch.tensor(reg_list, dtype=self.loss_func.dtype, device=self.loss_func.device)
        h = torch.diff(alphas)
        dD = torch.diff(torch.log(D)) / h
        d2D = torch.diff(torch.log(dD)) / (0.5 * (h[0:-1] + h[1:]))
        dD = 0.5 * (dD[0:-1] + dD[1:])
        dS = torch.diff(torch.log(S)) / h
        d2S = torch.diff(torch.log(torch.abs(dS))) / (0.5 * (h[0:-1] + h[1:]))
        dS = 0.5 * (dS[0:-1] + dS[1:])
        kappa = 2 * (dD * d2S - d2D * dS) / (dD ** 2 + dS ** 2) ** (3 / 2)
        im = int(torch.max(kappa, dim=0).indices) + 1
        ret = {'min_alpha': [alphas[-1], B_list[-1], corr1_list[-1], corr2_list[-1]], 'max_alpha': [alphas[0], B_list[0], corr1_list[0], corr2_list[0]], 'max_curvature': [alphas[im], B_list[im], corr1_list[im], corr2_list[im]], 'OT': [B0_OT]}
        if return_all:
            ret['all'] = [im, alphas, B_list, corr1_list, corr2_list, dist_list, reg_list, kappa]

        stop = time.time()
        minutes = int((stop - start) / 60)
        seconds = (stop - start) - 60 * minutes
        self.log.log_message("total runtime:\t%d min %2.4f sec\n" % (minutes, seconds))

        save_data(B_list[im].reshape(list(m_plus(self.corr_obj.dataObj.m))).permute(self.corr_obj.dataObj.p),
                  self.path + '-EstFieldMap.nii.gz')
        save_data(corr1_list[im].permute(self.dataObj.p), self.path + '-im1Corrected.nii.gz')
        save_data(corr2_list[im].permute(self.dataObj.p), self.path + '-im2Corrected.nii.gz')

        self.im = im
        self.dist_list = dist_list
        self.D = D
        self.reg_list = reg_list
        self.S = S
        self.alphas = alphas
        self.kappa = kappa
        self.B_list = B_list
        self.corr1_list = corr1_list
        self.corr2_list = corr2_list

        return ret

    def visualize(self, slice_num=None, diffusion_num=None, img_min=0, img_max=50, diff_min=-50, diff_max=0):
        """
        Visualize the results, including LCurve and corrected images.

        Parameters
        ----------
        slice_num : int, optional
            Slice number to visualize (default is None).
        diffusion_num : int, optional
            Diffusion number to visualize (default is None).
        img_min : int, optional
            Minimum value for image visualization (default is 0).
        img_max : int, optional
            Maximum value for image visualization (default is 50).
        diff_min : int, optional
            Minimum value for difference visualization (default is -50).
        diff_max : int, optional
            Maximum value for difference visualization (default is 0).
        """
        plt.figure(figsize=(15, 15))

        plt.subplot(2, 2, 1)
        plt.loglog(self.alphas.cpu(), self.D.cpu(), '-o')
        plt.loglog(self.alphas[self.im].cpu(), self.dist_list[self.im].cpu(), 'or')
        plt.title("distance")
        plt.xlabel("alpha")

        plt.subplot(2, 2, 2)
        plt.loglog(self.alphas.cpu(), self.S.cpu(), '-o')
        plt.loglog(self.alphas[self.im].cpu(), self.reg_list[self.im].cpu(), 'or')
        plt.title("regularizer")
        plt.xlabel("alpha")

        plt.subplot(2, 2, 3)
        plt.loglog(self.D.cpu(), self.S.cpu(), '-o')
        plt.loglog(self.dist_list[self.im].cpu(), self.reg_list[self.im].cpu(), 'or')
        plt.text(self.dist_list[self.im].cpu() + 500000, self.reg_list[self.im].cpu(), r'$\alpha=$' + str(round(self.alphas[self.im].item(), 2)))
        plt.text(self.dist_list[-1].cpu() + 500000, self.reg_list[-1].cpu(), r'$\alpha=$' + str(self.alphas[-1].item()))
        plt.text(self.dist_list[0].cpu() + 500000, self.reg_list[0].cpu(), r'$\alpha=$' + str(self.alphas[0].item()))
        plt.title("L-curve")
        plt.xlabel("distance")
        plt.ylabel("regularizer")

        plt.subplot(2, 2, 4)
        plt.semilogx(self.alphas[1:-1].cpu(), self.kappa.cpu(), '-o')
        plt.semilogx(self.alphas[self.im].cpu(), self.kappa[self.im - 1].cpu(), 'or')
        plt.xlabel("alpha")
        plt.title("kappa")

        plt.savefig(self.path + '-LCurve.png', bbox_inches='tight')
        plt.show()

        dist_init = \
            self.loss_func.distance(self.dataObj.I1.data.reshape(-1, 1), self.dataObj.I2.data.reshape(-1, 1))[
                0].item()
        if torch.numel(self.corr_obj.dataObj.m) == 2:
            im1 = self.corr_obj.dataObj.I1.data.cpu()
            im2 = self.corr_obj.dataObj.I2.data.cpu()
            Bmin = self.B_list[-1].reshape(list(m_plus(self.dataObj.m))).cpu()
            Bopt = self.B_list[self.im].reshape(list(m_plus(self.corr_obj.dataObj.m))).cpu()
            Bmax = self.B_list[0].reshape(list(m_plus(self.corr_obj.dataObj.m))).cpu()
            im1min = self.corr1_list[-1].cpu()
            im2min = self.corr2_list[-1].cpu()
            im1opt = self.corr1_list[self.im].cpu()
            im2opt = self.corr2_list[self.im].cpu()
            im1max = self.corr1_list[0].cpu()
            im2max = self.corr2_list[0].cpu()
        elif torch.numel(self.corr_obj.dataObj.m) == 3:
            slice_num = int(self.corr_obj.dataObj.m[0] / 2)
            im1 = self.corr_obj.dataObj.I1.data[slice_num, :, :].cpu()
            im2 = self.corr_obj.dataObj.I2.data[slice_num, :, :].cpu()
            Bmin = self.B_list[-1].reshape(list(m_plus(self.dataObj.m)))[slice_num, :, :].cpu()
            Bopt = self.B_list[self.im].reshape(list(m_plus(self.corr_obj.dataObj.m)))[slice_num, :, :].cpu()
            Bmax = self.B_list[0].reshape(list(m_plus(self.corr_obj.dataObj.m)))[slice_num, :, :].cpu()
            im1min = self.corr1_list[-1][slice_num, :, :].cpu()
            im2min = self.corr2_list[-1][slice_num, :, :].cpu()
            im1opt = self.corr1_list[self.im][slice_num, :, :].cpu()
            im2opt = self.corr2_list[self.im][slice_num, :, :].cpu()
            im1max = self.corr1_list[0][slice_num, :, :].cpu()
            im2max = self.corr2_list[0][slice_num, :, :].cpu()
        else:
            slice_num = int(self.corr_obj.dataObj.m[1] / 2)
            diffusion_num = 0
            im1 = self.corr_obj.dataObj.I1.data[diffusion_num, slice_num, :, :].cpu()
            im2 = self.corr_obj.dataObj.I2.data[diffusion_num, slice_num, :, :].cpu()
            Bmin = self.B_list[-1].reshape(list(m_plus(self.dataObj.m)))[diffusion_num, slice_num, :, :].cpu()
            Bopt = self.B_list[self.im].reshape(list(m_plus(self.corr_obj.dataObj.m)))[diffusion_num, slice_num, :, :].cpu()
            Bmax = self.B_list[0].reshape(list(m_plus(self.corr_obj.dataObj.m)))[diffusion_num, slice_num, :, :].cpu()
            im1min = self.corr1_list[-1][diffusion_num, slice_num, :, :].cpu()
            im2min = self.corr2_list[-1][diffusion_num, slice_num, :, :].cpu()
            im1opt = self.corr1_list[self.im][diffusion_num, slice_num, :, :].cpu()
            im2opt = self.corr2_list[self.im][diffusion_num, slice_num, :, :].cpu()
            im1max = self.corr1_list[0][diffusion_num, slice_num, :, :].cpu()
            im2max = self.corr2_list[0][diffusion_num, slice_num, :, :].cpu()

        vmin = 0
        vmax = 50
        diffmin = -50
        diffmax = 0
        plt.figure(figsize=(10, 10))
        plt.subplot(4, 4, 2)
        plt.imshow(im1, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$I_{+v}$')
        plt.subplot(4, 4, 3)
        plt.imshow(im2, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$I_{+v}$')
        plt.subplot(4, 4, 4)
        plt.imshow(1-torch.abs(im1-im2), cmap='gray', origin='lower', vmin=diffmin, vmax=diffmax)
        plt.xticks([])
        plt.yticks([])
        plt.title("difference=%1.2f%%" % (dist_init / dist_init * 100))
        plt.subplot(4, 4, 5)
        plt.imshow(Bmin, cmap='gray', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title(r"$\alpha$=%1.2f, S(b)=%1.2f" % (self.alphas[-1], self.reg_list[-1]))
        plt.subplot(4, 4, 6)
        plt.imshow(im1min, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 7)
        plt.imshow(im2min, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 8)
        plt.imshow(1-torch.abs(im1min-im2min), cmap='gray', origin='lower', vmin=diffmin, vmax=diffmax)
        plt.xticks([])
        plt.yticks([])
        plt.title("difference=%1.2f%%" % (self.dist_list[-1] / dist_init * 100))
        plt.subplot(4, 4, 9)
        plt.imshow(Bopt, cmap='gray', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title(r"$\alpha$=%1.2f, S(b)=%1.2f" % (self.alphas[self.im], self.reg_list[self.im]))
        plt.subplot(4, 4, 10)
        plt.imshow(im1opt, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 11)
        plt.imshow(im2opt, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 12)
        plt.imshow(1-torch.abs(im1opt-im2opt), cmap='gray', origin='lower', vmin=diffmin, vmax=diffmax)
        plt.xticks([])
        plt.yticks([])
        plt.title("difference=%1.2f%%" % (self.dist_list[self.im] / dist_init * 100))
        plt.subplot(4, 4, 13)
        plt.imshow(Bmax, cmap='gray', origin='lower')
        plt.title(r"$\alpha$=%1.2f, S(b)=%1.2f" % (self.alphas[0], self.reg_list[0]))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 14)
        plt.imshow(im1max, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 15)
        plt.imshow(im2max, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 4, 16)
        plt.imshow(1-torch.abs(im1max-im2max), cmap='gray', origin='lower', vmin=diffmin, vmax=diffmax)
        plt.xticks([])
        plt.yticks([])
        plt.title("difference=%1.2f%%" % (self.dist_list[0] / dist_init * 100))
        plt.savefig(self.path + '_Brains.png', bbox_inches='tight')
        plt.show()
