from optimization.GaussNewton import GaussNewton
from optimization.LinearSolvers import BlockPCG
from EPI_MRI.EPIMRIDistortionCorrection import *
import torch
import time


class ADMM(EPIOptimize):
    """
    Alternating Direction Method of Multipliers (ADMM) for EPI-MRI Distortion Correction.

    Solves min over b,z of D(b) + alpha S1(b) + beta P(b) + alpha S2(z) subject to b=z.

        - b is used for all separable parts and solved using GN PCG

        - z is for the smoothness in non-distortion directions and solved using a proximal solve.

    Attributes
    ----------
    corr_obj : EPIMRIDistortionCorrection
       Contains optimization problem objective function, data, and parameters.
    max_iter : int
       Maximum number of ADMM iterations.
    rho_min : float
       Minimum value of the Lagrangian constant.
    rho_max : float
       Maximum value of the Lagrangian constant.
    verbose : boolean
       Flag to print details of the optimization.
    path : str
       Filepath for log file, results, and saved images.
    g : `Regulariers.QuadRegularizer`
        optimizer for z term
    opt : `GaussNewton.GaussNewton`
        optimizer for b term
    b : torch.Tensor
        current value of b
    z : torch.Tensor
        current value of z
    u : torch.Tensor
        current value of u
    B0 : torch.Tensor
        initial guess for field map
    Bc : torch.Tensor
        optimal field map
    log : `OptimizationLogger`
        class logging optimization information and metrics

    Parameters
    ----------
    corr_obj : EPIMRIDistortionCorrection
        Contains optimization problem objective function, data, and parameters.
    max_iter : int, optional
        Maximum number of ADMM iterations (default is 10).
    rho_min : float, optional
        Minimum value of the Lagrangian constant (default is 100).
    rho_max : float, optional
        Maximum value of the Lagrangian constant (default is 100).
    max_iter_gn : int, optional
        Maximum number of Gauss-Newton (GN) iterations (outer iterations) (default is 3).
    max_iter_pcg : int, optional
        Maximum number of Preconditioned Conjugate Gradient (PCG) iterations (inner iterations) (default is 20).
    tol_gn : float, optional
        Tolerance for the GN optimization (default is 0.1).
    verbose : boolean, optional
        Flag to print details of the optimization (default is False).
    path : str, optional
        Filepath for log file, results, and saved images (default is '').
    """
    def __init__(self, corr_obj, max_iter=10,  rho_min=100, rho_max=100, max_iter_gn=3, max_iter_pcg=20, tol_gn=0.1, verbose=False, path=''):
        super().__init__(corr_obj, max_iter=max_iter, verbose=verbose, path=path)
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.g = QuadRegularizer(myLaplacian2D( corr_obj.dataObj.omega,corr_obj.dataObj.m, corr_obj.dtype, corr_obj.device))
        self.opt = GaussNewton(self.corr_obj, max_iter=max_iter_gn, linear_solver=BlockPCG(max_iter=max_iter_pcg, tol=tol_gn, verbose=False), verbose=False, path=path)
        self.opt.log = OptimizationLogger(path+'GN-', verbose=False)
        self.dist_init = self.corr_obj.distance(self.corr_obj.dataObj.im1.reshape(-1, 1),
                                                self.corr_obj.dataObj.im2.reshape(-1, 1))[0].item()
        self.b = None
        self.z = None
        self.u = None

    def run_correction(self, B0, debug=False):
        """
        Perform the ADMM optimization for EPI-MRI distortion correction.

        Parameters
        ----------
        B0 : torch.Tensor (size mplus(m))
           Initial guess for the field map.
        debug : Boolean, optional
            flag to compute and show Langrangian (default is False)

        Returns
        ----------
        None, sets self.Bc to optimal field map
       """
        start = time.time()
        self.B0 = B0
        self.Bc = torch.clone(B0)

        z = torch.clone(B0)
        b = torch.clone(B0)
        u = b - z

        if debug:
            L,Jc,Dc,Sc,Pc,Qc = self.eval_Lagrangian(b, z, u)
            self.log.log_iteration(
                {'iteration': "Iteration", 'rho': "Rho", 'L after z update': "L after z", 'L after b update': "L after b",
                 'L after u update': "L after u", 'loss': "Loss Val", 'primal residual': "Primal Res", 'dual residual': "Dual Res",
                 'relative distance': "Rel Distance", 'reg val': "Reg Val"})
            self.log.log_iteration(
             {'iteration': 0, 'rho': self.corr_obj.rho, 'Lagrangian': L.item(), '--': "N/A",
              '---': "N/A", 'loss': Jc.item(), 'primal residual': "N/A",
              'dual residual': "N/A",
              'relative distance': (100 - (self.dist_init - Dc.item()) / self.dist_init * 100), 'reg val': Sc.item()})
        else:
            Jc = self.corr_obj.eval(b,yref=z-u, do_derivative=False)
            hd = torch.prod(self.corr_obj.dataObj.h)
            self.log.log_iteration(
                {'iteration': "Iteration", 'rho': "Rho", 'loss': "Loss Val", 'primal residual': "Primal Res",
                 'dual residual': "Dual Res",
                 'Dc': "Dist Val", 'Sc': "Smoothness Val", 'Pc': "Constraint Val", 'Qc': "Prox Val"})
            self.log.log_iteration(
             {'iteration': 0, 'rho': self.corr_obj.rho, 'loss': Jc.item(), 'primal residual': "N/A",
              'dual residual': "N/A",
              'Dc': self.corr_obj.Dc.item(), 'Sc': self.corr_obj.Sc.item(), 'Pc': self.corr_obj.Pc.item(), 'Qc': self.corr_obj.Qc.item()})

        tolY = 1e-2

        for i in range(1, self.max_iter + 1):
            # z update
            z_old = z
            z = self.g.prox_solve(b + u, self.corr_obj.rho/self.corr_obj.alpha)
            if debug:
                L_z = self.eval_Lagrangian(b, z, u)[0]

            # b update
            b = self.opt.run_correction(b, yref=z-u)
            if debug:
                L_b = self.eval_Lagrangian(b, z, u)[0]

            # u update
            u = u + b - z
            if debug:
                L_u,Jc,Dc,Sc,Pc,Qc = self.eval_Lagrangian(b, z,u)

            rk_rel = torch.norm((b-z)/torch.max(torch.norm(b), torch.norm(z)))
            sk_rel = torch.norm((z-z_old)/torch.norm(u))

            if debug:
                self.log.log_iteration({'iteration': i, 'rho': self.corr_obj.rho, 'L after z update': L_z.item(), 'L after b update': L_b.item(), 'L after u update': L_u.item(), 'loss': Jc.item(), 'primal residual': rk_rel.item(), 'dual residual': sk_rel.item(), 'relative distance': (100 - (self.dist_init - Dc) / self.dist_init * 100).item(), 'reg val': Sc.item()})
            else:
                Dc = self.corr_obj.Dc
                Sc = self.corr_obj.Sc
                Pc = self.corr_obj.Pc
                Qc = self.corr_obj.Qc
                Jc = Dc + hd* self.corr_obj.alpha*Sc + hd*self.corr_obj.beta*Pc + self.corr_obj.rho*Qc
                self.log.log_iteration(
                    {'iteration': i, 'rho': self.corr_obj.rho, 'loss': Jc.item(), 'primal residual':  rk_rel.item(),
                    'dual residual': sk_rel.item(),
                    'Dc': Dc.item(), 'Sc': Sc.item(), 'Pc': Pc.item(), 'Qc': Qc.item()})

            # update rho adaptively
            if rk_rel > 10*sk_rel and self.corr_obj.rho*2 <= self.rho_max:
                self.corr_obj.rho *= 2
                u = u / 2

            elif sk_rel > 10*rk_rel and self.corr_obj.rho/2 >= self.rho_min:
                self.corr_obj.rho /=  2
                u = u * 2

            if rk_rel < 1e-5 and sk_rel < 1e-5:
                self.log.log_message("stopping because primal/dual residual tolerance reached")
                break

            stop_fmap_change = i > 1 and (torch.norm(self.b - b)) <= tolY * (1 + torch.norm(B0)) and (
                torch.norm(self.z - z)) <= tolY * (1 + torch.norm(B0)) and (torch.norm(self.u - u)) <= tolY * (
                                           1 + torch.norm(B0))
            if stop_fmap_change:
                self.log.log_message("stopping because field map change tolerance reached")
                break

            self.b = b
            self.z = z
            self.u = u
            self.Bc = b

        stop = time.time()
        minutes = int((stop - start) / 60)
        seconds = (stop - start) - 60 * minutes
        self.log.log_message("total runtime:\t%d min %2.4f sec\n" % (minutes, seconds))

    def eval_Lagrangian(self, b, z, u):
        """
        Evaluate the Lagrangian function for ADMM.

        Parameters
        ----------
        b : torch.Tensor
            Current estimate for the field map.
        z : torch.Tensor
            Dual variable for ADMM.
        u : torch.Tensor
            Proximal variable for ADMM.

        Returns
        ----------
        Lc : torch.Tensor
            Value of the Lagrangian function.
        Jc : torch.Tensor
            Value of objective function for b
        Dc : torch.Tensor
            Distance term value
        Sc : torch.Tensor
            Smoothness term value
        Pc : torch.Tensor
            Intensity modulation term value
        Qc : torch.Tensor
            Proximal term value
        """
        hd = torch.prod(self.corr_obj.dataObj.h)
        Lc = self.corr_obj.eval(b,yref=z-u, do_derivative=False)
        S1 = self.corr_obj.Sc
        S2 = self.g.eval(z,do_derivative=False)[0]
        Lc = Lc + self.corr_obj.alpha/2*hd*S2
        Jc = self.corr_obj.Dc + hd*self.corr_obj.alpha*(S1+S2)+ hd*self.corr_obj.beta*self.corr_obj.Pc 
        return Lc, Jc, self.corr_obj.Dc, S1+S2, self.corr_obj.Pc , self.corr_obj.Qc
        
    