from optimization.EPIOptimize import EPIOptimize
from optimization.LinearSolvers import *
import time
import torch

class GaussNewton(EPIOptimize):
    """
    A class for performing EPI-MRI distortion correction using Gauss-Newton optimization with Jacobi smoother.

    Attributes
    ----------
    corr_obj : EPIMRIDistortionCorrection
        An object containing the optimization problem's objective function, data, and parameters.
    max_iter : int
        Maximum number of Gauss-Newton (outer) iterations
    verbose : bool
        Flag to print details of optimization
    path: str
        Filepath for log file, results, and saved images
    tolG : float
        tolerance for stopping due to gradient norm
    linear_solver : `LinearSolver`
        inner solver
    line_search : `Armijo` or `FixedStep`
        line search method

    Parameters
    ----------
    corr_obj : EPIMRIDistortionCorrection
        An object containing the optimization problem's objective function, data, and parameters.
    max_iter : int, optional
        Maximum number of Gauss-Newton (outer) iterations. Default is 10.
    verbose : bool, optional
        Flag to print details of optimization. Default is False.
    path: str, optional
        Filepath for log file, results, and saved images. Default is an empty string.
    tolG : float, optional
        Tolerance for stopping due to gradient norm. Default is 1e-4.
    linear_solver : `LinearSolver`, optional
        Class to use for inner solver. Default is None (uses `PCG`).
    line_search : `Armijo` or `FixedStep`
        Class to use for line search. Default is None (uses `Armijo`).
    """
    def __init__(self, corr_obj, max_iter=10, tolG=1e-4, linear_solver=None, line_search=None, verbose=False, path=''):
        super().__init__(corr_obj, max_iter=max_iter, verbose=verbose, path=path)
        self.tolG = tolG
        if linear_solver is None:
            self.linear_solver = PCG(max_iter=10, tol=0.1, verbose=False)
        else:
            self.linear_solver = linear_solver
        if line_search is None:
            self.line_search = Armijo() 
        else:   
            self.line_search = line_search
  
    def run_correction(self, B0, yref=None):
        """
        Run the optimization using the calculated derivative.

        Parameters
        ----------
        B0 : torch.Tensor (size m_plus(m))
            Initial guess for the field map.
        yref : torch.Tensor, optional
            Reference data for distortion correction. Default is None.

        Returns
        -------
        Bc : torch.Tensor (size m_plus(m))
            optimal field map
        """
        start = time.time()
        self.B0 = B0
        self.Bc = torch.clone(B0)

        Jc, dJ, H, M = self.corr_obj.eval(self.B0, yref=yref, do_derivative=True, calc_hessian=True)

        Jstop = abs(Jc)
        tolJ = 1e-3
        tolY = 1e-2

        self.log.log_iteration(
            {'iteration': "Iteration", 'loss': "Loss Value", 'CG iters': "Inner Iters", 'CG rel residual': "Inner Rel Residual", 'LS iters': "LS iters",
             'stepsize': "Step Size", 'dist val': "Dist Val", 'reg val': "Reg Val",
             'grad norm': "Grad Norm"})
        self.log.log_iteration({'iteration': 0, 'loss': Jc.item()})
        fevals = 1
        for i in range(1, self.max_iter + 1):
            dy, resOpt, iterOpt, it, resvec = self.linear_solver.eval(H, -1.0 * dJ, M, x=torch.zeros_like(dJ))
            
            # Armijo line-search
            t, yt, LSiter, LS = self.line_search.eval(self.corr_obj,self.Bc, dy, Jc, dJ, yref=yref)
            if t == 0:  # line search failed
                self.log.log_message("line search failed: stopping at iteration %i" % i)
                break

            # save old values and update
            stop_fmap_change = i>1 and (torch.norm(self.Bc - yt)) <= tolY * (1 + torch.norm(B0))
            self.Bc = yt
            Jc, dJ, H, M = self.corr_obj.eval(self.Bc, yref=yref, do_derivative=True, calc_hessian=True)
            
            self.log.log_iteration({'iteration': i, 'loss': Jc.item(), 'CG iters': it, 'CG rel residual': resOpt/resvec[0].item(), 'LS iters': LSiter, 'stepsize': t, 'dist val': self.corr_obj.Dc.item(), 'reg val': self.corr_obj.Sc.item(), 'grad norm': torch.norm(dJ).item()})
            fevals = fevals + 1 + LSiter
            stop_grad = torch.norm(dJ) <= self.tolG
            stop_func_change = i > 1 and (torch.abs(Jc - self.log.history[-2]['loss'])) <= tolJ * (1 + Jstop)
            stop_grad_relative = torch.norm(dJ) <= self.tolG * (1 + Jstop)
            if stop_grad or stop_grad_relative:
                self.log.log_message("reached norm gradient tolerance")
                break
            if stop_func_change:
                self.log.log_message("reached function value change tolerance")
                break
            if stop_fmap_change:
                self.log.log_message("reached field map change tolerance")
                break

        self.log.log_message(f"total function evaluations: {fevals}")
        stop = time.time()
        minutes = int((stop - start) / 60)
        seconds = (stop - start) - 60 * minutes
        self.log.log_message("total runtime:\t%d min %2.4f sec\n" % (minutes, seconds))
        return self.Bc


class FixedStep:
    """
    Uses a fixed-length step size in a line search.

    Attributes
    ---------
    t : float
        step-size

    Parameters
    ----------
    t : float, optional
        step-size (Default is 1)
    """
    def __init__(self, t=1):
        self.t = t
    
    def eval(self, obj, Yc, dY, Jc, dJ, yref=None):
        """
        Computes line search and update.

        Parameters
        ----------
        obj : `EPIMRIDistortionCorrection`
            objective function
        Yc : torch.Tensor
            current field map estimate
        dY : torch.Tensor
            update direction
        Jc : torch.Tensor
            current objective function value
        dJ: torch.Tensor
            current objective function gradient
        yref : torch.Tensor, optional
            reference value (default is None)

        Returns
        -------
        t : float
            step-size
        Yt : torch.Tensor
            updated field map estimate
        i : int
            number of line search iterations (always 0)
        success : Boolean
            success of line search (always True)
        """
        Yt = Yc + self.t * dY
        return self.t, Yt, 0, True


class Armijo:
    """
    Uses Armijo line search.

    Attributes
    ---------
    maxIter : int
        maximum number of line search iterations
    reduction: float
        minimum reduction of objective function value
    t : float
        initial line search step-size
    verbose : Boolean
        flag to print information about line search

    Parameters
    ----------
    maxIter : int, optional
        maximum number of line search iterations (default is 25)
    reduction: float, optional
        minimum reduction of objective function value (default is 1e-6)
    t : float, optional
        initial line search step-size (default is 1)
    verbose : Boolean, optional
        flag to print information about line search (default is False)
    """
    def __init__(self, max_iter=25, reduction=1e-6, t=1, verbose=False):
        self.maxIter = max_iter
        self.reduction = reduction
        self.t = t
        self.verbose = verbose
    
    def eval(self, obj, Yc, dY, Jc, dJ, yref=None):
        """
        Computes line search and update.

        Parameters
        ----------
        obj : `EPIMRIDistortionCorrection`
            objective function
        Yc : torch.Tensor
            current field map estimate
        dY : torch.Tensor
            update direction
        Jc : torch.Tensor
            current objective function value
        dJ: torch.Tensor
            current objective function gradient
        yref : torch.Tensor, optional
            reference value (default is None)

        Returns
        -------
        t : float
            step-size
        Yt : torch.Tensor
            updated field map estimate
        i : int
            number of line search iterations
        success : Boolean
            success of line search
        """
        descent = torch.dot(dJ.view(-1), dY.view(-1))
        if descent > 0:
            print("swapping descent direction in GN")
            dY = -dY
        
        LS = False
        Yt = Yc
        LSiter = 0
        t = self.t
        for LSiter in range(self.maxIter):
            Yt = Yc + t * dY
            Jt = obj.eval(Yt, yref=yref, do_derivative=False)
            LS = (Jt < Jc + t * self.reduction * descent)
            if LS:
                break
            t = t / 2
        
        if LS:
            return t, Yt, LSiter, LS
        else:
            return 0, Yc, LSiter, LS


