import torch

class Jacobi:
    """
    Damped Jacobi solver for Ax=b

    Attributes
    ----------
    max_iter : int
        maximum number of iterations
    tol : float
        tolerance for convergence
    omega : float
        dampening parameter
    verbose : Boolean
        flag to print optimization information

    Parameters
    ----------
    max_iter : int, optional
        maximum number of iterations (default is 10)
    tol : float, optional
        tolerance for convergence (default is 1e-2)
    omega : float, optional
        dampening parameter (default is 2/3)
    verbose : Boolean, optional
        flag to print optimization information (default is False)
    """
    def __init__(self, omega=2/3, max_iter=10, tol=1e-2, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.omega = omega
        self.verbose = verbose
        
    def eval(self, A, b, M, x=None):
        """
        Solve system Ax=b using Jacobi with initial guess x and preconditioner M.

        Parameters
        ---------
        A : callable
            Hessian mat-vec
        b : torch.Tensor
            RHS of the system (negative Jacobian)
        M : callable
            preconditioner
        x : torch.Tensor, optional
            initial guess (default is None)

        Returns
        -------
        x : torch.Tensor
            optimal value
        res : float
            residual of system
        it : int
            number of iterations
        itOpt : int
            optimal iteration
        resvec : list
            list of residuals at each iteration
        """
        if x is None:
            x = b * 0.0
        r = b - A(x)
        resvec = []
        resvec.append(torch.norm(r))

        it = 0
        while it < self.max_iter:
            it += 1
            x = self.omega*(M(b)) + (x - M(self.omega*A(x)))
            # only compute running residual if requested
            if self.verbose:
                r = b - A(x)
                resvec.append(torch.norm(r))
                if resvec[-1]/resvec[0] < self.tol:
                    print("stopping because tol reached")
                    break
                print("Jacobi iter:%d\trel res:%1.4e\n" % (it, resvec[-1].item() / resvec[0].item()))
        # calculate final residual if not yet done
        if not self.verbose:
            r = b - A(x)
            resvec.append(torch.norm(r))

        return x, resvec[-1].item(), it, it, resvec


class PCG:
    """
    A class for performing Preconditioned Conjugate Gradient (PCG) optimization.

    Attributes
    ----------
    max_iter : int
       Maximum number of PCG iterations.
    tol : float
       Stopping tolerance for PCG.
    verbose : bool
       Flag to print details of the optimization.

    Parameters
    ----------
    max_iter : int, optional
       Maximum number of PCG iterations. Default is 10.
    tol : float, optional
       Stopping tolerance for PCG. Default is 1e-2.
    verbose : bool, optional
       Flag to print details of the optimization. Default is False.
   """
    def __init__(self, max_iter=10, tol=1e-2, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def eval(self, A, b, M=None, x=None):
        """
        Solve the linear system A * x = b using Preconditioned Conjugate Gradient (PCG) optimization.

        Parameters
        ----------
        A : Callable
            A function that represents the matrix-vector multiplication for matrix A.
        M : Callable, optional
            preconditioner (default is None)
        b : torch.Tensor
            The right-hand side vector of the linear system.
        x : torch.Tensor, optional
            Initial guess for the solution. (default is None)

        Returns
        ----------
        x : torch.Tensor
            optimal value
        res : float
            residual of system
        it : int
            number of iterations
        itOpt : int
            optimal iteration
        resvec : list
            list of residuals at each iteration
        """
        if x is None:
            x = b * 0.0
            r = torch.clone(b)
        else:
            r = b + -1.0 * A(x)
        if M is None:
            M = lambda x: x.clone()
        
        z = M(r)
        p = z

        resvec = []
        resvec.append(torch.norm(r))

        # save best iterate
        xOpt = torch.clone(x)
        resOpt = resvec[0]
        iterOpt = 0

        it = 0
        while it < self.max_iter:
            it += 1
            Ap = A(p)
            gamma = torch.dot(r.view(-1), z.view(-1))
            curv = torch.dot(p.view(-1), Ap.view(-1))
            alpha = gamma / curv

            if alpha == float("Inf") or torch.all(torch.isnan(alpha)) or alpha < 0:
                print("stopping because of alpha: " + str(alpha))
                break

            x = x + alpha * p
            r = r + -alpha * Ap

            resvec.append(torch.norm(r))

            if True or resvec[-1] < resOpt:
                resOpt = resvec[-1]
                iterOpt = it
                xOpt = torch.clone(x)

            if resvec[-1]/resvec[0] < self.tol:
                if self.verbose:
                    print("stopping because tol reached")
                break

            z = M(r)

            beta = torch.dot(z.view(-1), r.view(-1)) / gamma
            p = z + beta * p
            if self.verbose:
                print("CG iter:%d\trel res:%1.4e\n" % (it, resOpt.item() / resvec[0].item()))
        if self.verbose:
            print("# CG iters: " + str(it) + "\topt CG iter: " + str(iterOpt) + "\trel residual: " + str(resOpt.item()/resvec[0].item()) + "\n")
        return xOpt, resOpt.item(), iterOpt, it, resvec


class BlockPCG:
    """
    PCG solver for blkdiag(A1, \ldots, An) * x = (b1, \ldots, bn)

    Attributes
    ----------
    max_iter : int
       Maximum number of PCG iterations.
    tol : float
       Stopping tolerance for PCG.
    verbose : bool
       Flag to print details of the optimization.

    Parameters
    ----------
    max_iter : int, optional
       Maximum number of PCG iterations. Default is 10.
    tol : float, optional
       Stopping tolerance for PCG. Default is 1e-2.
    verbose : bool, optional
       Flag to print details of the optimization. Default is False.
    """
    def __init__(self, max_iter=10, tol=1e-2, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def eval(self, A, b, M=None, x=None):        
        """
        Solve the linear system blkdiag(A1, \ldots, An) * x = (b1, \ldots, bn) using parallelized PCG.

        Parameters
        ----------
        A : Callable
           A function that represents the matrix-vector multiplication for matrix A.
        b : torch.Tensor
           The right-hand side vector of the linear system.
        M : Callable, optional
           Preconditioner. (default is None)
        x : torch.Tensor, optional
           Initial guess for the solution. (default is None)

        Returns
        ----------
        x : torch.Tensor
            optimal value
        res : float
            residual of system
        it : int
            number of iterations
        itOpt : int
            optimal iteration
        resvec : list
            list of residuals at each iteration
       """
        if x is None:
            x = b * 0.0
            r = torch.clone(b)
        else:
            x = x
            r = b - A(x)
        if M is None:
            M = lambda x: x.clone()

        z = M(r)
        p = z.clone()

        resvec = []
        resvec_batched = []
        resvec.append(torch.norm(r))
        resvec_batched.append(torch.norm(r, dim=-1, keepdim=True))

        # save best iterate
        xOpt = torch.clone(x)
        resOpt = resvec_batched[0].clone()
        iterOpt = torch.zeros(*(x.shape[:-1]+(1,)), device=x.device, dtype=x.dtype)
        rOpt = torch.clone(r)

        converged = torch.zeros_like(iterOpt)

        it = 0
        while it < self.max_iter:
            it += 1
            Ap = A(p)
            gamma = torch.sum(r*z, dim=-1, keepdim=True)
            curv = torch.sum(p*Ap, dim=-1, keepdim=True)
            alpha = gamma / curv

            converged = torch.logical_or(converged, torch.logical_or(torch.isinf(alpha), torch.logical_or(torch.isnan(alpha), alpha < 0)))

            if torch.all(converged):
                print("stopping because of alpha")
                break

            alpha[converged] = 0

            x = x + alpha * p
            r = r - alpha * Ap

            resvec.append(torch.norm(r))
            resvec_batched.append(torch.norm(r, dim=-1, keepdim=True))

            better = (resvec_batched[-1] < resOpt)
            resOpt[better] = resvec_batched[-1][better].clone()
            iterOpt[better] = it
            xOpt[better.squeeze()] = x[better.squeeze()]
            rOpt[better.squeeze()] = r[better.squeeze()]
            converged = torch.logical_or(converged, ((resvec_batched[-1]/resvec_batched[0]) < self.tol))
            if torch.norm(rOpt).item() / resvec[0].item() < self.tol or torch.all(converged):
                if self.verbose:
                    print("stopping because tol reached")
                break
            
            z = M(r)

            beta = torch.sum(z*r, dim=-1, keepdim=True) / gamma
            beta[converged] = 0
            p = z + beta * p
            if self.verbose:
                print("batch CG iter:%d\trel res:%1.4e\tnum converged:%d" % (it, torch.norm(rOpt).item() / resvec[0].item(), torch.count_nonzero(converged).item()))
        if self.verbose:
            print("# BlockCG iters: " + str(it) + "\trel residual: " + str(torch.norm(rOpt).item()/resvec[0].item()))

        return xOpt, torch.norm(rOpt).item(), iterOpt, it, resvec
