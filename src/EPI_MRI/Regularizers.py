from EPI_MRI.LinearOperators import *


class QuadRegularizer:
    """
    Defines the structure of an quadratic regularization term of the form S(x) = 1/2 ||x||_H^2 where H is positive semi-definite.
    
    Attributes
    ----------
    H :`LinearOperators.LinearOperator`
        The linear operator used in the regularization.
    
    Parameters
    ----------
    H :`LinearOperators.LinearOperator`
        The linear operator used in the regularization.
    """
    def __init__(self, H):
        self.H = H
        
    def eval(self, x, do_derivative=False):
        """
        Evaluate regularizer term.

        Parameters
        ----------
        x : torch.Tensor
            input to regularizer term
        do_derivative : boolean, optional
            flag to compute derivative (default is False)

        Returns
        --------
        S : torch.Tensor
            value of regularizer term
        dS : torch.Tensor
            derivative of regularizer term, None if do_derivative is False
        d2S : `LinearOperators.LinearOperator`
            Hessian of regularizer term, None if do_derivative is False
        """
        Hx = self.H.mat_mul(x)
        S = 0.5*torch.dot(x.reshape(-1), Hx.reshape(-1))
        if do_derivative:
            dS = Hx
            return S, dS, self.H
        else:
            return S, None, None

    def prox_solve(self, z, rho):
        """
        Compute proximal step for the quadratic regularizer, i.e., solve
        
            argmin_x  S(x) + rho/2 || x - z||^2.

        Parameters
        ----------
        z : torch.Tensor
            input
        rho : float
            scalar multiplier

        Returns
        -------
        x : torch.Tensor
            result of solving proximal step
        """
        return self.H.inv(rho*z, rho)


class TikRegularizer:
    """
    Defines the structure of a Tikonov Regularization object.
    Of the form 1/2 ||x - y||^2 where x is input and y is a reference value.

    Attributes
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    hd : float
        product of image cell sizes

    Parameters
    ----------
    omega : torch.Tensor
        image domain
    m : torch.Tensor
        image size
    """
    def __init__(self, omega, m):
        self.omega = omega
        self.m = m
        self.hd = torch.prod((omega[1::2]-omega[:omega.shape[0]-1:2])/m)

    def eval(self, x, rho, y=None, do_derivative=False):
        """
        Evaluate regularizer term.

        Parameters
        ----------
        x : torch.Tensor
            input to regularizer term
        rho : float
            scalar multiplier
        y : torch.Tensor, optional
            reference value (default is None)
        do_derivative : boolean, optional
            flag to compute derivative (default is False)

        Returns
        --------
        S : torch.Tensor
            value of regularizer term
        dS : torch.Tensor
            derivative of regularizer term, None if do_derivative is False
        d2S : `LinearOperators.LinearOperator`
            Hessian of regularizer term, None if do_derivative is False
        """
        if y is None: 
            y = torch.zeros_like(x)
        
        res = x-y
        S = rho*self.hd*0.5*torch.norm(res)**2
        if do_derivative:
            dS = rho*self.hd*res
            return S, dS, Identity(rho*self.hd)
        else:
            return S, None, None
