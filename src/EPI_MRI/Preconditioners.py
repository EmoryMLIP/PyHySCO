from EPI_MRI.Regularizers import *
from optimization.EPIOptimize import *


class Preconditioner(ABC):
    """
    Defines structure of a preconditioner object. All Preconditioner objects have method to build and
    evaluate using preconditioner.
    """
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, x, *args, **kwargs):
        """
        Apply preconditioner.

        Parameters
        ----------
        x : torch.Tensor (size m_plus(m))
            Tensor on which to apply preconditioner.

        Returns
        ----------
        preconditioned tensor
        """
        pass

    @abstractmethod
    def getM(self, *args, **kwargs):
        """
        Calculate preconditioner matrix.

        Parameters
        ----------
        *args, **kwargs : any

        Returns
        ----------
        None, sets self.M to preconditioner for use in PCG
        """
        pass


class JacobiCG(Preconditioner):
    """
    Defines Jacobi preconditioner.

    Attributes
    ----------
    dataObj : `EPIMRIDistortionCorrection.DataObject`
        image data

    Parameters
    ----------
    data : DataObject
        DataObject containing information about the original image data.
    """
    def __init__(self, data):
        super().__init__()
        self.dataObj = data
        self.M = None

    def eval(self, x,  *args, **kwargs):
        """
        Applies the Jacobi preconditioner to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor on which to apply the preconditioner.
      
        Returns
        ----------
        Mx : torch.Tensor
            preconditioned tensor
        """
        return x / self.M

    def getM(self, geom, intensity, hd, d2G, D, A, S, alpha, beta):
        """
        Computes and stores the preconditioner matrix as callable matrix-vector product.

        Parameters
        ----------
        geom : torch.Tensor
            Geometric modulation component of the correction model.
        intensity : torch.Tensor
            Intensity modulation component of the correction model.
        hd : torch.Tensor
            product of cell sizes in image
        d2G : torch.Tensor
            Second derivative of the intensity modulation regularization term.
        D : `LinearOperators.LinearOperator`
           Derivative operator.
        A : `LinearOperators.LinearOperator`
            Averaging operator.
        S : `LinearOperators.LinearOperator`
            Smoothness Laplacian operator.
        alpha : float
            Coefficient for smoothness regularizer.
        beta : float
            Coefficient for the intensity modulation term.

        Returns
        ----------
        Sets self.M to the preconditioner for use in PCG.
        diagD : torch.Tensor
            PC component corresponding to distance term
        diagS : torch.Tensor
            PC component corresponding to smoothness regularizer
        diagP : torch.Tensor
            PC component corresponding to intensity regularizer
        """
        # assert that A is a Conv1D
        assert isinstance(A, Conv1D)
        assert isinstance(D, Conv1D)
        # assert isinstance(S, Conv3D)

        AD = A.op_mul(D)
        D2 = D.op_mul(D)
        A2 = A.op_mul(A)
        # S2 = S.op_mul(S)

        diagD = 2*AD.transp_mat_mul(geom*intensity) + A2.transp_mat_mul(intensity**2) + D2.transp_mat_mul(geom**2)
        diagP = D2.transp_mat_mul(d2G)        
        diagS = S.diag()

        M = hd* diagD + hd*beta*diagP + hd*alpha*diagS
        self.M = M
        return hd* diagD, hd*beta* diagP, hd*alpha*diagS