import unittest
import torch
from EPI_MRI.LinearOperators import  Conv3D,  FFT3D, getLaplacianStencil
from EPI_MRI.utils import m_plus

class TestLaplacians(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        self.m = torch.tensor([128, 192, 256])  # replace with your actual value
        self.omega = torch.tensor([0.0, 1.0, 0.0, 1.0,0.0,  1.0])  # replace with your actual value
        self.dtype = torch.float64  # replace with your actual value
        self.device = torch.device('cpu')  # replace with your actual value

        # Compute the Laplacian stencil
        self.kernel, _, _, _ = getLaplacianStencil(self.m, self.omega, self.dtype, self.device)



    def test_laplacians_3d(self):
        """ Test 3D Laplacian kernel as built in getLaplacianStencil """
        # Initialize the operators
        conv_operator = Conv3D(self.kernel, self.m)
        fft_operator = FFT3D(self.kernel, self.m)

        # Create some input data
        x = torch.randn(tuple(m_plus(self.m)), dtype=self.dtype, device=self.device)

        # Compute the Laplacians
        conv_laplacian = conv_operator.mat_mul(x).reshape(tuple(m_plus(self.m)))
        fft_laplacian = fft_operator.mat_mul(x)

        # Exclude the first and last entry in every dimension
        conv_laplacian = conv_laplacian[1:-1, 1:-1, 1:-1]
        fft_laplacian = fft_laplacian[1:-1, 1:-1, 1:-1]

        # Check if the relative error between the two Laplacians is small
        assert torch.norm(conv_laplacian - fft_laplacian) / torch.norm(conv_laplacian) < 1e-5
    
        
    def test_inverse_laplacians_3d(self):
        """ Test 3D Laplacian kernel inverse as built in getLaplacianStencil and defined in FFT3D """
        # Initialize the operators
        fft_operator = FFT3D(self.kernel, self.m)
        rho = 1e-2

        # Create some input data
        x = torch.randn(tuple(m_plus(self.m)), dtype=self.dtype, device=self.device)

        # Compute the Laplacians
        fft_laplacian = fft_operator.mat_mul(x) + rho*x
        inv_laplacian = fft_operator.inv(fft_laplacian, rho)

        # Check if inv_laplacian is close to x
        assert torch.norm(inv_laplacian - x ) / torch.norm(x) < 1e-5
    
if __name__ == '__main__':
    unittest.main()