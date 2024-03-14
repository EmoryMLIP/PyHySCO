import unittest
import os
from EPI_MRI.EPIMRIDistortionCorrection import *


class TestPreconditioners(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(81)  # reproducibility with randomness
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def test_PC_3D(self):
        """ Tests Preconditioner with 3D input. """
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 7)
        image_data1 = np.asarray(torch.randn(*shape, dtype=self.dtype)) * 256
        img_file1 = os.getcwd() + 'test1.nii.gz'
        img1 = nib.Nifti1Image(image_data1, np.eye(4))
        nib.save(img1, img_file1)
        image_data2 = np.asarray(torch.randn(*shape, dtype=self.dtype)) * 256
        img_file2 = os.getcwd() + 'test2.nii.gz'
        img2 = nib.Nifti1Image(image_data2, np.eye(4))
        nib.save(img2, img_file2)
        # Make DataObject with the dummy image files (PED = 1)
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0, PC=JacobiCG)
        yc = loss_func.initialize()
        Jc, dJ, H, M = loss_func.eval(yc, do_derivative=True, calc_hessian=True)
        self.assertIsNotNone(M)
        x_pc = M(yc)
        self.assertEqual(x_pc.shape, yc.shape)
        # # regular (not parallelized) PCG solve
        # PC = BlockJacobiCG(data, block_solve=False)
        # loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0, PC=PC)
        # yc = loss_func.initialize()
        # fx = loss_func.eval(yc, do_derivative=True, calc_hessian=True)
        # self.assertIsNotNone(PC.M)
        # x_pc = PC.eval(yc)
        # self.assertEqual(x_pc.shape, yc.shape)

        # Make DataObject with the dummy image files (PED = 1)
        # Derivative check
        dx = torch.randn_like(yc)
        dx = dx / torch.norm(dx)
        fx, dfx, H, M = loss_func.eval(yc, do_derivative=True, calc_hessian=True)

        dfdx = torch.sum(dfx.reshape(-1,1) * dx.reshape(-1,1), dim=0, keepdim=True)
        Hx = torch.sum(M(dx).reshape(-1,1), dim=0, keepdim=True)

        zero = []
        one = []
        two = []
        ratio_zero = []
        ratio_one = []
        ratio_two = []

        num_rounds = 20

        for k in range(num_rounds):
            h = 2 ** (-k)
            ft = loss_func.eval(yc+h*dx, do_derivative=False, calc_hessian=False)

            E0 = torch.norm(fx - ft).cpu()
            E1 = torch.norm(fx + h * dfdx - ft).cpu()
            E2 = torch.norm(fx + h * dfdx + h ** 2 * Hx - ft).cpu()

            zero.append(E0)
            one.append(E1)
            two.append(E2)
            if len(zero) > 1:
                ratio_zero.append(zero[-1] / zero[-2])
                ratio_one.append(one[-1] / one[-2])
                ratio_two.append(two[-1] / two[-2])
                # print("%1.4f\t%1.4f\t%1.4f" % (ratio_zero[-1], ratio_one[-1], ratio_two[-1]))

        # check for expected slope
        zero_count = torch.nonzero((torch.abs(torch.tensor(ratio_zero) - 0.5) <= 1e-2)).shape[0]
        one_count = torch.nonzero(torch.tensor(ratio_one) < 0.25).shape[0]
        two_count = torch.nonzero(torch.tensor(ratio_two) < 0.25).shape[0]

        self.assertGreaterEqual(zero_count, 5)
        self.assertGreaterEqual(one_count, 5)
        self.assertGreaterEqual(two_count, 5)

        # Delete the temporary files
        os.remove(img_file1)
        os.remove(img_file2)

    def test_PC_Jacobi_diag(self):
        """ Checks that JacobiCG preconditioner is diagonal of Hessian """
        shape = (12, 10, 7)
        image_data1 = np.asarray(torch.randn(*shape, dtype=self.dtype)) * 256
        img_file1 = os.getcwd() + 'test1.nii.gz'
        img1 = nib.Nifti1Image(image_data1, np.eye(4))
        nib.save(img1, img_file1)
        image_data2 = np.asarray(torch.randn(*shape, dtype=self.dtype)) * 256
        img_file2 = os.getcwd() + 'test2.nii.gz'
        img2 = nib.Nifti1Image(image_data2, np.eye(4))
        nib.save(img2, img_file2)
        # Make DataObject with the dummy image files (PED = 1)
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0, PC=JacobiCG)
        yc = loss_func.initialize()
        Jc, dJ, H, M = loss_func.eval(yc, do_derivative=True, calc_hessian=True)

        # Approximate the diagonal of Hessian using its matrix-vector product function
        n = torch.prod(m_plus(data.m))
        diagonal = torch.zeros(n, dtype=self.dtype)
        for i in range(n):
            e = torch.zeros(n, dtype=self.dtype)
            e[i] = 1
            diagonal[i] = torch.dot(H(e.reshape(list(m_plus(data.m)))).view(-1), e.view(-1))
        diagonal = diagonal.reshape(list(m_plus(data.m)))
        self.assertSequenceEqual(loss_func.PC.M.shape, diagonal.shape)
        self.assertLessEqual(torch.norm(loss_func.PC.M - diagonal), 1e-5)

if __name__ == '__main__':
    unittest.main()
