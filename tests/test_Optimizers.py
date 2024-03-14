import unittest
import os
from optimization.LBFGS import *
from optimization.ADMM import *
from optimization.LinearSolvers import *


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(117)  # reproducibility with randomness
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def test_LBFGS(self):
        """ Tests LBFGS does something that decreases objective function value. """
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
        # Same data has loss func equal to 0
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0)
        yc = loss_func.initialize()
        opt = LBFGS(loss_func, max_iter=10, verbose=False, path=os.getcwd())
        opt.run_correction(yc)
        self.assertIsNotNone(opt.Bc)
        self.assertNotEqual(torch.norm(opt.B0-opt.Bc), 0)
        self.assertLessEqual(loss_func.eval(opt.Bc), loss_func.eval(opt.B0))

        # using autograd
        yc = loss_func.initialize()
        opt.run_correction_use_autograd(yc)
        self.assertIsNotNone(opt.Bc)
        self.assertNotEqual(torch.norm(opt.B0-opt.Bc), 0)
        self.assertLessEqual(loss_func.eval(opt.Bc), loss_func.eval(opt.B0))

        # Delete the temporary files
        os.remove(img_file1)
        os.remove(img_file2)

    def test_GNPCG(self):
        """ Tests GN-PCG does something that decreases objective function value. """
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
        # Same data has loss func equal to 0
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 1e-4, 1e-4)
        yc = loss_func.initialize(blur_result=False)

        # no preconditioning
        opt = GaussNewton(loss_func, max_iter=2, verbose=False, path=os.getcwd())
        opt.run_correction(yc)
        self.assertIsNotNone(opt.Bc)
        self.assertNotEqual(torch.norm(opt.B0-opt.Bc), 0)
        self.assertLessEqual(loss_func.eval(opt.Bc), loss_func.eval(opt.B0))

        # normal CG preconditioned
        loss_func = EPIMRIDistortionCorrection(data, 1e-4, 1e-4, PC=JacobiCG)
        opt = GaussNewton(loss_func, max_iter=2, verbose=False, path=os.getcwd())
        opt.run_correction(yc)
        self.assertIsNotNone(opt.Bc)
        self.assertNotEqual(torch.norm(opt.B0-opt.Bc), 0)
        self.assertLessEqual(loss_func.eval(opt.Bc), loss_func.eval(opt.B0))

        # CG preconditioned block/parallel solve
        # loss_func = EPIMRIDistortionCorrection(data, 1e-4, 1e-4, PC=BlockPCG)
        # opt = GaussNewton(loss_func, max_iter=2, verbose=False, path=os.getcwd())
        # opt.run_correction(yc)
        # self.assertIsNotNone(opt.Bc)
        # self.assertNotEqual(torch.norm(opt.B0-opt.Bc), 0)
        # self.assertLessEqual(loss_func.eval(opt.Bc), loss_func.eval(opt.B0))

        # Delete the temporary files
        os.remove(img_file1)
        os.remove(img_file2)

    def test_ADMM(self):
        """ Tests ADMM does something that decreases objective function value. """
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
        # Same data has loss func equal to 0
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0, regularizer=myLaplacian1D, rho=1e3, PC=JacobiCG)
        yc = loss_func.initialize()
        opt = ADMM(loss_func, max_iter=10, verbose=False, path=os.getcwd())
        opt.run_correction(yc)
        self.assertIsNotNone(opt.Bc)
        self.assertNotEqual(torch.norm(opt.B0-opt.Bc), 0)
        self.assertLessEqual(loss_func.eval(opt.Bc), loss_func.eval(opt.B0))

        # Delete the temporary files
        os.remove(img_file1)
        os.remove(img_file2)

    def test_linear_solvers(self):
        n = 100
        mask = torch.kron(torch.eye(10, 10), torch.ones(10, 10))
        A = torch.randn(n, n)
        H = (A.T @ A) * mask
        # make lambda function for matrix-vector product
        Hfun = lambda x: H @ x
        # D is diagonal of H
        D = torch.diag(H)
        M = lambda x: x / D
        # compute eigenvalues of H
        eigs = torch.real(torch.linalg.eig(H / D)[0])

        xtrue = torch.randn(n)
        b = Hfun(xtrue) + 0.01 * torch.randn(n)
        solverJac = Jacobi(omega=0.3, max_iter=30, tol=1e-6)
        xJac, resOptJac, iterOptJac, itJac, resvecJac = solverJac.eval(Hfun, b, M)
        solverPCG = PCG(max_iter=15, tol=1e-6)
        xPCG, resOptPCG, iterOptPCG, itPCG, resvecPCG = solverPCG.eval(Hfun, b, M)
        solverBlockPCG = BlockPCG(max_iter=15, tol=1e-6)
        xBlockPCG, resOptBlockPCG, iterOptBlockPCG, itBlockPCG, resvecBlockPCG = solverBlockPCG.eval(Hfun, b, M)
        # print("Jacobi: rel.res=%1.4e\titer=%d" % (resOptJac / torch.norm(b), iterOptJac))
        # print("PCG: rel.res=%1.4e\titer=%d" % (torch.norm(H @ xPCG - b) / torch.norm(b), iterOptPCG))
        # print("BlockPCG: rel.res=%1.4e\titer=%d" % (torch.norm(H @ xBlockPCG - b) / torch.norm(b), iterOptBlockPCG))
        self.assertLessEqual(torch.norm(xtrue-xJac), 0.1)
        self.assertLessEqual(torch.norm(xtrue - xPCG), 0.1)
        self.assertLessEqual(torch.norm(xtrue - xBlockPCG), 0.1)



if __name__ == '__main__':
    unittest.main()
