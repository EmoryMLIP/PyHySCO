import unittest
import os
from EPI_MRI.EPIMRIDistortionCorrection import *


class TestEPIMRIDistortionCorrection(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(81)  # reproducibility with randomness
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def test_obj_func_3D(self):
        """ Tests objective function evaluation with 3D input. """
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
        data = DataObject(img_file1, img_file1, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0)
        fx = loss_func.eval(torch.zeros(list(m_plus(data.m)), dtype=self.dtype, device=self.device), do_derivative=False)
        self.assertEqual(fx, 0)

        # Make DataObject with the dummy image files (PED = 1)
        # Derivative check
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 0.1, 10.0)
        yc = loss_func.initialize()
        dx = torch.randn_like(yc)
        dx = dx / torch.norm(dx)
        fx, dfx, H, _ = loss_func.eval(yc, do_derivative=True, calc_hessian=True)

        dfdx = torch.sum((dfx * dx).reshape(-1,1), dim=0, keepdim=True)
        Hx = torch.sum(H(dx).reshape(-1,1), dim=0, keepdim=True)

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

    def test_obj_func_4D(self):
        """ Tests objective function evaluation with 4D input. """
        torch.manual_seed(163)
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 7, 9)
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
        data = DataObject(img_file1, img_file1, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 100.0, 10.0)
        fx = loss_func.eval(torch.zeros(list(m_plus(data.m)), dtype=self.dtype, device=self.device), do_derivative=False)
        self.assertEqual(fx, 0)

        # Make DataObject with the dummy image files (PED = 1)
        # Derivative check
        data = DataObject(img_file1, img_file2, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        loss_func = EPIMRIDistortionCorrection(data, 0.1, 10.0)
        yc = loss_func.initialize(blur_result=False)
        dx = torch.randn_like(yc)
        dx = dx / torch.norm(dx)
        fx, dfx, H, _ = loss_func.eval(yc, do_derivative=True, calc_hessian=True)

        dfdx = torch.sum(dfx * dx, dim=0, keepdim=True)
        Hx = torch.sum(H(dx), dim=0, keepdim=True)

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


if __name__ == '__main__':
    unittest.main()
