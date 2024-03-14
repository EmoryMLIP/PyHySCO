import unittest
import torch
from EPI_MRI.utils import get_cell_centered_grid
from EPI_MRI.ImageModels import Interp1D


class TestImageModels(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(81)  # reproducibility with randomness
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64

    def test_2D_interp(self):
        """ Tests for eval method of Interp1D with 2D input. """
        omega = torch.tensor([0, 1, 0, 3], device=self.device, dtype=self.dtype)
        m = torch.tensor([14, 26], device=self.device, dtype=torch.int)
        xc_full = get_cell_centered_grid(omega, m, return_all=True).reshape(2, -1)
        xc = xc_full[-1, :]
        y = (xc ** 2).reshape(list(m))

        im_model = Interp1D(y, omega, m, dtype=self.dtype, device=self.device)

        # interpolation on same points should yield original data
        ys_same = im_model.eval(xc)
        self.assertLessEqual(torch.norm(y - ys_same.reshape(list(m))), 1e-10)

        # close linear interpolation
        xc_full[-1, :] = xc_full[-1, :] + 1e-2
        dist = torch.randn_like(xc) * 1e-2
        xc = xc - dist
        xc_full[-1, :] = xc_full[-1, :] - dist
        ys_1d = im_model.eval(xc)
        y_dist = (xc ** 2).reshape(list(m))
        self.assertLessEqual(torch.norm(ys_1d.reshape(list(m)) - y_dist), 1)

    def test_3D_interp(self):
        """ Tests for eval method Interp1D with 3D input. """
        omega = torch.tensor([0, 1, 0, 3, 1, 5], device=self.device, dtype=torch.float64)
        m = torch.tensor([7, 14, 26], device=self.device, dtype=torch.int)
        xc_full = get_cell_centered_grid(omega, m, return_all=True).reshape(3, -1)
        xc = xc_full[-1, :]
        y = (xc ** 2).reshape(list(m))

        im_model = Interp1D(y, omega, m, device=self.device, dtype=self.dtype)

        # interpolation on same points should yield original data
        ys_same = im_model.eval(xc)
        self.assertLessEqual(torch.norm(y - ys_same.reshape(list(m))), 1e-10)

        # close interpolation
        xc = xc - 1e-5
        xc_full[-1, :] = xc_full[-1, :] - 1e-5
        y_dist = (xc ** 2).reshape(list(m))
        fx = im_model.eval(xc)
        self.assertLessEqual(torch.norm(fx.reshape(list(m)) - y_dist), 1)

    def test_4D_interp(self):
        """ Tests for eval method of Interp1D with 4D input. """
        omega = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=self.device, dtype=torch.float64)
        m = torch.tensor([3, 7, 14, 26], device=self.device, dtype=torch.int)
        xc_full = get_cell_centered_grid(omega, m, return_all=True).reshape(4, -1)
        xc = xc_full[-1, :]
        y = (xc ** 2).reshape(list(m))

        im_model = Interp1D(y, omega, m, device=self.device, dtype=self.dtype)

        # interpolation on same points should yield original data
        ys_same = im_model.eval(xc)
        self.assertLessEqual(torch.norm(y - ys_same.reshape(list(m))), 1e-10)

        # close interpolation
        xc = xc - 1e-5
        xc_full[-1, :] = xc_full[-1, :] - 1e-5
        y_dist = (xc ** 2).reshape(list(m))
        fx = im_model.eval(xc)
        self.assertLessEqual(torch.norm(fx.reshape(list(m)) - y_dist), 1)

    def test_interp_derivative_check_2D(self):
        """ Performs derivative check on Interp1D with 2D input. """
        omega = torch.tensor([0, 5, 0, 3], device=self.device, dtype=torch.float64)
        m = torch.tensor([14, 26], device=self.device)
        h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
        xc_full = get_cell_centered_grid(omega, m, return_all=True).reshape(2, -1)
        xc = (xc_full[-1, :]).reshape(list(m))
        y = (xc ** 2).reshape(list(m))
        xc = xc + 0.2
        dx = torch.randn_like(xc) / h[-1]

        im_model = Interp1D(y, omega, m)

        fx, dfx = im_model.eval(xc, do_derivative=True)
        # dfdx = torch.sum(dfx * dx, dim=0, keepdim=True)
        dfdx = dfx * dx

        zero = []
        one = []
        ratio_zero = []
        ratio_one = []

        num_rounds = 20

        for k in range(num_rounds):
            l = 2 ** (-k)
            ft = im_model.eval(xc + l * dx)

            E0 = torch.norm(fx - ft)
            E1 = torch.norm(fx + l * dfdx - ft)

            zero.append(E0)
            one.append(E1)
            if len(zero) > 1:
                ratio_zero.append(zero[-1] / zero[-2])
                ratio_one.append(one[-1] / one[-2])
                # print("%1.4f\t%1.4f" % (ratio_zero[-1], ratio_one[-1]))

        # check for expected slope
        zero_count = torch.nonzero((torch.abs(torch.tensor(ratio_zero) - 0.5) <= 1e-2)).shape[0]
        one_count = torch.nonzero(torch.tensor(ratio_one) < 0.4).shape[0]

        self.assertGreaterEqual(zero_count, 5)
        self.assertGreaterEqual(one_count, 3)

    def test_interp_derivative_check_3D(self):
        """ Performs derivative check on Interp1D with 3D input. """
        omega = torch.tensor([0, 5, 0, 3, 0, 1], device=self.device, dtype=torch.float64)
        m = torch.tensor([5, 14, 26], device=self.device)
        h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
        xc_full = get_cell_centered_grid(omega, m, return_all=True).reshape(3, -1)
        xc = (xc_full[-1, :]).reshape(list(m))
        y = (xc ** 2).reshape(list(m))
        xc = xc + 0.2
        dx = torch.randn_like(xc) / h[-1]

        im_model = Interp1D(y, omega, m, device=self.device, dtype=self.dtype)

        fx, dfx = im_model.eval(xc, do_derivative=True)
        # dfdx = torch.sum(dfx * dx.unsqueeze(dim=1), dim=1, keepdim=True)
        dfdx = dfx * dx

        zero = []
        one = []
        ratio_zero = []
        ratio_one = []

        num_rounds = 20

        for k in range(num_rounds):
            l = 2 ** (-k)
            ft = im_model.eval(xc + l * dx)

            E0 = torch.norm(fx - ft)
            E1 = torch.norm(fx + l * dfdx - ft)

            zero.append(E0)
            one.append(E1)
            if len(zero) > 1:
                ratio_zero.append(zero[-1] / zero[-2])
                ratio_one.append(one[-1] / one[-2])

        # check for expected slope
        zero_count = torch.nonzero((torch.abs(torch.tensor(ratio_zero) - 0.5) <= 1e-2)).shape[0]
        one_count = torch.nonzero(torch.tensor(ratio_one) < 0.4).shape[0]

        self.assertGreaterEqual(zero_count, 5)
        self.assertGreaterEqual(one_count, 3)

    def test_interp_derivative_check_4D(self):
        """ Performs derivative check on Interp1D with 4D input. """
        omega = torch.tensor([0, 5, 0, 3, 0, 1, 0, 1], device=self.device, dtype=torch.float64)
        m = torch.tensor([5, 14, 26, 4], device=self.device)
        h = (omega[1::2] - omega[:omega.shape[0] - 1:2]) / m
        xc_full = get_cell_centered_grid(omega, m, return_all=True).reshape(4, -1)
        xc = (xc_full[-1, :]).reshape(list(m))
        y = (xc ** 2).reshape(list(m))
        xc = xc + 0.2
        dx = torch.randn_like(xc) / h[-1]

        im_model = Interp1D(y, omega, m, device=self.device, dtype=self.dtype)

        fx, dfx = im_model.eval(xc, do_derivative=True)
        # dfdx = torch.sum(dfx * dx.unsqueeze(dim=1), dim=1, keepdim=True)
        dfdx = dfx * dx

        zero = []
        one = []
        ratio_zero = []
        ratio_one = []

        num_rounds = 20

        for k in range(num_rounds):
            l = 2 ** (-k)
            ft = im_model.eval(xc + l * dx)

            E0 = torch.norm(fx - ft)
            E1 = torch.norm(fx + l * dfdx - ft)

            zero.append(E0)
            one.append(E1)
            if len(zero) > 1:
                ratio_zero.append(zero[-1] / zero[-2])
                ratio_one.append(one[-1] / one[-2])

        # check for expected slope
        zero_count = torch.nonzero((torch.abs(torch.tensor(ratio_zero) - 0.5) <= 1e-2)).shape[0]
        one_count = torch.nonzero(torch.tensor(ratio_one) < 0.4).shape[0]

        self.assertGreaterEqual(zero_count, 5)
        self.assertGreaterEqual(one_count, 3)


if __name__ == '__main__':
    unittest.main()
