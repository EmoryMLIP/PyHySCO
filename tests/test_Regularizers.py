import unittest
from EPI_MRI.Regularizers import *


class TestRegularizers(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(81)  # reproducibility with randomness
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def test_tik_reg(self):
        """ Performs derivative check on eval of TikRegularizer. """
        m = torch.tensor([2, 5, 6], device=self.device)
        S = TikRegularizer(torch.tensor([0, 1, 0, 1, 0, 1], device=self.device, dtype=self.dtype), m)
        xc = torch.randn(list(m_plus(m)), dtype=self.dtype, device=self.device).reshape(-1, 1)
        y = torch.randn_like(xc)
        dx = torch.randn_like(xc)
        dx = dx / torch.norm(dx)

        fx, dfx, H = S.eval(xc, 15.0, y=y, do_derivative=True)
        dfdx = torch.sum(dfx * dx,dim=0,keepdim=True)
        Hx = torch.sum(H.mat_mul(dx), dim=0, keepdim=True)

        zero = []
        one = []
        two = []
        ratio_zero = []
        ratio_one = []
        ratio_two = []

        num_rounds = 20

        for k in range(num_rounds):
            h = 2 ** (-k)
            ft = S.eval(xc + h * dx, 15.0, y=y, do_derivative=False)[0]

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

    def test_quad_reg(self):
        """ Performs derivative check on eval of QuadRegularizer. """
        m = torch.tensor([2, 5, 6], device=self.device)
        L_xyz = myLaplacian3D(torch.tensor([0, 1, 0, 1, 0, 1], device=self.device, dtype=self.dtype), m, self.dtype, self.device)
        S = QuadRegularizer(L_xyz)
        xc = torch.randn(list(m_plus(m)), dtype=self.dtype, device=self.device).reshape(-1, 1)
        y = torch.randn_like(xc)
        dx = torch.randn_like(xc)
        dx = dx / torch.norm(dx)

        fx, dfx, H = S.eval(xc, do_derivative=True)
        dfdx = torch.sum(dfx.reshape(-1, 1) * dx, dim=0, keepdim=True)
        Hx = torch.sum(H.mat_mul(dx), dim=0, keepdim=True)

        zero = []
        one = []
        two = []
        ratio_zero = []
        ratio_one = []
        ratio_two = []

        num_rounds = 20

        for k in range(num_rounds):
            h = 2 ** (-k)
            ft = S.eval(xc + h * dx, do_derivative=False)[0]

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

    # def test_tik_reg_prox_single(self):
    #     """ Performs derivative check on eval_single of TikRegularizerProximal. """
    #     m = torch.tensor([2, 5, 6], device=self.device)
    #     L_xyz = Laplacian3D(m, torch.tensor([0, 1, 0, 1, 0, 1], device=self.device, dtype=self.dtype), self.dtype,
    #                         self.device)
    #     S = TikRegularizerProximal(L_xyz)
    #     xc = torch.randn(list(m_plus(m)), dtype=self.dtype, device=self.device).reshape(-1, 1)
    #     y = torch.randn_like(xc)
    #     dx = torch.randn_like(xc)
    #     dx = dx / torch.norm(dx)
    #
    #     fx, dfx, H = S.eval_single(L_xyz, xc, yref=y, do_derivative=True)
    #     dfdx = torch.sum(dfx * dx, dim=0, keepdim=True)
    #     Hx = torch.sum(H(dx), dim=0, keepdim=True)
    #
    #     zero = []
    #     one = []
    #     two = []
    #     ratio_zero = []
    #     ratio_one = []
    #     ratio_two = []
    #
    #     num_rounds = 20
    #
    #     for k in range(num_rounds):
    #         h = 2 ** (-k)
    #         ft = S.eval_single(L_xyz, xc + h * dx, yref=y, do_derivative=False)[0]
    #
    #         E0 = torch.norm(fx - ft).cpu()
    #         E1 = torch.norm(fx + h * dfdx - ft).cpu()
    #         E2 = torch.norm(fx + h * dfdx + h ** 2 * Hx - ft).cpu()
    #
    #         zero.append(E0)
    #         one.append(E1)
    #         two.append(E2)
    #         if len(zero) > 1:
    #             ratio_zero.append(zero[-1] / zero[-2])
    #             ratio_one.append(one[-1] / one[-2])
    #             ratio_two.append(two[-1] / two[-2])
    #
    #     # check for expected slope
    #     zero_count = torch.nonzero((torch.abs(torch.tensor(ratio_zero) - 0.5) <= 1e-2)).shape[0]
    #     one_count = torch.nonzero(torch.tensor(ratio_one) < 0.25).shape[0]
    #     two_count = torch.nonzero(torch.tensor(ratio_two) < 0.25).shape[0]
    #
    #     self.assertGreaterEqual(zero_count, 5)
    #     self.assertGreaterEqual(one_count, 5)
    #     self.assertGreaterEqual(two_count, 5)

    def test_quad_reg_prox_solve(self):
        """ Tests proximal_solve method of QuadRegularizer. """
        m = torch.tensor([12, 5, 6], device=self.device)
        L_xyz = FFT3D(getLaplacianStencil(torch.tensor([0, 1, 0, 1, 0, 1], device=self.device, dtype=self.dtype), m, self.dtype, self.device)[0], m)
        S = QuadRegularizer(L_xyz)
        y = torch.randn(list(m_plus(m)), dtype=self.dtype, device=self.device)
        rho = 7.5
        x = S.prox_solve(y, rho)

        # calculate derivative to see if x gives critical point of original function:
        # 1/2*||L(x)||**2 + rho/2*||x-y||**2
        _, Hx, _ = S.eval(x, do_derivative=True)
        dfx = Hx + rho*x - rho*y
        self.assertLessEqual(torch.norm(dfx), 1e-5)


# run tests when this file is called
if __name__ == '__main__':
    unittest.main()
