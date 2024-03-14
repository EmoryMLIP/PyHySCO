import unittest
from EPI_MRI.LinearOperators import *


class TestLinearOperators(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(81)  # reproducibility with randomness
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def test_avg1dconv(self):
        """ Test averaging operator as defined in myAvg1D."""
        # 2D
        m1 = torch.tensor((5, 7), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 1], dtype=self.dtype, device=self.device)
        img1a = torch.randn(list(m1), dtype=self.dtype, device=self.device) * 500
        A = myAvg1D(omega, m1, self.dtype, self.device)
        # A = Avg1DConv(m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = A.mat_mul(img)
        # check shape of output
        self.assertEqual(res.shape[0], 5)
        self.assertEqual(res.shape[1], 7)
        # transp_mat_mul
        resT = A.transp_mat_mul(img1a)
        self.assertEqual(resT.shape[0], 5)
        self.assertEqual(resT.shape[1], 8)
        # adjoint test
        s = torch.tensor([3, 4], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        A = myAvg1D(omega, s, self.dtype, self.device)
        u = torch.randn(list(s), dtype=torch.float64)
        one = A.mat_mul(v).reshape(-1, 1)
        two = A.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 3D
        m1 = torch.tensor((5, 7, 3), dtype=torch.int, device=self.device)
        img1a = torch.randn(list(m1), dtype=self.dtype, device=self.device) * 500
        omega = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=self.dtype, device=self.device)
        A = myAvg1D(omega, m1, self.dtype, self.device)
        # A = Avg1DConv(m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = A.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m1))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = A.transp_mat_mul(img1a).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4, 2], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        A = myAvg1D(omega, s, self.dtype, self.device)
        u = torch.randn(list(s), dtype=torch.float64)
        one = A.mat_mul(v).reshape(-1, 1)
        two = A.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 4D
        m1 = torch.tensor((5, 7, 3, 2), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=self.dtype, device=self.device)
        img1a = torch.randn(list(m1), dtype=self.dtype, device=self.device) * 500
        A = myAvg1D(omega, m1, self.dtype, self.device)
        # A = Avg1DConv(m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = A.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m1))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = A.transp_mat_mul(img1a).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4, 2, 3], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        A = myAvg1D(omega, s, self.dtype, self.device)
        u = torch.randn(list(s), dtype=torch.float64)
        one = A.mat_mul(v).reshape(-1, 1)
        two = A.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

    def test_diff1dconv(self):
        """ Test derivative operator as defined in myDiff1D."""
        # 2D
        m1 = torch.tensor((5, 7), dtype=torch.int, device=self.device)
        img1a = torch.randn(list(m1), dtype=self.dtype, device=self.device) * 500
        omega = torch.tensor([0, 1, 0, 2], dtype=self.dtype, device=self.device)
        D = myDiff1D(omega,m1, self.dtype, self.device)
        # D = Diff1DConv(omega, m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = D.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m1))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = D.transp_mat_mul(img1a).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        D = myDiff1D(omega,s, self.dtype, self.device)
        # D = Diff1DConv(omega, s, self.dtype, self.device)
        u = torch.randn(list(s), dtype=torch.float64)
        one = D.mat_mul(v).reshape(-1, 1)
        two = D.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 3D
        m1 = torch.tensor((5, 7, 3), dtype=torch.int, device=self.device)
        img1a = torch.randn(list(m1), dtype=self.dtype, device=self.device) * 500
        omega = torch.tensor([0, 1, 0, 2, 0, 2], dtype=self.dtype, device=self.device)
        D = myDiff1D(omega,m1, self.dtype, self.device)
        # D = Diff1DConv(omega, m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = D.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m1))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = D.transp_mat_mul(img1a).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4, 2], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        D = myDiff1D(omega, s, self.dtype, self.device)
        u = torch.randn(list(s), dtype=torch.float64)
        one = D.mat_mul(v).reshape(-1, 1)
        two = D.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 4D
        m1 = torch.tensor((5, 7, 3, 2), dtype=torch.int, device=self.device)
        img1a = torch.randn(list(m1), dtype=self.dtype, device=self.device) * 500
        omega = torch.tensor([0, 1, 0, 2, 0, 2, 0, 1], dtype=self.dtype, device=self.device)
        D = myDiff1D(omega,m1, self.dtype, self.device)
        # D = Diff1DConv(omega, m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = D.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m1))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = D.transp_mat_mul(img1a).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4, 2, 3], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        D = myDiff1D(omega, s, self.dtype, self.device)
        u = torch.randn(list(s), dtype=torch.float64)
        one = D.mat_mul(v).reshape(-1, 1)
        two = D.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

    def test_lap3D(self):
        """ Test 3D Laplacian operator as defined in myLaplacian3D."""
        # 2D
        # m1 = torch.tensor((5, 7), dtype=torch.int, device=self.device)
        # omega = torch.tensor([0, 1, 0, 2], dtype=self.dtype, device=self.device)
        # L = myLaplacian3D(omega, m1, self.dtype, self.device)
        # # L = Laplacian3D(m1, omega, self.dtype, self.device)
        # # mat_mul
        # img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        # res = L.mat_mul(img).reshape(-1, 1)
        # print(L.mat_mul(img).shape)
        # # check shape of output
        # self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        # self.assertEqual(res.shape[1], 1)
        # # transp_mat_mul
        # resT = L.transp_mat_mul(res).reshape(-1, 1)
        # self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        # self.assertEqual(resT.shape[1], 1)
        # # adjoint test
        # s = torch.tensor([3, 4], dtype=torch.int)
        # v = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=self.dtype, device=self.device)
        # L = myLaplacian3D(s,omega, self.dtype, self.device)
        # # L = Laplacian3D(s, omega, self.dtype, self.device)
        # u = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=torch.float64)
        # one = L.mat_mul(v)
        # two = L.transp_mat_mul(u)
        # one = torch.mm(u.T, one)
        # two = torch.mm(v.T, two)
        # self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 3D
        m1 = torch.tensor((5, 7, 3), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 2, 0, 2], dtype=self.dtype, device=self.device)
        L = myLaplacian3D(omega, m1, self.dtype, self.device)
        # L = Laplacian3D(m1, omega, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = L.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = L.transp_mat_mul(res).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4, 2], dtype=torch.int)
        v = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=self.dtype, device=self.device)
        L = myLaplacian3D(omega, s, self.dtype, self.device)
        # Lt = Laplacian3D(s, omega, self.dtype, self.device)
        u = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=torch.float64)
        one = L.mat_mul(v).reshape(-1, 1)
        # onet = L.transp_mat_mul(v)
        # self.assertAlmostEqual(torch.norm(one - onet).item(), 0)
        two = L.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

    def test_lap2D(self):
        """ Test 2D Laplacian operator as defined in myLaplacian2D."""
        # # 2D
        # m1 = torch.tensor((5, 7), dtype=torch.int, device=self.device)
        # omega = torch.tensor([0, 1, 0, 2], dtype=self.dtype, device=self.device)
        # L = myLaplacian2D(omega, m1, self.dtype, self.device)
        # # mat_mul
        # img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        # res = L.mat_mul(img).reshape(-1, 1)
        # # check shape of output
        # self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        # self.assertEqual(res.shape[1], 1)
        # # transp_mat_mul
        # resT = L.transp_mat_mul(res).reshape(-1, 1)
        # self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        # self.assertEqual(resT.shape[1], 1)
        # # adjoint test
        # s = torch.tensor([3, 4], dtype=torch.int)
        # v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        # L = myLaplacian2D(omega, s, self.dtype, self.device)
        # u = torch.randn(list(m_plus(s)), dtype=torch.float64)
        # one = L.mat_mul(v).reshape(-1, 1)
        # two = L.transp_mat_mul(u).reshape(-1, 1)
        # one = torch.mm(u.reshape(-1, 1).T, one)
        # two = torch.mm(v.reshape(-1, 1).T, two)
        # self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 3D
        m1 = torch.tensor((5, 7, 4), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 2, 0, 2], dtype=self.dtype, device=self.device)
        L = myLaplacian2D(omega, m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = L.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = L.transp_mat_mul(img).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([5, 4, 7], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        L = myLaplacian2D(omega, s, self.dtype, self.device)
        u = torch.randn(list(m_plus(s)), dtype=torch.float64)
        one = L.mat_mul(v).reshape(-1, 1)
        two = L.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

    def test_lap1D(self):
        """ Test 1D Laplacian operator as defined in myLaplacian1D."""
        # # 2D
        # m1 = torch.tensor((5, 7), dtype=torch.int, device=self.device)
        # omega = torch.tensor([0, 1, 0, 2], dtype=self.dtype, device=self.device)
        # L = Laplacian1D(m1, omega, self.dtype, self.device)
        # Lt = myLaplacian1D(m1, omega, self.dtype, self.device)

        # # mat_mul
        # img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        # res = L.mat_mul(img)
        # rest = Lt.mat_mul(img)
        # self.assertAlmostEqual(torch.norm(res - rest).item(), 0)
        # # check shape of output
        # self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        # self.assertEqual(res.shape[1], 1)
        # # transp_mat_mul
        # resT = L.transp_mat_mul(res)
        # self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        # self.assertEqual(resT.shape[1], 1)
        # # adjoint test
        # s = torch.tensor([3, 4], dtype=torch.int)
        # v = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=self.dtype, device=self.device)
        # L = Laplacian1D(s, omega, self.dtype, self.device)
        # u = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=torch.float64)
        # one = L.mat_mul(v)
        # two = L.transp_mat_mul(u)
        # one = torch.mm(u.T, one)
        # two = torch.mm(v.T, two)
        # self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 3D
        m1 = torch.tensor((5, 7, 4), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 2, 0, 2], dtype=self.dtype, device=self.device)
        L = myLaplacian1D(omega, m1, self.dtype, self.device)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = L.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = L.transp_mat_mul(img).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([5, 4, 6], dtype=torch.int)
        v = torch.randn(list(m_plus(s)), dtype=self.dtype, device=self.device)
        L = myLaplacian1D(omega, s, self.dtype, self.device)
        u = torch.randn(list(m_plus(s)), dtype=torch.float64)
        one = L.mat_mul(v).reshape(-1, 1)
        two = L.transp_mat_mul(u).reshape(-1, 1)
        one = torch.mm(u.reshape(-1, 1).T, one)
        two = torch.mm(v.reshape(-1, 1).T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

    def test_identity(self):
        """ Test Identity operator."""
        # 2D
        m1 = torch.tensor((5, 7), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 2], dtype=self.dtype, device=self.device)
        L = Identity(2.7)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = L.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = L.transp_mat_mul(res).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4], dtype=torch.int)
        v = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=self.dtype, device=self.device)
        L = Identity(2.7)
        u = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=torch.float64)
        one = L.mat_mul(v)
        two = L.transp_mat_mul(u)
        one = torch.mm(u.T, one)
        two = torch.mm(v.T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

        # 3D
        m1 = torch.tensor((5, 7, 3), dtype=torch.int, device=self.device)
        omega = torch.tensor([0, 1, 0, 2, 0, 2], dtype=self.dtype, device=self.device)
        L = Identity(2.7)
        # mat_mul
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = L.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(res.shape[1], 1)
        # transp_mat_mul
        resT = L.transp_mat_mul(res).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # adjoint test
        s = torch.tensor([3, 4, 2], dtype=torch.int)
        v = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=self.dtype, device=self.device)
        L = Identity(2.7)
        u = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=torch.float64)
        one = L.mat_mul(v)
        two = L.transp_mat_mul(u)
        one = torch.mm(u.T, one)
        two = torch.mm(v.T, two)
        self.assertLessEqual(torch.norm(one - two).item(), 1e-5)

    def test_convfft(self):
        """ Test 3D convolution using FFTs."""
        # 3D
        m1 = torch.tensor((5, 7, 4), dtype=torch.int, device=self.device)
        K = torch.randn(3, 3, 3, dtype=self.dtype, device=self.device)
        L = FFT3D(K, m1)
        # forward
        img = torch.randn(list(m_plus(m1)), dtype=torch.float64)
        res = L.mat_mul(img).reshape(-1, 1)
        # check shape of output
        self.assertEqual(res.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(res.shape[1], 1)
        # equivalent to circular convolution

        def conv(x, K):
            dim = torch.numel(m1)
            shape = [1, 1, 1, 1, 1]
            shape[-dim:] = list(m_plus(m1))
            padding = [1, 1, 1, 1, 1, 1]
            return torch.nn.functional.conv3d(torch.nn.functional.pad(x.reshape(shape), pad=padding, mode='circular'),
                                              K.unsqueeze(dim=0).unsqueeze(dim=0), padding=0).contiguous().view(-1, 1)

        self.assertLessEqual(torch.norm(res) - torch.norm(conv(img, K)), 1e-5)
        # inverse
        resT = L.inv(img, 1.0).reshape(-1, 1)
        self.assertEqual(resT.shape[0], torch.prod(m_plus(m1)))
        self.assertEqual(resT.shape[1], 1)
        # inverse test
        s = torch.tensor([5, 4, 6], dtype=torch.int)
        v = torch.randn(torch.prod(m_plus(s)).item(), 1, dtype=self.dtype, device=self.device)
        L = FFT3D(K, s)
        self.assertLessEqual(torch.norm(v - L.inv(L.mat_mul(v.reshape(list(m_plus(s)))), 0.0).reshape(-1, 1)), 1e-5)


# run tests when this file is called
if __name__ == '__main__':
    unittest.main()
