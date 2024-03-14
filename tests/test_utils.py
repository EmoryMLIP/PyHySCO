import unittest
import os
from EPI_MRI.utils import *


class TestUtils(unittest.TestCase):
	""" Unit test for EPI_MRI.utils.py"""

	def setUp(self):
		# Initialize any common data or setup needed for the tests.
		torch.manual_seed(81)  # reproducibility with randomness
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.dtype = torch.float64

	def test_load_data(self):
		""" Test function load_data. """
		# TOPUP style input 2D
		# Create a synthetic NIfTI image in memory
		shape = (12, 10, 2)
		image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file = os.getcwd() + 'test.nii.gz'
		img = nib.Nifti1Image(image_data, np.eye(4))
		nib.save(img, img_file)
		# Call the load_data method with the dummy image file (PED = 1)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file, phase_encoding_direction=1)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data[:, :, 0], dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data[:, :, 1], dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 10.0, 0.0, 12.0], dtype=torch.float64)
		expected_m = torch.tensor([10, 12], dtype=torch.int)
		expected_h = torch.tensor([1, 1], dtype=torch.float64)
		expected_permute_back = [1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Call the load_data method with the dummy image file (PED = 2)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file, phase_encoding_direction=2)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data[:, :, 0], dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data[:, :, 1], dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 12.0, 0.0, 10.0], dtype=torch.float64)
		expected_m = torch.tensor([12, 10], dtype=torch.int)
		expected_h = torch.tensor([1, 1], dtype=torch.float64)
		expected_permute_back = [0, 1]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)

		# TOPUP style input 3D
		# Create a synthetic NIfTI image in memory
		shape = (12, 10, 5, 2)
		image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file = os.getcwd() + 'test.nii.gz'
		img = nib.Nifti1Image(image_data, np.eye(4))
		nib.save(img, img_file)
		# Call the load_data method with the dummy image file (PED = 1)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file, phase_encoding_direction=1)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data[:, :, :, 0], dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data[:, :, :, 1], dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 5.0, 0.0, 10.0, 0.0, 12.0], dtype=torch.float64)
		expected_m = torch.tensor([5, 10, 12], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [2, 1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Call the load_data method with the dummy image file (PED = 2)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file, phase_encoding_direction=2)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data[:, :, :, 0], dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data[:, :, :, 1], dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 5.0, 0.0, 12.0, 0.0, 10.0], dtype=torch.float64)
		expected_m = torch.tensor([5, 12, 10], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [1, 2, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)

		# TOPUP style input 4D
		# Create a synthetic NIfTI image in memory
		shape = (12, 10, 5, 9,  2)
		image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file = os.getcwd() + 'test.nii.gz'
		img = nib.Nifti1Image(image_data, np.eye(4))
		nib.save(img, img_file)
		# Call the load_data method with the dummy image file (PED = 1)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file, phase_encoding_direction=1)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data[:, :, :, :, 0], dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data[:, :, :, :, 1], dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 9.0, 0.0, 5.0, 0.0, 10.0, 0.0, 12.0], dtype=torch.float64)
		expected_m = torch.tensor([9, 5, 10, 12], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [3, 2, 1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Call the load_data method with the dummy image file (PED = 2)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file, phase_encoding_direction=2)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data[:, :, :, :, 0], dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data[:, :, :, :, 1], dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 9.0, 0.0, 5.0, 0.0, 12.0, 0.0, 10.0], dtype=torch.float64)
		expected_m = torch.tensor([9, 5, 12, 10], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [2, 3, 1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Delete the temporary file
		os.remove(img_file)

		# two file input 2D
		# Create a synthetic NIfTI image in memory
		shape = (12, 10)
		image_data1 = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file1 = os.getcwd() + 'test1.nii.gz'
		img1 = nib.Nifti1Image(image_data1, np.eye(4))
		nib.save(img1, img_file1)
		image_data2 = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file2 = os.getcwd() + 'test2.nii.gz'
		img2 = nib.Nifti1Image(image_data2, np.eye(4))
		nib.save(img2, img_file2)
		# Call the load_data method with the dummy image file (PED = 1)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file1, img_file2, phase_encoding_direction=1)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data1, dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data2, dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 10.0, 0.0, 12.0], dtype=torch.float64)
		expected_m = torch.tensor([10, 12], dtype=torch.int)
		expected_h = torch.tensor([1, 1], dtype=torch.float64)
		expected_permute_back = [1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Call the load_data method with the dummy image file (PED = 2)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file1, img_file2, phase_encoding_direction=2)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data1, dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data2, dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 12.0, 0.0, 10.0], dtype=torch.float64)
		expected_m = torch.tensor([12, 10], dtype=torch.int)
		expected_h = torch.tensor([1, 1], dtype=torch.float64)
		expected_permute_back = [0, 1]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)

		# two file input 3D
		# Create a synthetic NIfTI image in memory
		shape = (12, 10, 5)
		image_data1 = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file1 = os.getcwd() + 'test1.nii.gz'
		img1 = nib.Nifti1Image(image_data1, np.eye(4))
		nib.save(img1, img_file1)
		image_data2 = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file2 = os.getcwd() + 'test2.nii.gz'
		img2 = nib.Nifti1Image(image_data2, np.eye(4))
		nib.save(img2, img_file2)
		# Call the load_data method with the dummy image file (PED = 1)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file1, img_file2, phase_encoding_direction=1)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data1, dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data2, dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 5.0, 0.0, 10.0, 0.0, 12.0], dtype=torch.float64)
		expected_m = torch.tensor([5, 10, 12], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [2, 1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)

		# Call the load_data method with the dummy image file (PED = 2)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file1, img_file2, phase_encoding_direction=2)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data1, dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data2, dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 5.0, 0.0, 12.0, 0.0, 10.0], dtype=torch.float64)
		expected_m = torch.tensor([5, 12, 10], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [1, 2, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)

		# two file input 4D
		# Create a synthetic NIfTI image in memory
		shape = (12, 10, 5, 9)
		image_data1 = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file1 = os.getcwd() + 'test1.nii.gz'
		img1 = nib.Nifti1Image(image_data1, np.eye(4))
		nib.save(img1, img_file1)
		image_data2 = np.asarray(torch.randn(*shape, dtype=self.dtype))
		img_file2 = os.getcwd() + 'test2.nii.gz'
		img2 = nib.Nifti1Image(image_data2, np.eye(4))
		nib.save(img2, img_file2)
		# Call the load_data method with the dummy image file (PED = 1)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file1, img_file2, phase_encoding_direction=1)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data1, dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data2, dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 9.0, 0.0, 5.0, 0.0, 10.0, 0.0, 12.0], dtype=torch.float64)
		expected_m = torch.tensor([9, 5, 10, 12], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [3, 2, 1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Call the load_data method with the dummy image file (PED = 2)
		rho0, rho1, omega, m, h, permute_back = load_data(img_file1, img_file2, phase_encoding_direction=2)
		# Define the expected values
		expected_rho0 = torch.tensor(image_data1, dtype=torch.float64)
		expected_rho1 = torch.tensor(image_data2, dtype=torch.float64)
		expected_omega = torch.tensor([0.0, 9.0, 0.0, 5.0, 0.0, 12.0, 0.0, 10.0], dtype=torch.float64)
		expected_m = torch.tensor([9, 5, 12, 10], dtype=torch.int)
		expected_h = (expected_omega[1::2] - expected_omega[:expected_omega.shape[0] - 1:2]) / expected_m
		expected_permute_back = [2, 3, 1, 0]
		# Check if the returned values match the expected ones
		self.assertTrue(torch.allclose(rho0.permute(permute_back), expected_rho0))
		self.assertTrue(torch.allclose(rho1.permute(permute_back), expected_rho1))
		self.assertTrue(torch.allclose(omega, expected_omega))
		self.assertTrue(torch.all(torch.eq(m, expected_m)))
		self.assertTrue(torch.allclose(h, expected_h))
		self.assertEqual(permute_back, expected_permute_back)
		# Delete the temporary files
		os.remove(img_file1)
		os.remove(img_file2)

	def test_normalize(self):
		""" Test method normalize """
		# Test the normalize function with sample inputs (2D).
		im1 = torch.tensor([[-1.0, 100.0], [350.0, 200.0]], device=self.device, dtype=self.dtype)
		im2 = torch.tensor([[100.0, 0.0], [500.0, -50.0]], device=self.device, dtype=self.dtype)
		i1, i2 = normalize(im1, im2)
		# Assertions to check if the normalized outputs are within the expected range and shape is unchanged.
		self.assertEqual(im1.shape, i1.shape)
		self.assertEqual(im1.shape, i2.shape)
		self.assertTrue(torch.min(i1) >= 0 and torch.max(i1) <= 256)
		self.assertTrue(torch.min(i2) >= 0 and torch.max(i2) <= 256)

		# Test the normalize function with sample inputs (3D).
		im1 = torch.tensor([[-1.0, 100.0], [350.0, 200.0], [257.5, 15.2]], device=self.device, dtype=self.dtype)
		im2 = torch.tensor([[100.0, 0.0], [500.0, -50.0], [-150.0, 16.7]], device=self.device, dtype=self.dtype)
		i1, i2 = normalize(im1, im2)
		# Assertions to check if the normalized outputs are within the expected range and shape is unchanged.
		self.assertEqual(im1.shape, i1.shape)
		self.assertEqual(im1.shape, i2.shape)
		self.assertTrue(torch.min(i1) >= 0 and torch.max(i1) <= 256)
		self.assertTrue(torch.min(i2) >= 0 and torch.max(i2) <= 256)

		# Test the normalize function with sample inputs (4D).
		im1 = torch.tensor([[-1.0, 100.0], [350.0, 200.0], [257.5, 15.2], [-1.0, 42.5]], device=self.device, dtype=self.dtype)
		im2 = torch.tensor([[100.0, 0.0], [500.0, -50.0], [-150.0, 16.7], [260.7, 2.0]], device=self.device, dtype=self.dtype)
		i1, i2 = normalize(im1, im2)
		# Assertions to check if the normalized outputs are within the expected range and shape is unchanged.
		self.assertEqual(im1.shape, i1.shape)
		self.assertEqual(im1.shape, i2.shape)
		self.assertTrue(torch.min(i1) >= 0 and torch.max(i1) <= 256)
		self.assertTrue(torch.min(i2) >= 0 and torch.max(i2) <= 256)

	def test_m_plus_and_m_minus(self):
		""" Test the m_plus and m_minus functions with sample inputs. """
		# 2D
		m = torch.tensor([256, 256], device=self.device)
		# Test m_plus
		m2 = m_plus(m)
		self.assertEqual(m2.tolist(), [256, 257])
		# Test m_minus
		m2 = m_minus(m2)
		self.assertEqual(m2.tolist(), [256, 256])

		# 3D
		m = torch.tensor([132, 200, 200], device=self.device)
		# Test m_plus
		m2 = m_plus(m)
		self.assertEqual(m2.tolist(), [132, 200, 201])
		# Test m_minus
		m2 = m_minus(m2)
		self.assertEqual(m2.tolist(), [132, 200, 200])

		# 4D
		m = torch.tensor([95, 132, 200, 200], device=self.device)
		# Test m_plus
		m2 = m_plus(m)
		self.assertEqual(m2.tolist(), [95, 132, 200, 201])
		# Test m_minus
		m2 = m_minus(m2)
		self.assertEqual(m2.tolist(), [95, 132, 200, 200])

	def test_interp_parallel_1D(self):
		""" Tests for method interp_parallel with 1D input """
		x = torch.linspace(0, 6, 80).reshape((1, -1))
		y = x.sin()
		xs = torch.linspace(0, 6, 101).reshape((1, -1))

		# interpolation on same points should yield original data
		ys_same = interp_parallel(x, y, x)
		self.assertLessEqual(torch.norm(y - ys_same), 1e-10)

		# check gives close interpolation of sin
		ys_p = interp_parallel(x, y, xs)
		self.assertLessEqual(torch.norm(ys_p.squeeze()-xs.sin().squeeze()), 1e-2)

	def test_interp_parallel_2D(self):
		""" Tests for method interp_parallel with 2D input """
		m = torch.tensor([196, 256], dtype=torch.int)
		x = torch.linspace(0, m[1], m[1] + 1).reshape(1, -1)
		x = x.expand(m[0], -1)
		x_sparse = x[:, ::2]
		y = torch.zeros((m[0], m[1] + 1), dtype=torch.float64)
		y[100:150, 100:170] = 1

		# interpolation on same points should yield original data
		ys_same = interp_parallel(x, y, x)
		self.assertLessEqual(torch.norm(y - ys_same), 1e-10)

		# linear interpolation should be exact (up to boundary)
		y = 5 * x + 4
		y_sparse = y[:, ::2]
		ys = interp_parallel(x_sparse, y_sparse, x)
		self.assertLessEqual(torch.norm(ys[:-1, :-1] - y[:-1, :-1]), 1e-2)

	def test_grid_2D(self):
		""" Tests for a 2-D cell-centered grid. """
		omega = torch.tensor([0, 2, 0, 1], dtype=torch.float64)
		m = torch.tensor([2, 7], dtype=torch.int)
		xc = get_cell_centered_grid(omega, m, return_all=True)

		# check number of values = prod(m1)*2 [2 coordinates per point]
		self.assertEqual(xc.shape[0], torch.prod(m)*2)

		# get grid lines of cells
		xc = xc.reshape(2, -1)
		grid_lines_x = torch.tensor([i/(m[0]/omega[1]) for i in range(m[0]+1)])
		grid_lines_y = torch.tensor([i/(m[1]/omega[3]) for i in range(m[1]+1)])

		# check cell-centered values lie in center of cells
		# y-coordinates
		for i in range(m[1]):
			self.assertLessEqual(torch.norm(xc[1, i]-(grid_lines_y[i+1]+grid_lines_y[i])/2), 1e-7)
			for k in range(1, m[0]):
				self.assertEqual(xc[1, i], xc[1, k*m[1]+i])

		# x-coordinates
		for j in range(m[0]*m[1]):
			self.assertLessEqual(torch.norm(xc[0, j]-(grid_lines_x[int(j/m[1])+1]+grid_lines_x[int(j/m[1])])/2), 1e-7)

		# return_all = False only gives points in distortion dimension (last dimension)
		xc_single = get_cell_centered_grid(omega, m, return_all=False)
		self.assertLessEqual(torch.norm(xc[-1, :].squeeze() - xc_single.squeeze()), 1e-5)

	def test_grid_3D(self):
		""" Tests for a 3-D cell-centered grid. """
		omega = torch.tensor([0, 1, 0, 2, 0, 1], dtype=torch.float64)
		m = torch.tensor([3, 5, 4], dtype=torch.int)
		xc = get_cell_centered_grid(omega, m, return_all=True)

		# check number of values = prod(m2)*3 [3 coordinates per point]
		self.assertEqual(xc.shape[0], torch.prod(m)*3)

		# get grid lines of cells
		xc = xc.reshape(3, -1)
		grid_lines_x = torch.tensor([i/(m[0]/omega[1]) for i in range(m[0]+1)])
		grid_lines_y = torch.tensor([i/(m[1]/omega[3]) for i in range(m[1]+1)])
		grid_lines_z = torch.tensor([i/(m[2]/omega[5]) for i in range(m[2]+1)])

		# check cell-centered values lie in center of cells
		# y-coordinates
		for i in range(m[1]):
			self.assertLessEqual(torch.norm(xc[1, i*m[2]]-(grid_lines_y[i+1]+grid_lines_y[i])/2), 1e-7)
			for k in range(0, m[0]):
				for l in range(0, m[2]):
					self.assertEqual(xc[1, i*m[2]], xc[1, k*m[1]*m[2]+i*m[2]+l])

		# x-coordinates
		for j in range(m[0]*m[1]*m[2]):
			self.assertLessEqual(torch.norm(xc[0, j]-(grid_lines_x[int(j/(m[1]*m[2]))+1]+grid_lines_x[int(j/(m[1]*m[2]))])/2), 1e-7)

		# z-coordinates
		for i in range(m[2]):
			self.assertLessEqual(torch.norm(xc[2, i]-(grid_lines_z[i+1]+grid_lines_z[i])/2), 1e-7)
			for j in range(m[0]*m[1]):
				self.assertEqual(xc[2, i], xc[2, i+j*m[2]])

		# return_all = False only gives points in distortion dimension (last dimension)
		xc_single = get_cell_centered_grid(omega, m, return_all=False)
		self.assertLessEqual(torch.norm(xc[-1, :].squeeze() - xc_single.squeeze()), 1e-5)

	def test_grid_4D(self):
		""" Tests for a 4-D cell-centered grid. """
		omega = torch.tensor([0, 1, 0, 2, 0, 1, 0, 2], dtype=torch.float64)
		m = torch.tensor([3, 5, 4, 7], dtype=torch.int)
		xc = get_cell_centered_grid(omega, m, return_all=True)

		# check number of values = prod(m2)*4 [4 coordinates per point]
		self.assertEqual(xc.shape[0], torch.prod(m)*4)

		# get grid lines of cells
		xc = xc.reshape(4, -1)
		grid_lines_x = torch.tensor([i/(m[0]/omega[1]) for i in range(m[0]+1)])
		grid_lines_y = torch.tensor([i/(m[1]/omega[3]) for i in range(m[1]+1)])
		grid_lines_z = torch.tensor([i/(m[2]/omega[5]) for i in range(m[2]+1)])
		grid_lines_w = torch.tensor([i/(m[3]/omega[7]) for i in range(m[3]+1)])

		# check cell-centered values lie in center of cells
		# y-coordinates
		for i in range(m[1]):
			self.assertLessEqual(torch.norm(xc[1, i*m[2]*m[3]]-(grid_lines_y[i+1]+grid_lines_y[i])/2), 1e-7)
			for k in range(0, m[0]):
				for l in range(0, m[2]):
					self.assertEqual(xc[1, i*m[2]*m[3]], xc[1, k*m[1]*m[2]*m[3]+i*m[2]*m[3]+l])

		# x-coordinates
		for j in range(m[0]*m[1]*m[2]*m[3]):
			self.assertLessEqual(torch.norm(xc[0, j]-(grid_lines_x[int(j/(m[1]*m[2]*m[3]))+1]+grid_lines_x[int(j/(m[1]*m[2]*m[3]))])/2), 1e-7)

		# z-coordinates
		for i in range(m[2]):
			self.assertLessEqual(torch.norm(xc[2, i*m[3]]-(grid_lines_z[i+1]+grid_lines_z[i])/2), 1e-7)
			for j in range(m[0]*m[1]):
				self.assertEqual(xc[2, i], xc[2, i+j*m[2]*m[3]])

		# w-coordinates
		for i in range(m[3]):
			self.assertLessEqual(torch.norm(xc[3, i]-(grid_lines_w[i+1]+grid_lines_w[i])/2), 1e-7)
			for j in range(m[0]*m[1]*m[2]):
				self.assertEqual(xc[3, i], xc[3, i+j*m[3]])

		# return_all = False only gives points in distortion dimension (last dimension)
		xc_single = get_cell_centered_grid(omega, m, return_all=False)
		self.assertLessEqual(torch.norm(xc[-1, :].squeeze() - xc_single.squeeze()), 1e-5)


# run tests when this file is called
if __name__ == '__main__':
	unittest.main()
