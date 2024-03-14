import unittest
from tests.test_utils import TestUtils 
from tests.test_ImageModels import TestImageModels
from tests.test_Regularizers import TestRegularizers
from tests.test_LinearOperators import TestLinearOperators
from tests.test_Preconditioners import TestPreconditioners
from tests.test_InitializationMethods import TestInitializationMethods
from tests.test_EPIMRIDistortionCorrection import TestEPIMRIDistortionCorrection
from tests.test_Optimizers import TestOptimizers
from tests.test_Laplacians import TestLaplacians

if __name__ == '__main__':
	suite = unittest.TestSuite()

	# utils
	suite.addTest(TestUtils("test_load_data"))
	suite.addTest(TestUtils("test_normalize"))
	suite.addTest(TestUtils("test_m_plus_and_m_minus"))
	suite.addTest(TestUtils("test_interp_parallel_1D"))
	suite.addTest(TestUtils("test_interp_parallel_2D"))
	suite.addTest(TestUtils("test_grid_2D"))
	suite.addTest(TestUtils("test_grid_3D"))
	suite.addTest(TestUtils("test_grid_4D"))

	# Image Models
	suite.addTest(TestImageModels("test_2D_interp"))
	suite.addTest(TestImageModels("test_3D_interp"))
	suite.addTest(TestImageModels("test_4D_interp"))
	suite.addTest(TestImageModels("test_interp_derivative_check_2D"))
	suite.addTest(TestImageModels("test_interp_derivative_check_3D"))
	suite.addTest(TestImageModels("test_interp_derivative_check_4D"))

	# Regularizers
	suite.addTest(TestRegularizers("test_tik_reg"))
	suite.addTest(TestRegularizers("test_quad_reg"))
	suite.addTest(TestRegularizers("test_quad_reg_prox_solve"))

	# Linear Operators
	suite.addTest(TestLinearOperators("test_avg1dconv"))
	suite.addTest(TestLinearOperators("test_diff1dconv"))
	suite.addTest(TestLinearOperators("test_lap3D"))
	suite.addTest(TestLinearOperators("test_lap2D"))
	suite.addTest(TestLinearOperators("test_lap1D"))
	suite.addTest(TestLinearOperators("test_identity"))
	suite.addTest(TestLinearOperators("test_convfft"))

	# Preconditioners
	suite.addTest(TestPreconditioners("test_PC_3D"))
	suite.addTest(TestPreconditioners("test_PC_Jacobi_diag"))

	# Initialization Methods
	suite.addTest(TestInitializationMethods("test_init_OT"))
	suite.addTest(TestInitializationMethods("test_init_rand"))
	suite.addTest(TestInitializationMethods("test_init_zeros"))

	# EPIMRIDistortionCorrection
	suite.addTest(TestEPIMRIDistortionCorrection("test_obj_func_3D"))
	suite.addTest(TestEPIMRIDistortionCorrection("test_obj_func_4D"))

	# Optimizers
	suite.addTest(TestOptimizers("test_LBFGS"))
	suite.addTest(TestOptimizers("test_GNPCG"))
	suite.addTest(TestOptimizers("test_ADMM"))
	suite.addTest(TestOptimizers("test_linear_solvers"))

	# Laplacians
	suite.addTest(TestLaplacians("test_laplacians_3d"))
	suite.addTest(TestLaplacians("test_inverse_laplacians_3d"))

	# RUN TESTS
	unittest.TextTestRunner().run(suite)