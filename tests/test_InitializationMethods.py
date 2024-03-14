import unittest
import os
from EPI_MRI.EPIMRIDistortionCorrection import DataObject
from EPI_MRI.InitializationMethods import *


class TestInitializationMethods(unittest.TestCase):
    def setUp(self):
        """ Initialize any common data or setup needed for the tests. """
        torch.manual_seed(81)  # reproducibility with randomness
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

    def test_init_OT(self):
        """ Test function load_data. """
        # TOPUP style input 2D
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeOT()
        B0 = init.eval(data, blur_result=False)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1] + 1)

        # TOPUP style input 3D
        # Create a synthetic NIfTI image in memory
        shape = (50, 62, 73, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeOT()
        B0 = init.eval(data, blur_result=False)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2] + 1)
        B0 = init.eval(data, blur_result=True)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2] + 1)

        # TOPUP style input 4D
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 9, 4, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeOT()
        B0 = init.eval(data, blur_result=False)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2])
        self.assertEqual(B0.shape[3], data.m[3] + 1)
        # Delete the temporary file
        os.remove(img_file)

    def test_init_rand(self):
        """ Test function load_data. """
        # TOPUP style input 2D
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeRandom()
        B0 = init.eval(data)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1] + 1)

        # TOPUP style input 3D
        # Create a synthetic NIfTI image in memory
        shape = (50, 62, 73, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeRandom()
        B0 = init.eval(data)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2] + 1)

        # TOPUP style input 4D
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 9, 4, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeRandom()
        B0 = init.eval(data)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2])
        self.assertEqual(B0.shape[3], data.m[3] + 1)
        # Delete the temporary file
        os.remove(img_file)

    def test_init_zeros(self):
        """ Test function load_data. """
        # TOPUP style input 2D
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeZeros()
        B0 = init.eval(data)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1] + 1)

        # TOPUP style input 3D
        # Create a synthetic NIfTI image in memory
        shape = (50, 62, 73, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeZeros()
        B0 = init.eval(data)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2] + 1)

        # TOPUP style input 4D
        # Create a synthetic NIfTI image in memory
        shape = (12, 10, 9, 4, 2)
        image_data = np.asarray(torch.randn(*shape, dtype=self.dtype))
        img_file = os.getcwd() + 'test.nii.gz'
        img = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(img, img_file)
        # Make DataObject with the dummy image file (PED = 1)
        data = DataObject(img_file, phase_encoding_direction=1, dtype=self.dtype, device=self.device)
        init = InitializeZeros()
        B0 = init.eval(data)
        self.assertEqual(B0.shape[0], data.m[0])
        self.assertEqual(B0.shape[1], data.m[1])
        self.assertEqual(B0.shape[2], data.m[2])
        self.assertEqual(B0.shape[3], data.m[3] + 1)
        # Delete the temporary file
        os.remove(img_file)


# run tests when this file is called
if __name__ == '__main__':
    unittest.main()
