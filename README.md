## PyHySCO
This is a package for Echo-Planar MRI susceptibility artifact correction implemented in PyTorch.

## Installation
From PyPI:
```bash
pip install PyHySCO
```

Python package dependencies (automatically installed by pip) are listed in `requirements.txt`.
It is suggested to run the python file `tests/test_all.py` to ensure all tests are passing and the code is setup properly.

## Usage

### Command Line Correction
The program can be run directly from a terminal or command line by using the ```python``` command to run the file ```pyhysco.py``` or if installed using pip the command ```pyhysco```.
Supplying the following required parameters:
* file_1: file path of first image (stored as nii.gz) with phase encoding direction opposite of file_2
* file_2: file path of second image (stored as nii.gz) with phase encoding direction opposite of file_1
* ped: phase-encoding dimension (1, 2, or 3)

Use the help flag (--help) to see optional parameters available.

Minimal Usage:
```bash
pyhysco --file_1 <image1> --file_2 <image2> --ped <phase encoding direction>
```
Example:
```bash
pyhysco --file_1 image1.nii.gz --file_2 image2.nii.gz --ped 1 --output_dir results/ --max_iter 25
```

### Write a Correction Script
A user-written script can be used to call the methods of the program. 

Example:
```python
from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the image and domain information
# change this function call to be the filepath for your data
data = DataObject('../data/156334_v.nii.gz', '../data/156334_-v.nii.gz', 1, device=device,dtype=torch.float32)

loss_func = EPIMRIDistortionCorrection(data, 300, 1e-4, regularizer=myLaplacian3D, PC = JacobiCG)
# initialize the field map
B0 = loss_func.initialize(blur_result=True)
# set-up the optimizer
# change path to be where you want logfile and corrected images to be stored
opt = GaussNewton(loss_func, max_iter=500, verbose=True, path='results/gnpcg-Jac/')
# optimize!
opt.run_correction(B0)
# save field map and corrected images
opt.apply_correction()
# see plot of corrected images
opt.visualize()
```

### Examples and Further Documentation
There are a set of examples in the `examples` directory. Full API documentation is in the `docs` directory. See also `Instructions.md` for an overview of the correction process.