from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.LBFGS import *
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the image and domain information
# change this function call to be the filepath for your data
data = DataObject('../data/156334_v.nii.gz', '../data/156334_-v.nii.gz', 1, device=device, dtype=torch.float32)
# set-up the objective function
loss_func = EPIMRIDistortionCorrection(data, alpha=300.0, beta=1e-4)
# initialize the field map
B0 = loss_func.initialize(blur_result=True)
# set-up the optimizer
# change path to be where you want logfile and corrected images to be stored
opt = LBFGS(loss_func, max_iter=200, verbose=True, path='results/lbfgs/')
# optimize!
opt.run_correction(B0)
# save field map and corrected images
opt.apply_correction()
# see plot of corrected images
opt.visualize()
