from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.ADMM import *
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the image and domain information
# change this function call to be the filepath for your data
data = DataObject('../data/156334_v.nii.gz', '../data/156334_-v.nii.gz', 1, device=device)
# set-up the objective function
loss_func = EPIMRIDistortionCorrection(data, 300, 1e-4,averaging_operator=myAvg1D, derivative_operator=myDiff1D, regularizer=myLaplacian1D, rho=1e3, PC=JacobiCG)
# initialize the field map
B0 = loss_func.initialize(blur_result=True)
# set-up the optimizer
# change path to be where you want logfile and corrected images to be stored
opt = ADMM(loss_func, max_iter=500, rho_max=1e6, rho_min=1e1, max_iter_gn=1, max_iter_pcg=20, verbose=True, path='results/admm/')
# optimize!
opt.run_correction(B0)
# save field map and corrected images
opt.apply_correction()
# see plot of corrected images
opt.visualize()
