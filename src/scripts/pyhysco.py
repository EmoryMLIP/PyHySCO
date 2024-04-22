from optimization.ADMM import *
from EPI_MRI.EPIMRIDistortionCorrection import *
import argparse
import warnings

def main():
    parser = argparse.ArgumentParser(description="PyHySCO: EPI-MRI Distortion Correction.")
    parser.add_argument("file_1", type=str, help="Path to the input 1 data file (NIfTI format .nii.gz)")
    parser.add_argument("file_2", type=str, help="Path to the input 2 data file (NIfTI format .nii.gz)")
    parser.add_argument("ped", action='store', type=int, choices=range(1, 4), help="Dimension of phase encoding direction")
    parser.add_argument("--output_dir", type=str, default=" ", help="Directory to save the corrected images and reports (default=cwd)")
    parser.add_argument("--alpha", type=float, default=300, help="Smoothness regularization parameter (default=300)")
    parser.add_argument("--beta", type=float, default=1e-4, help="Intensity modulation constraint parameter (default=1e-4)")
    parser.add_argument("--rho", type=float, default=1e3, help="Initial Lagrangian parameter (ADMM only) (default=1e3)")
    parser.add_argument("--optimizer", default=GaussNewton, help="Optimizer to use (default=GaussNewton)")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum number of iterations (default=50)")
    parser.add_argument("--verbose", action="store_true", help="Print details of optimization (default=True)")
    parser.add_argument("--precision", choices=['single', 'double'], default='single', help="Use (single/double) precision (default=single)")
    parser.add_argument("--correction", choices=['jac', 'lstsq'], default='jac', help="Use (Jacobian ['jac']/ Least Squares ['lstsq']) correction (default=lstsq)")
    parser.add_argument("--averaging", default=myAvg1D, help="LinearOperator to use as averaging operator (default=myAvg1D)")
    parser.add_argument("--derivative", default=myDiff1D, help="LinearOperator to use as derivative operator (default=myDiff1D)")
    parser.add_argument("--initialization", default=InitializeCF, help="Initialization method to use (default=InitializeCF)")
    parser.add_argument("--regularizer", default=myLaplacian3D, help="LinearOperator to use for smoothness regularization term (default=myLaplacian3D)")
    parser.add_argument("--PC", default=JacobiCG, help="Preconditioner to use (default=JacobiCG)")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # print(device)

    args = parser.parse_args()
    # defaults
    if args.precision == 'single':
        dtype = torch.float32
    else:
        dtype = torch.float64
    if args.optimizer is ADMM and args.regularizer is not myLaplacian1D:
        warnings.warn("Recommended use is a one-dimensional (in the distortion dimension) regularizer for ADMM.")
    else:
        args.rho = 0.0
    data = DataObject(args.file_1, args.file_2, args.ped, device=device, dtype=dtype)
    loss_func = EPIMRIDistortionCorrection(data, alpha=args.alpha, beta=args.beta, averaging_operator=args.averaging, derivative_operator=args.derivative, regularizer=args.regularizer, rho=args.rho, PC=args.PC)
    B0 = loss_func.initialize(blur_result=True)
    # set-up the optimizer
    # change path to be where you want logfile and corrected images to be stored
    opt = args.optimizer(loss_func, max_iter=args.max_iter, verbose=True, path=args.output_dir)
    opt.run_correction(B0)
    opt.apply_correction(method=args.correction)
    if args.verbose:
        opt.visualize()


if __name__ == "__main__":
    main()
