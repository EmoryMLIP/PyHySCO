# Usage Instructions

## Setup
1. Load Data
```python
# load the image and domain information
# requires input image file paths and phase encoding direction
data = DataObject('im1.nii.gz','im2.nii.gz',1, device=device)
```
2. Correction Object
```python
# set-up the objective function
# requires data object & regularization parameters
# optionally set linear operators, regularizer, preconditioning
loss_func = EPIMRIDistortionCorrection(data, alpha=300.0, beta=1e-4)
```
3. Initialization
```python
# initialize the field map
# see EPI_MRI.InitializationMethods for details
B0 = loss_func.initialize()
```

## Optimization
1. Choose one of the following optimization methods:
   1. LBFGS
    ```python
    # set-up the optimizer
    # requires correction object
    # optionally maximum # of iterations, verbose flag, log file path
    opt = LBFGS(loss_func, max_iter=200, verbose=True)
    ```
   2. Gauss Newton
    ```python
    # set-up the optimizer
    # requires correction object
    # optionally maximum # of GN iterations, linear solver,
    # verbose flag, log file path
    opt = GaussNewton(loss_func, max_iter=20, verbose=True)
    ```
   3. ADMM
    ```python
    # set-up the optimizer
    # requires correction object
    # optionally maximum # of iterations, verbose flag, log file path
   loss_func_ADMM = EPIMRIDistortionCorrection(data, alpha=300.0, beta=1e-4, regularizer=myLaplacian1D, rho=1e3) 
   opt = ADMM(loss_func_ADMM, max_iter=20,  verbose=True)
    ```
2. Run Correction
```python
# optimize!
opt.run_correction(B0)
```

## Correction
1. Apply Correction and Save Images
   1. Jacobian
    ```python
    # apply optimal field map to get corrected images
    # field map and corrected image(s) will be saved
    opt.apply_correction(method='jac')
    ```
   2. Least Squares
    ```python
    # apply optimal field map to get corrected images
    # field map and corrected image(s) will be saved
    opt.apply_correction(method='lstsq')
    ```

## Evaluation
1. Metrics
```python
# calculate distance improvement and smoothness improvement
opt.evaluate_correction()
```
2. Visualization
```python
# show and save image of optimal field and corrected images
# can specify slice number and image intensity
opt.visualize()
```
3. Log Files
    1. Optimization history automatically saved in {path}+log_file.txt
    2. Print full optimization history
        ```python
         opt.log.print_history()
         ```