"""
Perform optimization to find optimal field map for distortion correction.

Files:
- `ADMM`: A class implementing the Alternating Direction Method of Multipliers optimization algorithm.
- `EPIOptimize`: A base class for different optimization algorithms in EPI-MRI distortion correction.
- `GaussNewton`: A class implementing the Gauss-Newton optimization algorithm.
- `LBFGS`: A class implementing the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm.
- `LCurve`: A class for EPI-MRI distortion correction choosing alpha parameter with LCurve.
- `LinearSolvers` : Classes for linear solvers used in Gauss Newton and ADMM (e.g. conjugate gradient).
- `OptimizationLogger`: A class for logging optimization iteration information and generating reports.
"""