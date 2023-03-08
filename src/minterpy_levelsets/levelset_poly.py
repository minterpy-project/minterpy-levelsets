"""
Base class for levelset polynomials
"""

import numpy as np
from scipy.linalg import lu, lstsq
from numpy.random import Generator, PCG64
from minterpy import MultiIndexSet, Grid, LagrangePolynomial, NewtonPolynomial
from minterpy import get_transformation
from minterpy.extras.regression.ordinary_regression import compute_regression_matrix, OrdinaryRegression
from .pointcloud_utils import output_VTR, sample_points, closest_points

__all__ = ['LevelsetPoly', 'NBLevelsetPoly']

class LevelsetPoly:
    """Constructs a polynomial representation for a levelset function of a surface that passes through a point cloud."""

    def __init__(self, pointcloud: np.ndarray, poly_degree: int = None, lp_degree: float = 2.0,
                 method: str = 'BK', tol: float = 1e-4, verbose = False):
        """ Constructor

        Attributes
        ----------
        pointcloud : A numpy array of shape n x m
        poly_degree : degree of polynomial for fit
        lp_degree : lp-degree of polynomial
        method : 'BK', 'LB'
        tol : tolerance of fit (Note: If specified, poly_degree will be treated as starting degree)
        """

        self._pointcloud = pointcloud
        self._lagrange_poly = None
        self._newton_poly = None
        self._gradient_poly = None

        if method == 'BK':
            if poly_degree is None:
                poly_degree = 1
                while True:
                    poly_degree += 1
                    lag_poly, newt_poly = interpolate_bk(pointcloud, poly_degree, lp_degree)
                    eval_points = newt_poly(pointcloud)

                    dx_poly = newt_poly.diff([1, 0, 0])
                    dy_poly = newt_poly.diff([0, 1, 0])
                    dz_poly = newt_poly.diff([0, 0, 1])

                    gradient_poly = NewtonPolynomial.from_poly(newt_poly,
                                                                     new_coeffs=np.c_[dx_poly.coeffs,
                                                                     dy_poly.coeffs,
                                                                     dz_poly.coeffs])

                    val_grads = gradient_poly(pointcloud)
                    norm_val_grads = np.linalg.norm(val_grads, axis=1)
                    error_at_points = eval_points  / norm_val_grads
                    max_error = np.max(np.abs(error_at_points))
                    if verbose:
                        print(f"Levelset error (method = 'BK'), n = {poly_degree}, lp = {lp_degree} : {max_error}")

                    if np.abs(max_error) < tol:
                        self._lagrange_poly = lag_poly
                        self._newton_poly = newt_poly
                        self._gradient_poly = gradient_poly
                        break
            else:
                lag_poly, newt_poly = interpolate_bk(pointcloud, poly_degree, lp_degree)
                self._lagrange_poly = lag_poly
                self._newton_poly = newt_poly

        elif method == 'LB':
            #LB_sum method
            if poly_degree is None:
                poly_degree = 1
                while True:
                    poly_degree += 1
                    lag_poly, newt_poly = interpolate_lb(pointcloud, poly_degree, lp_degree)
                    eval_points = newt_poly(pointcloud)

                    dx_poly = newt_poly.diff([1, 0, 0])
                    dy_poly = newt_poly.diff([0, 1, 0])
                    dz_poly = newt_poly.diff([0, 0, 1])

                    gradient_poly = NewtonPolynomial.from_poly(newt_poly,
                                                                     new_coeffs=np.c_[dx_poly.coeffs,
                                                                     dy_poly.coeffs,
                                                                     dz_poly.coeffs])

                    val_grads = gradient_poly(pointcloud)
                    norm_val_grads = np.linalg.norm(val_grads, axis=1)
                    error_at_points = eval_points / norm_val_grads
                    max_error = np.max(np.abs(error_at_points))

                    if verbose:
                        print(f"Levelset error (method = 'LB'), n = {poly_degree}, lp = {lp_degree} : {max_error}")

                    if np.abs(max_error) < tol:
                        self._lagrange_poly = lag_poly
                        self._newton_poly = newt_poly
                        self._gradient_poly = gradient_poly
                        break
            else:
                lag_poly, newt_poly = interpolate_lb(pointcloud, poly_degree, lp_degree)
                self._lagrange_poly = lag_poly
                self._newton_poly = newt_poly

        else:
            raise ValueError(f"Unrecognized levelset interpolation method {method}.")


    def __call__(self, xx):
        #Evaluation
        return self._newton_poly(xx)

    @property
    def lagrange_coeffs(self):
        if self._lagrange_poly is not None:
            return self._lagrange_poly.coeffs
        else:
            return None

    @property
    def newton_coeffs(self):
        if self._newton_poly is not None:
            return self._newton_poly.coeffs
        else:
            return None

    def compute_gradients_at(self, xx):
        """Compute the gradients at each point on the given points

        """
        if self._gradient_poly is None:
            dx_poly = self._newton_poly.diff([1, 0, 0])
            dy_poly = self._newton_poly.diff([0, 1, 0])
            dz_poly = self._newton_poly.diff([0, 0, 1])

            self._gradient_poly = NewtonPolynomial.from_poly(self._newton_poly,
                                                             new_coeffs=np.c_[dx_poly.coeffs,
                                                                              dy_poly.coeffs,
                                                                              dz_poly.coeffs])

        val_grads = self._gradient_poly(xx)

        return val_grads

    def compute_curvatures_at(self, xx):
        #Evaluate curvatures
        """Compute mean and gauss curvatures at the pointcloud
            """
        val_grads = self.compute_gradients_at(xx)
        grad_poly_newt = self._gradient_poly

        num_points = xx.shape[0]
        dx_newt_poly = NewtonPolynomial.from_poly(polynomial=self._newton_poly, new_coeffs=grad_poly_newt.coeffs[:, 0])
        dy_newt_poly = NewtonPolynomial.from_poly(polynomial=self._newton_poly, new_coeffs=grad_poly_newt.coeffs[:, 1])
        dz_newt_poly = NewtonPolynomial.from_poly(polynomial=self._newton_poly, new_coeffs=grad_poly_newt.coeffs[:, 2])

        dxx_newton = dx_newt_poly.diff([1, 0, 0])
        dxy_newton = dx_newt_poly.diff([0, 1, 0])
        dxz_newton = dx_newt_poly.diff([0, 0, 1])
        dyx_newton = dy_newt_poly.diff([1, 0, 0])
        dyy_newton = dy_newt_poly.diff([0, 1, 0])
        dyz_newton = dy_newt_poly.diff([0, 0, 1])
        dzx_newton = dz_newt_poly.diff([1, 0, 0])
        dzy_newton = dz_newt_poly.diff([0, 1, 0])
        dzz_newton = dz_newt_poly.diff([0, 0, 1])

        val_hessian = np.zeros((3, 3, num_points))

        val_hessian[0, 0, :] = dxx_newton(xx)
        val_hessian[0, 1, :] = dxy_newton(xx)
        val_hessian[0, 2, :] = dxz_newton(xx)
        val_hessian[1, 0, :] = dyx_newton(xx)
        val_hessian[1, 1, :] = dyy_newton(xx)
        val_hessian[1, 2, :] = dyz_newton(xx)
        val_hessian[2, 0, :] = dzx_newton(xx)
        val_hessian[2, 1, :] = dzy_newton(xx)
        val_hessian[2, 2, :] = dzz_newton(xx)

        gauss_curvature = np.zeros(num_points)
        mean_curvature = np.zeros(num_points)
        # Evaluate curvatures
        for i in range(num_points):
            denom1 = np.linalg.norm(val_grads[i, :]) ** 4
            adjoint_matrix = np.zeros((3, 3))
            adjoint_matrix[0, 0] = val_hessian[1, 1, i] * val_hessian[2, 2, i] - val_hessian[1, 2, i] * val_hessian[
                2, 1, i]
            adjoint_matrix[0, 1] = val_hessian[1, 2, i] * val_hessian[2, 0, i] - val_hessian[1, 0, i] * val_hessian[
                2, 2, i]
            adjoint_matrix[0, 2] = val_hessian[1, 0, i] * val_hessian[2, 1, i] - val_hessian[1, 1, i] * val_hessian[
                2, 0, i]
            adjoint_matrix[1, 0] = val_hessian[0, 2, i] * val_hessian[2, 1, i] - val_hessian[0, 1, i] * val_hessian[
                2, 2, i]
            adjoint_matrix[1, 1] = val_hessian[0, 0, i] * val_hessian[2, 2, i] - val_hessian[0, 2, i] * val_hessian[
                2, 0, i]
            adjoint_matrix[1, 2] = val_hessian[0, 1, i] * val_hessian[2, 0, i] - val_hessian[0, 0, i] * val_hessian[
                2, 1, i]
            adjoint_matrix[2, 0] = val_hessian[0, 1, i] * val_hessian[1, 2, i] - val_hessian[0, 2, i] * val_hessian[
                1, 1, i]
            adjoint_matrix[2, 1] = val_hessian[1, 0, i] * val_hessian[0, 2, i] - val_hessian[0, 0, i] * val_hessian[
                1, 2, i]
            adjoint_matrix[2, 2] = val_hessian[0, 0, i] * val_hessian[1, 1, i] - val_hessian[0, 1, i] * val_hessian[
                1, 0, i]

            numer1 = np.dot(val_grads[i, :], np.dot(adjoint_matrix, np.transpose(val_grads[i, :])))

            gauss_curvature[i] = numer1 / denom1

            denom2 = 2.0 * np.linalg.norm(val_grads[i, :]) ** 3

            numer2 = np.dot(val_grads[i, :], np.dot(val_hessian[:, :, i], val_grads[i, :])) - (
                    np.linalg.norm(val_grads[i, :]) ** 2) * np.trace(val_hessian[:, :, i])

            mean_curvature[i] = numer2 / denom2

        return mean_curvature, gauss_curvature

    def sample_points(self, max_points, bounds=1.0, tol=1e-6, max_iters=10,
                              random_seed=42):
        """ Randomly sample points on the zero isosurface of a polynomial.
        """
        return sample_points(self._newton_poly, max_points=max_points, bounds=bounds,
                             tol=tol, max_iters=max_iters, random_seed=random_seed,
                             grad_newt_poly=self._gradient_poly)

    def output_VTR(self, frame=0, prefix='surf_', mesh_size=50, bounds=1.00):
        # Generate VTR output
        output_VTR(self._newton_poly, frame=frame, prefix=prefix, mesh_size=mesh_size, bounds=bounds)

    def closest_point_projection(self, x0, tol=1e-6, max_iters=10):
        return closest_points(self._newton_poly, self._gradient_poly, x0, tol, max_iters)

class NBLevelsetPoly(LevelsetPoly):
    """Constructs a polynomial representation for a level set function on a narrowband."""

    def __init__(self, nb_points: np.ndarray, ls_vals: np.ndarray, poly_degree: int = None, lp_degree: float = 2.0,
                tol: float = 1e-4, verbose = False):
        """ Constructor

        Attributes
        ----------
        nb_points : A numpy array of shape n x m
        ls_vals : A numpy array of shape n
        poly_degree : degree of polynomial for fit
        lp_degree : lp-degree of polynomial
        tol : tolerance of fit (Note: If specified, poly_degree will be treated as starting degree)
        """

        self._pointcloud = nb_points
        self._ls_vals = ls_vals
        self._lagrange_poly = None
        self._newton_poly = None
        self._gradient_poly = None

        if poly_degree is None:
            poly_degree = 1
            while True:
                poly_degree += 1
                ## Narrowband levelset fit
                mi = MultiIndexSet.from_degree(3, poly_degree, lp_degree)
                grid = Grid(mi)
                regressor = OrdinaryRegression(mi, grid)
                regressor.fit(nb_points, ls_vals)
                max_error = regressor.regfit_l2_error
                if verbose:
                    print(f"NBLevelset error, n = {poly_degree}, lp = {lp_degree} : {max_error}")

                if np.abs(max_error) < tol:
                    self._lagrange_poly = regressor.origin_poly
                    self._newton_poly = regressor.eval_poly
                    self._gradient_poly = None
                    break
        else:
            mi = MultiIndexSet.from_degree(3, poly_degree, lp_degree)
            grid = Grid(mi)
            regressor = OrdinaryRegression(mi, grid)
            regressor.fit(nb_points, ls_vals)
            self._lagrange_poly = regressor.origin_poly
            self._newton_poly = regressor.eval_poly


def interpolate_bk(pointcloud, n, lp_degree):
    """Implements the Basis of Kernel (BK) method for finding the polynomial representation."""

    num_points, m = pointcloud.shape

    # Set up the regressor
    mi = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    lag_poly = LagrangePolynomial(mi, None)

    num_coeffs = len(mi)
    if num_points < num_coeffs:
        raise RuntimeError(
            f"Not enough points {num_points} to fit with degree {n}. Atleast {num_coeffs} needed.")

    # Construct the R matrix
    regression_matrix = compute_regression_matrix(lag_poly, pointcloud)

    # Normalizing the R matrix
    regression_matrix /= np.linalg.norm(regression_matrix, np.inf, axis=(0, 1))

    # LU Decompose
    P, L, U = lu(regression_matrix)

    # print(f"Tolerance for rank computation is {np.max(U.shape)*np.spacing(np.linalg.norm(U,2))}")
    # The tolerance for SVD in rank estimation is different in MATLAB and numpy. The following line makes them same.
    # np.spacing is equivalent to 'eps' in MATLAB
    # K = np.linalg.matrix_rank(U, np.max(U.shape) * np.spacing(np.linalg.norm(U, 2)))
    u = np.diag(U)
    # v = np.sort(np.abs(u))
    J = np.argsort(np.abs(u))

    U[J[0], J[0]] = 0.0
    reg_mat_new = P @ L @ U
    P, L, U = lu(reg_mat_new)

    # K = np.linalg.matrix_rank(U, np.max(U.shape) * np.spacing(np.linalg.norm(U, 2)))
    u = np.diag(U)
    # v = np.sort(np.abs(u))
    J = np.argsort(np.abs(u))

    # Currently, we hardcode the level_dim to 1. In principle, this can also be > 1.
    level_dim = 1

    BK = np.zeros((num_coeffs, level_dim))
    UK = np.identity(num_coeffs)
    UK[:, J[1:num_coeffs]] = U[:, J[1:num_coeffs]]

    b = -U[:, J[0]]
    BK[:, 0] = np.linalg.solve(UK, b)
    BK[J[0], 0] = 1

    bk_lag_poly = LagrangePolynomial.from_poly(lag_poly, new_coeffs=BK[:, 0])
    transformer_l2n = get_transformation(bk_lag_poly, NewtonPolynomial)

    newt_poly = transformer_l2n()

    return bk_lag_poly, newt_poly



# Attempts to interpolate a pointcloud with a polynomial of degree n in dimension m
def interpolate_lb(pointcloud, n, lp_degree: float = 2.0, tol=1e-4):
    """Implementation of the sum of Lagrange Basis method for interpolation. Constructs a Lagrange basis on
    a suitable set of sub-sampled points."""

    num_points, m = pointcloud.shape

    # Set up the regressor
    mi = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    lag_poly = LagrangePolynomial(mi, None)

    num_coeffs = len(mi)
    if num_points < num_coeffs:
        raise RuntimeError(
            f"Not enough points {num_points} to fit with degree {n}. At least {num_coeffs} needed.")

    # Construct the R matrix
    regression_matrix = compute_regression_matrix(lag_poly, pointcloud)

    P, L, U = lu(regression_matrix)

    # print(f"Tolerance for rank computation is {np.max(U.shape)*np.spacing(np.linalg.norm(U,2))}")
    # The tolerance for SVD in rank estimation is different in MATLAB and numpy. The following line makes them same.
    # np.spacing is equivalent to 'eps' in MATLAB
    #K = np.linalg.matrix_rank(U, np.max(U.shape) * np.spacing(np.linalg.norm(U, 2)))
    u = np.diag(U)
    #v = np.sort(np.abs(u))
    J = np.argsort(np.abs(u))

    index_set = [*range(0, num_points)]
    ordered_points = P @ index_set
    ordered_points = np.delete(ordered_points, np.where(ordered_points == J[0])).astype(int)

    mi_new = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    lag_poly_new = LagrangePolynomial(mi_new, None)

    reg_mat_new = compute_regression_matrix(lag_poly_new, pointcloud[ordered_points[:num_coeffs - 1], :])

    soln, _, _, _ = lstsq(reg_mat_new, np.eye(num_coeffs - 1))

    LB_sum = soln[:, 0]
    for i in range(1, num_coeffs - 1):
        LB_sum += soln[:, i]

    # The isocontour with 1.0 should be the zero levelset
    LB_sum -= 1.0

    LB_sum_lag = LagrangePolynomial.from_poly(polynomial=lag_poly_new, new_coeffs=LB_sum)
    l2n_transformer = get_transformation(LB_sum_lag, NewtonPolynomial)
    LB_sum_newton = l2n_transformer(LB_sum_lag)


    return LB_sum_lag, LB_sum_newton