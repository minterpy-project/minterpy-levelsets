# -*- coding:utf-8 -*-
# pointcloud_utils.py

import numpy as np
from scipy.linalg import lu, lstsq
from numpy.random import Generator, PCG64
from minterpy.core import MultiIndexSet, Grid
from minterpy import LagrangePolynomial, CanonicalPolynomial, NewtonPolynomial
from minterpy import Regression
from minterpy import NewtonPolyDerivator
from minterpy import get_transformation

__all__ = ['points_on_ellipsoid', 'points_on_biconcave_disc', 'points_on_torus', 'interpolate_pointcloud',
           'find_polynomial', 'get_gradients', 'get_curvatures', 'output_VTK', 'output_VTR',
           'interpolate_pointcloud_CK_sum', 'find_polynomial_CK_sum', 'sample_points_on_poly',
           'interpolate_pointcloud_arb', 'find_polynomial_arb']


def points_on_ellipsoid(num_points, radius_x=1.0, radius_y=1.0, radius_z=1.0, random_seed=42, verbose=False):
    """Generates a pointcloud on an ellipsoid surface

    Parameters:
    num_points (int): Number of points to be generated
    radius_x (float): The radius along the X-axis
    radius_y (float): The radius along the Y-axis
    radius_z (float): The radius along the Z-axis
    random_seed (int): Seed for the random number generator
    verbose (bool): Print information on screen

    Returns:
    np.array : The pointcloud generated with shape (3,num_points)

    """

    if verbose:
        print(f"No. of points sampled = {num_points}")

    rg = Generator(PCG64(random_seed))
    thetas = rg.random(num_points) * np.pi
    phis = rg.random(num_points) * np.pi * 2

    pointcloud = np.zeros((num_points, 3))
    pointcloud[:, 0] = radius_x*np.sin(thetas)*np.cos(phis)
    pointcloud[:, 1] = radius_y*np.sin(thetas)*np.sin(phis)
    pointcloud[:, 2] = radius_z*np.cos(thetas)

    return pointcloud


def points_on_biconcave_disc(num_points, param_c=0.5, param_d=0.375, random_seed=42, verbose=False):
    """Generates a pointcloud on a biconcave disc

    Parameters:
    num_points (int): Number of points to be generated
    param_c (float): Value of parameter 'c'
    param_d (float): Value of parameter 'd'
    random_seed (int): Seed for the random number generator
    verbose (bool): Print information on screen

    Returns:
    np.array : The pointcloud generated with shape (3,num_points)

    """
    if verbose:
        print(f"No. of points sampled = {num_points}")

    pointcloud = np.zeros((num_points, 3))
    rg = Generator(PCG64(random_seed))
    count = 0
    while count < num_points:
        y = 2.0*rg.random()-1.0
        z = 2.0*rg.random()-1.0

        t1 = (8*param_d*param_d*(y*y + z*z) + param_c*param_c*param_c*param_c)**(1.0/3.0)
        t2 = param_d*param_d + y*y + z*z
        if t1 >= t2:
            if rg.random() < 0.5:
                pointcloud[count, 0] = np.sqrt(t1 - t2)
            else:
                pointcloud[count, 0] = -np.sqrt(t1 - t2)

            pointcloud[count, 1] = y
            pointcloud[count, 2] = z
            count += 1

    return pointcloud


def points_on_torus(num_points, param_c=0.5, param_a=0.375, random_seed=42, verbose=False):
    """Generates a pointcloud on a torus

    Parameters:
    num_points (int): Number of points to be generated
    param_c (float): Distance from center of torus to center of tube
    param_a (float): Radius of torus tube
    random_seed (int): Seed for the random number generator
    verbose (bool): Print information on screen

    Returns:
    np.array : The pointcloud generated with shape (3,num_points)

    """
    if verbose:
        print(f"No. of points sampled = {num_points}")

    rg = Generator(PCG64(random_seed))
    us = rg.random(num_points) * np.pi * 2
    vs = rg.random(num_points) * np.pi * 2

    pointcloud = np.zeros((num_points, 3))
    pointcloud[:, 0] = (param_c + param_a * np.cos(vs)) * np.cos(us)
    pointcloud[:, 1] = (param_c + param_a * np.cos(vs)) * np.sin(us)
    pointcloud[:, 2] = param_a * np.sin(vs)

    return pointcloud


def interpolate_pointcloud(pointcloud, m, n, lp_degree: float = 2.0, tol=1e-4):
    """Attempts to interpolate a pointcloud with a polynomial of degree n in dimension m

    """

    N_points = pointcloud.shape[0]
    # Set up the regressor
    mi = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    grid = Grid(mi)
    regressor = Regression(grid, verbose=False)

    N = len(mi)

    if N_points < N:
        return [0, 0, 0, 0, 0]

    # Construct the R matrix
    regressor.cache_transform(pointcloud)
    R_matrix = regressor.regression_matrix
    # Normalizing the R matrix
    R_matrix /= np.linalg.norm(R_matrix, np.inf, axis=(0,1))

    P, L, U = lu(R_matrix)

    #print(f"Tolerance for rank computation is {np.max(U.shape)*np.spacing(np.linalg.norm(U,2))}")
    # The tolerance for SVD in rank estimation is different in MATLAB and numpy. The following line makes them same.
    # np.spacing is equivalent to 'eps' in MATLAB
    K = np.linalg.matrix_rank(U, np.max(U.shape)*np.spacing(np.linalg.norm(U,2)))
    u = np.diag(U)
    v = np.sort(np.abs(u))
    J = np.argsort(np.abs(u))

    actual_level_dim = N - K
    # Try forced construction of hypersurface with N-K = 1
    if actual_level_dim == 0:
        U[J[0],J[0]] = 0.0
        R_new = P @ L @ U
        P, L, U = lu(R_new)

        K = np.linalg.matrix_rank(U, np.max(U.shape)*np.spacing(np.linalg.norm(U,2)))
        u = np.diag(U)
        v = np.sort(np.abs(u))
        J = np.argsort(np.abs(u))
        K = N - 1

    level_dim = N-K

    BK = np.zeros((N,level_dim))
    UK = np.identity(N)
    UK[:,J[N-K:N]] = U[:,J[N-K:N]]

    for i in range(level_dim):
        b = -U[:,J[i]]
        BK[:,i] = np.linalg.solve(UK,b)
        BK[J[i],i] = 1

    # If the surface was force constructed, check if accuracy of fit is good enough in the excluded points
    if actual_level_dim == 0:
        max_error = np.max(np.abs(R_matrix @ BK))
        if max_error > tol:
            # Insufficient accuracy of fitting, try with a higher degree polynomial
            return [-1,0,0,0,0]

    # Find maximum error of fitting among all BKs
    max_error = -1.0
    error_at_points = np.zeros(N_points)
    for k in range(level_dim):
        # Normalize BK vector. This is redundant if the R_matrix is normalized (?).
        #max_bk = np.max(np.abs(BK[:,k]))
        #BK[:,k] = BK[:,k] / max_bk
        bk_lag_poly = LagrangePolynomial.from_poly(regressor._lagrange_poly, new_coeffs=BK[:, k])
        transformer_l2n = get_transformation(bk_lag_poly, NewtonPolynomial)

        newt_poly = transformer_l2n()

        eval_points = newt_poly(pointcloud)
        _, val_grads = get_gradients(pointcloud, newt_poly)
        norm_val_grads = np.linalg.norm(val_grads, axis=1)
        for p in range(N_points):
            error_at_points[p] = eval_points[p] / norm_val_grads[p]

        # error_vals = np.abs(newt_poly(pointcloud))

        # num_error_level = np.max(error_vals)
        # max_error = np.max([max_error, num_error_level])
    max_error = np.max(error_at_points)

    return [1, regressor, BK, newt_poly, max_error]


def interpolate_pointcloud_arb(pointcloud, m, n, lp_degree: float = 2.0, tol=1e-4):
    """Attempts to interpolate a pointcloud with a polynomial of degree n in dimension m

    """

    N_points = pointcloud.shape[0]
    # Set up the regressor
    mi = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    grid = Grid.from_domain(mi, pointcloud)
    regressor = Regression(grid, verbose=False)

    N = len(mi)

    if N_points < N:
        return [0, 0, 0, 0, 0]

    # Construct the R matrix
    regressor.cache_transform(pointcloud)
    R_matrix = regressor.regression_matrix
    # Normalizing the R matrix
    R_matrix /= np.linalg.norm(R_matrix, np.inf, axis=(0,1))

    P, L, U = lu(R_matrix)

    #print(f"Tolerance for rank computation is {np.max(U.shape)*np.spacing(np.linalg.norm(U,2))}")
    # The tolerance for SVD in rank estimation is different in MATLAB and numpy. The following line makes them same.
    # np.spacing is equivalent to 'eps' in MATLAB
    K = np.linalg.matrix_rank(U, np.max(U.shape)*np.spacing(np.linalg.norm(U,2)))
    u = np.diag(U)
    v = np.sort(np.abs(u))
    J = np.argsort(np.abs(u))

    actual_level_dim = N - K
    # Try forced construction of hypersurface with N-K = 1
    if actual_level_dim == 0:
        U[J[0],J[0]] = 0.0
        R_new = P @ L @ U
        P, L, U = lu(R_new)

        K = np.linalg.matrix_rank(U, np.max(U.shape)*np.spacing(np.linalg.norm(U,2)))
        u = np.diag(U)
        v = np.sort(np.abs(u))
        J = np.argsort(np.abs(u))
        K = N - 1

    level_dim = N-K

    BK = np.zeros((N,level_dim))
    UK = np.identity(N)
    UK[:,J[N-K:N]] = U[:,J[N-K:N]]

    for i in range(level_dim):
        b = -U[:,J[i]]
        BK[:,i] = np.linalg.solve(UK,b)
        BK[J[i],i] = 1

    # If the surface was force constructed, check if accuracy of fit is good enough in the excluded points
    if actual_level_dim == 0:
        max_error = np.max(np.abs(R_matrix @ BK))
        if max_error > tol:
            # Insufficient accuracy of fitting, try with a higher degree polynomial
            return [-1,0,0,0,0]

    # Find maximum error of fitting among all BKs
    max_error = -1.0
    error_at_points = np.zeros(N_points)
    for k in range(level_dim):
        # Normalize BK vector. This is redundant if the R_matrix is normalized (?).
        #max_bk = np.max(np.abs(BK[:,k]))
        #BK[:,k] = BK[:,k] / max_bk
        bk_lag_poly = LagrangePolynomial.from_poly(regressor._lagrange_poly, new_coeffs=BK[:, k])
        transformer_l2n = get_transformation(bk_lag_poly, NewtonPolynomial)

        newt_poly = transformer_l2n()

        eval_points = newt_poly(pointcloud)
        _, val_grads = get_gradients(pointcloud, newt_poly)
        norm_val_grads = np.linalg.norm(val_grads, axis=1)
        for p in range(N_points):
            error_at_points[p] = eval_points[p] / norm_val_grads[p]

        # error_vals = np.abs(newt_poly(pointcloud))

        # num_error_level = np.max(error_vals)
        # max_error = np.max([max_error, num_error_level])
    max_error = np.max(error_at_points)

    return [1, regressor, BK, newt_poly, max_error]


# Attempts to interpolate a pointcloud with a polynomial of degree n in dimension m
def interpolate_pointcloud_CK_sum(pointcloud, m, n, lp_degree: float = 2.0, tol=1e-4):
    N_points = pointcloud.shape[0]
    # Set up the regressor
    mi = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    grid = Grid(mi)
    regressor = Regression(grid, verbose=False)
    #l2n_transformer = get_transformation(lag_poly, NewtonPolynomial)
    N = len(mi)
    # Insufficient number of points to do interpolation
    if N_points < N:
        return [0, 0, 0, 0, 0]

    # Construct the R matrix
    regressor.cache_transform(pointcloud)
    R_matrix = regressor.regression_matrix

    P, L, U = lu(R_matrix)
    # print(R_matrix.shape)
    # print(f"Tolerance for rank computation is {np.max(U.shape)*np.spacing(np.linalg.norm(U,2))}")
    # The tolerance for SVD in rank estimation is different in MATLAB and numpy. The following line makes them same.
    # np.spacing is equivalent to 'eps' in MATLAB
    K = np.linalg.matrix_rank(U, np.max(U.shape) * np.spacing(np.linalg.norm(U, 2)))
    u = np.diag(U)
    v = np.sort(np.abs(u))
    J = np.argsort(np.abs(u))

    index_set = [*range(0, N_points)]
    ordered_points = P @ index_set
    ordered_points = np.delete(ordered_points, np.where(ordered_points == J[0])).astype(int)

    mi_new = MultiIndexSet.from_degree(spatial_dimension=m, poly_degree=n, lp_degree=lp_degree)
    grid_new = Grid(mi_new)
    regressor_new = Regression(grid_new, verbose=False)
    regressor_new.cache_transform(pointcloud[ordered_points[:N - 1], :])

    R_matrix_new = regressor_new.regression_matrix

    soln, _, _, _ = lstsq(R_matrix_new, np.eye(N - 1))

    CK_sum = soln[:, 0]
    for i in range(1, N - 1):
        CK_sum += soln[:, i]

    new_lag_poly = LagrangePolynomial.from_poly(polynomial=regressor._lagrange_poly, new_coeffs=CK_sum)
    l2n_transformer = get_transformation(regressor._lagrange_poly, NewtonPolynomial)
    DK_sum_newt = l2n_transformer(new_lag_poly)
    eval_points = DK_sum_newt(pointcloud)

    _, val_grads = get_gradients(pointcloud, DK_sum_newt)
    norm_val_grads = np.linalg.norm(val_grads, axis=1)
    error_at_points = np.zeros(N_points)
    for p in range(N_points):
        error_at_points[p] = (eval_points[p] - 1.0) / norm_val_grads[p]

    max_error_dk = np.max(np.abs(error_at_points))

    if max_error_dk > tol:
        return [-1, 0, 0, 0, 0]

    return [1, regressor, CK_sum, DK_sum_newt, max_error_dk]


def find_polynomial(pc, tol=1e-4):
    """Given a pointcloud, iteratively try to find the lowest degree polynomial that fits it with a hypersurface

    """
    m = 3
    n = 2
    lp_deg = 2 #np.inf
    res, reg, coeffs, newt_poly, err = interpolate_pointcloud(pc, m, n, lp_deg, tol)
    while res != 1:
        if res == 0:
            print("Requires more sample points")
            return [res, reg, coeffs, newt_poly, err]
        elif res == -1:
            n += 1
            lp_deg = 2.0
            res, reg, coeffs, newt_poly, err = interpolate_pointcloud(pc, m, n, lp_deg, tol)

    return [res, reg, coeffs, newt_poly, err]


def find_polynomial_arb(pc, tol=1e-4):
    """Given a pointcloud, iteratively try to find the lowest degree polynomial that fits it with a hypersurface

    """
    m = 3
    n = 2
    lp_deg = 2  # np.inf
    res, reg, coeffs, newt_poly, err = interpolate_pointcloud_arb(pc, m, n, lp_deg, tol)
    while res != 1:
        if res == 0:
            print("Requires more sample points")
            return [res, reg, coeffs, newt_poly, err]
        elif res == -1:
            n += 1
            lp_deg = 2.0
            res, reg, coeffs, newt_poly, err = interpolate_pointcloud_arb(pc, m, n, lp_deg, tol)

    return [res, reg, coeffs, newt_poly, err]


def find_polynomial_CK_sum(pc, tol=1e-4):
    """Given a pointcloud, iteratively try to find the lowest degree polynomial that fits it with a hypersurface

    """
    m = 3
    n = 2
    lp_deg = 2  # np.inf
    res, reg, coeffs, newt_poly, err = interpolate_pointcloud_CK_sum(pc, m, n, lp_deg, tol)
    while res != 1:
        if res == 0:
            print("Requires more sample points")
            return [res, reg, coeffs, newt_poly, err]
        elif res == -1:
            n += 1
            lp_deg = 2.0
            res, reg, coeffs, newt_poly, err = interpolate_pointcloud_CK_sum(pc, m, n, lp_deg, tol)

    return [res, reg, coeffs, newt_poly, err]

# def sample_points_on_poly(max_points, regressor, coeffs_newton, grad_op):
# 	"""Randomly sample points on a polynomial surface
#
# 	"""
# 	max_iters = 10
# 	num_error_level = 1e-14
# 	sampled_points = np.zeros((3,max_points))
# 	spos = 0
# 	while spos < max_points:
# 		coord = 2.0*np.random.random_sample(3) - 1.0
# 		fval = regressor.transformer.tree.eval(coord, coeffs_newton)
# 		dim = np.random.randint(3)
# 		iters = 0
# 		while np.abs(fval) > num_error_level and iters <= max_iters:
# 			dfxval = regressor.transformer.tree.eval(coord, grad_op[dim])
# 			xnew = coord[dim] - fval / dfxval
# 			if np.abs(xnew) >= 1.0:
# 				break
# 			coord[dim] = xnew
# 			fval = regressor.transformer.tree.eval(np.array(coord), coeffs_newton)
# 			iters += 1
#
# 		if np.abs(fval) < num_error_level:
# 			sampled_points[:,spos] = coord
# 			spos += 1
# 	return sampled_points


def sample_points_on_poly(max_points, newt_poly, grad_poly_newt = None, bounds = 1.0, tol = 1e-6, max_iters = 10, random_seed=42):
    """ Randomly sample points on the zero isosurface of a polynomial.
    """
    sampled_points = np.zeros((max_points,3))
    if grad_poly_newt is None:
        deriv = NewtonPolyDerivator(newt_poly)
        grad_poly_newt = deriv.get_gradient_poly()

    rg = Generator(PCG64(random_seed))

    spos = 0
    while spos < max_points:
        coord = 2.0*bounds*rg.random(3) - bounds
        fval = newt_poly(coord)
        dim = rg.integers(3)
        iters = 0
        while np.abs(fval) > tol and iters <= max_iters:
            dfval = grad_poly_newt(coord)
            dfxval = dfval[dim]
            xnew = coord[dim] - fval / dfxval
            if np.abs(xnew) >= bounds:
                break
            coord[dim] = xnew
            fval = newt_poly(np.array(coord))
            iters += 1

        if np.abs(fval) < tol:
            sampled_points[spos,:] = coord
            spos += 1

    return sampled_points


def output_VTK(pointcloud, frame=0, prefix='pc_', scalar_field=None, vector_field=None):
    """Visualize the pointcloud, the normal vectors, and curvatures evaluted at all points

    Parameters:
    pointcloud (np.array): The pointcloud data
    frame (int): Frame number in a timeseries
    prefix (str): Custom prefix for output files
    scalar_field (np.array): A scalar field defined on all points
    vector_field (np.array): A vector field defined on all points

    Returns:
    nothing

    """
    N_points, dim = pointcloud.shape

    outf = open(f"{prefix}{frame}.vtk","w")
    outf.write("# vtk DataFile Version 2.0\n")
    outf.write("Pointcloud data\n")
    outf.write("ASCII\n")
    outf.write("DATASET POLYDATA\n")
    outf.write(f"POINTS {N_points} float\n")

    for i in range(N_points):
        for j in range(dim):
            outf.write(f"{pointcloud[i, j]} ")
        outf.write("\n")

    if scalar_field is not None or vector_field is not None:
        outf.write(f"POINT_DATA {N_points}\n")
    if scalar_field is not None:
        outf.write(f"SCALARS curvatures float 1\n")
        outf.write(f"LOOKUP_TABLE default\n")
        for i in range(N_points):
            outf.write(f"{scalar_field[i]}\n")
    if vector_field is not None:
        outf.write(f"VECTORS normals float\n")
        for i in range(N_points):
            for j in range(dim):
                outf.write(f"{vector_field[i,j]} ")
            outf.write("\n")

    outf.close()


def output_VTR(newt_poly, frame=0, prefix='surf_', scalar_field=None, mesh_size=50, bounds=1.00):
    """Visualize the surface as a rectilinear grid

    """
    xvals = np.linspace(-bounds,bounds,mesh_size)
    yvals = np.linspace(-bounds,bounds,mesh_size)
    zvals = np.linspace(-bounds,bounds,mesh_size)

    outf = open(f"{prefix}{frame}.vtr","w")
    outf.write("<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
    outf.write(f"<RectilinearGrid WholeExtent=\"0 {mesh_size-1} 0 {mesh_size-1} 0 {mesh_size-1}\">\n")
    outf.write("<PointData Scalars=\"ls_phi\">\n")
    outf.write("<DataArray type=\"Float32\" Name=\"ls_phi\" format=\"ascii\">\n")
    for z in xvals:
        for y in yvals:
            for x in zvals:
                val = newt_poly(np.array([x,y,z]))
                outf.write(f"{str(val[0])}\n")
    outf.write("</DataArray>\n")
    if scalar_field is not None:
        outf.write("<DataArray type=\"Float32\" Name=\"field\" format=\"ascii\">\n")
        for z in xvals:
            for y in yvals:
                for x in zvals:
                    val = scalar_field(np.array([x, y, z]))
                    outf.write(f"{str(val[0])}\n")
        outf.write("</DataArray>\n")
    outf.write("</PointData>\n")
    outf.write("<Coordinates>\n")
    outf.write("<DataArray type=\"Float32\" Name=\"x_coords\" format=\"ascii\" RangeMin=\"-1\" RangeMax=\"1\">\n")
    for x in xvals:
        outf.write(f"{x}\n")
    outf.write("</DataArray>\n")
    outf.write("<DataArray type=\"Float32\" Name=\"y_coords\" format=\"ascii\" RangeMin=\"-1\" RangeMax=\"1\">\n")
    for x in xvals:
        outf.write(f"{x}\n")
    outf.write("</DataArray>\n")
    outf.write("<DataArray type=\"Float32\" Name=\"z_coords\" format=\"ascii\" RangeMin=\"-1\" RangeMax=\"1\">\n")
    for x in xvals:
        outf.write(f"{x}\n")
    outf.write("</DataArray>\n")
    outf.write("</Coordinates>\n")
    outf.write("</RectilinearGrid>\n")
    outf.write("</VTKFile>\n")
    outf.close()


def get_gradients(pointcloud, newt_poly):
    """Compute the gradients at each point on the pointcloud

    """
    grad_poly_newt = NewtonPolyDerivator(newt_poly).get_gradient_poly()
    val_grads = grad_poly_newt(pointcloud)

    return [grad_poly_newt, val_grads]


def get_curvatures(pointcloud, newt_poly):
    """Compute mean and gauss curvatures at the pointcloud
    """
    grad_poly_newt, val_grads = get_gradients(pointcloud, newt_poly)

    N_testing = pointcloud.shape[0]
    dx_newt_poly = NewtonPolynomial.from_poly(polynomial=newt_poly, new_coeffs=grad_poly_newt.coeffs[:, 0])
    dy_newt_poly = NewtonPolynomial.from_poly(polynomial=newt_poly, new_coeffs=grad_poly_newt.coeffs[:, 1])
    dz_newt_poly = NewtonPolynomial.from_poly(polynomial=newt_poly, new_coeffs=grad_poly_newt.coeffs[:, 2])

    grad_dx_newton = NewtonPolyDerivator(dx_newt_poly).get_gradient_poly()
    grad_dy_newton = NewtonPolyDerivator(dy_newt_poly).get_gradient_poly()
    grad_dz_newton = NewtonPolyDerivator(dz_newt_poly).get_gradient_poly()

    val_hessian = np.zeros((3, 3, N_testing))

    eval_grad_dx_newton = grad_dx_newton(pointcloud)
    eval_grad_dy_newton = grad_dy_newton(pointcloud)
    eval_grad_dz_newton = grad_dz_newton(pointcloud)

    val_hessian[0, 0, :] = eval_grad_dx_newton[:, 0]
    val_hessian[0, 1, :] = eval_grad_dx_newton[:, 1]
    val_hessian[0, 2, :] = eval_grad_dx_newton[:, 2]
    val_hessian[1, 0, :] = eval_grad_dy_newton[:, 0]
    val_hessian[1, 1, :] = eval_grad_dy_newton[:, 1]
    val_hessian[1, 2, :] = eval_grad_dy_newton[:, 2]
    val_hessian[2, 0, :] = eval_grad_dz_newton[:, 0]
    val_hessian[2, 1, :] = eval_grad_dz_newton[:, 1]
    val_hessian[2, 2, :] = eval_grad_dz_newton[:, 2]

    gauss_curvature = np.zeros(N_testing)
    mean_curvature = np.zeros(N_testing)
    # Evaluate curvatures
    for i in range(N_testing):
        denom1 = np.linalg.norm(val_grads[i, :]) ** 4
        adjoint_matrix = np.zeros((3, 3))
        adjoint_matrix[0, 0] = val_hessian[1, 1, i] * val_hessian[2, 2, i] - val_hessian[1, 2, i] * val_hessian[2, 1, i]
        adjoint_matrix[0, 1] = val_hessian[1, 2, i] * val_hessian[2, 0, i] - val_hessian[1, 0, i] * val_hessian[2, 2, i]
        adjoint_matrix[0, 2] = val_hessian[1, 0, i] * val_hessian[2, 1, i] - val_hessian[1, 1, i] * val_hessian[2, 0, i]
        adjoint_matrix[1, 0] = val_hessian[0, 2, i] * val_hessian[2, 1, i] - val_hessian[0, 1, i] * val_hessian[2, 2, i]
        adjoint_matrix[1, 1] = val_hessian[0, 0, i] * val_hessian[2, 2, i] - val_hessian[0, 2, i] * val_hessian[2, 0, i]
        adjoint_matrix[1, 2] = val_hessian[0, 1, i] * val_hessian[2, 0, i] - val_hessian[0, 0, i] * val_hessian[2, 1, i]
        adjoint_matrix[2, 0] = val_hessian[0, 1, i] * val_hessian[1, 2, i] - val_hessian[0, 2, i] * val_hessian[1, 1, i]
        adjoint_matrix[2, 1] = val_hessian[1, 0, i] * val_hessian[0, 2, i] - val_hessian[0, 0, i] * val_hessian[1, 2, i]
        adjoint_matrix[2, 2] = val_hessian[0, 0, i] * val_hessian[1, 1, i] - val_hessian[0, 1, i] * val_hessian[1, 0, i]

        numer1 = np.dot(val_grads[i, :], np.dot(adjoint_matrix, np.transpose(val_grads[i, :])))

        gauss_curvature[i] = numer1 / denom1

        denom2 = 2.0 * np.linalg.norm(val_grads[i, :]) ** 3

        numer2 = np.dot(val_grads[i, :], np.dot(val_hessian[:, :, i], val_grads[i, :])) - (
                    np.linalg.norm(val_grads[i, :]) ** 2) * np.trace(val_hessian[:, :, i])

        mean_curvature[i] = numer2 / denom2

    return mean_curvature, gauss_curvature


def find_match_positions(larger_idx_set, smaller_idx_set):
    """this is different from the one in multi_index_utils.
        doesn't require either of the multi index set to be lex sorted.
    """
    nr_exp_smaller, spatial_dimension = smaller_idx_set.shape
    positions = np.zeros(nr_exp_smaller, dtype=np.int64)
    for i in range(nr_exp_smaller):
        idx1 = smaller_idx_set[i, :]
        search_pos = -1
        while 1:
            search_pos += 1
            idx2 = larger_idx_set[search_pos, :]
            if is_equal(idx1, idx2):
                positions[i] = search_pos
                break
    return positions


def is_equal(index1: np.ndarray, index2: np.ndarray) -> bool:
    """ tells weather multi-index 1 equal to index2
    """
    spatial_dimension = len(index1)
    for m in range(spatial_dimension - 1, -1, -1):  # from last to first dimension
        if index1[m] > index2[m]:
            return False
        if index1[m] < index2[m]:
            return False
    return True  # all equal
