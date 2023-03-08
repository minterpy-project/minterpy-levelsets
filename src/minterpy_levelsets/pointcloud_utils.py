"""
Implements several utility functions for working with pointcloud data
"""
import numpy as np
from numpy.random import Generator, PCG64
from minterpy import NewtonPolynomial

__all__ = ['points_on_ellipsoid', 'points_on_biconcave_disc', 'points_on_torus', 'output_VTK', 'output_VTR',
           'sample_points', 'closest_points']

def points_on_ellipsoid(num_points: int, radius_x: float = 1.0, radius_y: float = 1.0,
                        radius_z: float = 1.0, random_seed: int = 42, verbose: bool = False):
    """Generates a pointcloud on an ellipsoid surface

    Parameters:
    num_points (int): Number of points to be generated
    radius_x (float): The radius along the X-axis
    radius_y (float): The radius along the Y-axis
    radius_z (float): The radius along the Z-axis
    random_seed (int): Seed for the random number generator
    verbose (bool): Print information on screen

    Returns:
    np.ndarray : The pointcloud generated with shape (num_points, 3)

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


def points_on_biconcave_disc(num_points: int, param_c: float = 0.5, param_d: float = 0.375,
                             random_seed: int = 42, verbose: bool = False):
    """Generates a pointcloud on a biconcave disc

    Parameters:
    num_points (int): Number of points to be generated
    param_c (float): Value of parameter 'c'
    param_d (float): Value of parameter 'd'
    random_seed (int): Seed for the random number generator
    verbose (bool): Print information on screen

    Returns:
    np.ndarray : The pointcloud generated with shape (num_points,3)

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


def points_on_torus(num_points: int, param_c: float = 0.5, param_a: float = 0.375,
                    random_seed: int = 42, verbose: bool = False):
    """Generates a pointcloud on a torus

    Parameters:
    num_points (int): Number of points to be generated
    param_c (float): Distance from center of torus to center of tube
    param_a (float): Radius of torus tube
    random_seed (int): Seed for the random number generator
    verbose (bool): Print information on screen

    Returns:
    np.ndarray : The pointcloud generated with shape (num_points,3)

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


def output_VTK(pointcloud, frame=0, prefix='pc_', scalar_field=None, vector_field=None):
    """Visualize the pointcloud, the normal vectors, and curvatures evaluted at all points

    Parameters
    ----------
    pointcloud (np.array): The pointcloud data
    frame (int): Frame number in a timeseries
    prefix (str): Custom prefix for output files
    scalar_field (np.array): A scalar field defined on all points
    vector_field (np.array): A vector field defined on all points
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
    """Visualize the surface as a VTK rectilinear grid

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

def sample_points(newt_poly, max_points, bounds=1.0, tol=1e-6, max_iters=10,
                  random_seed=42, grad_newt_poly=None):
    """ Randomly sample points on the zero isosurface of a given polynomial.
    """
    sampled_points = np.zeros((max_points, 3))
    if grad_newt_poly is None:
        dx_poly = newt_poly.diff([1, 0, 0])
        dy_poly = newt_poly.diff([0, 1, 0])
        dz_poly = newt_poly.diff([0, 0, 1])

        grad_newt_poly = NewtonPolynomial.from_poly(newt_poly,
                                                         new_coeffs=np.c_[dx_poly.coeffs,
                                                         dy_poly.coeffs,
                                                         dz_poly.coeffs])

    rg = Generator(PCG64(random_seed))

    spos = 0
    while spos < max_points:
        coord = 2.0 * bounds * rg.random(3) - bounds
        f = newt_poly(coord)
        iters = 0
        while np.abs(f) > tol and iters <= max_iters:
            dim = rg.integers(3)
            df = grad_newt_poly(coord)
            xnew = coord[dim] - f / df[dim]
            if np.any(np.abs(xnew) >= bounds):
                break
            coord[dim] = xnew
            f = newt_poly(np.array(coord))
            iters += 1

        if np.abs(f) < tol:
            sampled_points[spos, :] = coord
            spos += 1

    return sampled_points

def closest_points(newt_poly, grad_newt_poly, x0, tol = 1e-6, max_iter=10):
    x = x0.copy()
    phi_x = newt_poly(x)
    sign = np.sign(phi_x)
    grad_phi_norm2 = np.sum(grad_newt_poly(x) ** 2, axis=1)

    for p in range(x0.shape[0]):
        xp = x[p,:]
        for i in range(max_iter):
            grad_phi = grad_newt_poly(xp)
            y = xp - grad_phi * (phi_x[p] / grad_phi_norm2[p])
            dist = sign[p] * np.linalg.norm(y - x0[p])
            grad_phi_y = grad_newt_poly(y)
            dx = -dist * grad_phi_y / np.linalg.norm(grad_phi_y)
            xp += dx
            phi_x[p] = newt_poly(xp)
            grad_phi_norm2[p] = np.sum(grad_phi ** 2)
            err = np.sqrt(phi_x[p] ** 2 / grad_phi_norm2[p])
            if err < tol:
                break
            # # Update x0 and sign every 2 iterations for faster convergence
            # if i % 2 == 1:
            #     x0 = x
            #     phi_x = phi(x)
            #     sign = np.sign(phi_x)

    return x
