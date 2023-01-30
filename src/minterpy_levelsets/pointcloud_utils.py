"""
Implements several utility functions for working with pointcloud data
"""
import numpy as np
from numpy.random import Generator, PCG64

__all__ = ['points_on_ellipsoid', 'points_on_biconcave_disc', 'points_on_torus']

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


