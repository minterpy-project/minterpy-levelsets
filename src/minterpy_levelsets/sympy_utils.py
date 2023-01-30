"""
Implements convenient transformation functions between Sympy Polynomial and Minterpy Polynomial types.
"""
import numpy as np
import sympy as sp
from minterpy.core.utils import _get_poly_degree
from minterpy.core import MultiIndexSet
from minterpy import CanonicalPolynomial, get_transformation

__all__ = ['sympy_to_mp', 'mp_to_sympy']


def sympy_to_mp(poly, target_type=None):
    """converts a sympy Poly object to a minterpy polynomial in the target_type basis
    """
    given_coeffs = poly.coeffs()
    monoms = np.array(poly.monoms())

    # NOTE: The following two lines does not complete the MultiIndexSet as expected.
    #       Some indices are missing.
    # lex_sorted_monoms = mp.multi_index_utils.sort_lexicographically(monoms)
    # complete_monoms = mp.multi_index_utils.make_complete(lex_sorted_monoms)
    poly_deg = int(_get_poly_degree(monoms, 2.0))
    dim = monoms.shape[-1]
    mi = MultiIndexSet.from_degree(spatial_dimension=dim, poly_degree=poly_deg)
    complete_monoms = mi.exponents

    nr_coeffs, _ = complete_monoms.shape
    pos = find_match_positions(complete_monoms, monoms)

    coeffs = np.zeros(nr_coeffs)
    for i in range(len(given_coeffs)):
        coeffs[pos[i]] = given_coeffs[i]

    # By default return Canonical polynomial
    can_poly = CanonicalPolynomial(mi, coeffs)
    if target_type is None:
        return can_poly
    else:
        transform_to_target = get_transformation(can_poly, target_type)
        res_poly = transform_to_target()
        return res_poly


def mp_to_sympy(mp_poly):
    """converts a minterpy Polynomial object (in any basis) to a Sympy Poly object
    """

    # Convert to Canonical basis
    poly2can = get_transformation(mp_poly, CanonicalPolynomial)
    can_poly = poly2can()

    # Make symbol list
    num_coeffs = len(can_poly.coeffs)
    symbol_list = ""
    for i in range(can_poly.spatial_dimension):
        symbol_list += f"x_{i} "

    gen_list = sp.symbols(symbol_list)

    # Construct poly dict
    poly_dict = {tuple(can_poly.multi_index.exponents[p]): can_poly.coeffs[p] for p in range(num_coeffs)}

    poly = sp.Poly.from_dict(poly_dict, gen_list)
    return poly



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
