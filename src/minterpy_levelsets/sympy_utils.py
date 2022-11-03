
import sympy as sp
from minterpy.core.utils import _get_poly_degree


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
    ]