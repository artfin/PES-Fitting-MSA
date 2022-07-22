import argparse
import dataclasses
import json
import os

@dataclasses.dataclass
class Monomial:
    coeffs  : list
    degrees : list

@dataclasses.dataclass
class Polynomial:
    id:        int
    degree:    int
    nmon:      int
    monomials: list

def load_polynomials(fname):
    with open(fname, mode='r') as fp:
        lines = fp.readlines()

    NLINES = len(lines)

    polynomials = []

    k = 0
    while k < NLINES:
        line = lines[k]
        meta, mono = line.split(':')
        id, degree, nmon, _ = list(map(int, meta.split()))

        degrees = list(map(int, mono.split()))
        coeffs = [1.0 if degree > 0 else 0.0 for degree in degrees]
        mono = Monomial(coeffs=coeffs, degrees=degrees)

        if len(polynomials) > 0 and polynomials[-1].id == id:
            polynomials[-1].monomials.append(mono)
        else:
            p = Polynomial(id=id, degree=degree, nmon=nmon, monomials=[mono])
            polynomials.append(p)

        k = k + 1

    return polynomials

def simplify_mono(m):
    """
    Aggregate zero-order terms together

    coeff = [1, 1, 1], deg = [0, 0, 1] -> coeff = [0, 0, 1], deg = [0, 0, 1]
    """
    nvars = len(m.degrees)

    # zero order monomial
    if sum(m.degrees) == 0:
        coeffs    = [0] * nvars
        coeffs[0] = sum(m.coeffs)
        return Monomial(coeffs=coeffs, degrees=[0]*nvars)

    for nvar in range(nvars):
        if m.degrees[nvar] == 0 and m.coeffs[nvar] == 1.0:
            m.coeffs[nvar] = 0.0

    return m

def diff_mono(m, nvar):
    """
    Differentiate monomial w.r.t #var
    """
    nvars = len(m.degrees)
    assert nvar < nvars

    coeff  = m.coeffs[nvar]
    degree = m.degrees[nvar]

    if degree == 0:
        return Monomial(coeffs=[0]*nvars, degrees=[0]*nvars)

    result = Monomial(coeffs=m.coeffs.copy(), degrees=m.degrees.copy())

    result.degrees[nvar] = degree - 1
    result.coeffs[nvar] = float(coeff * degree)

    return simplify_mono(result)

def diff_poly(p, nvar):
    """
    Differentiate polynomial w.r.t #var
    """
    derivative_mono = [diff_mono(mono, nvar) for mono in p.monomials]
    return Polynomial(id=p.id, degree=p.degree-1, nmon=len(derivative_mono), monomials=derivative_mono)

def generate_mono_expr(m):
    nvars = len(m.degrees)

    expr = ""
    for j in range(nvars):
        coeff  = m.coeffs[j]
        degree = m.degrees[j]

        if coeff > 0 and len(expr) == 0:
            expr += str(float(coeff))
            if degree > 0:
                expr += "*" + "*".join([f"x[{j}]"] * degree)
        elif coeff > 0 and len(expr) > 0:
            expr += "*" + str(float(coeff))
            if degree > 0:
                expr += "*" + "*".join([f"x[{j}]"] * degree)

    return expr

def generate_poly_expr(p):
    expr = ""
    for m in p.monomials:
        subexpr = generate_mono_expr(m)
        if len(expr) == 0 and len(subexpr) > 0:
            expr = subexpr
        elif len(expr) > 0 and len(subexpr) > 0:
            expr = expr + " + " + subexpr

    if len(expr) == 0:
        expr = "0.0"

    return expr

def generate_jac_proc(poly, jac_name="jac", skip_zeros=True):
    """
    Produces code for elements of jacobian matrix with dimensions [3 * natoms, npoly]

    Assumes Eigen format for selecting element of the matrix
    """
    nvars = len(poly[0].monomials[0].degrees)

    code = "void evpoly_jac(Eigen::Ref<Eigen::MatrixXd> jac, double* x) {\n"

    for indp, p in enumerate(poly):
        for nvar in range(nvars):
            pd = diff_poly(p, nvar)

            rhs = generate_poly_expr(pd)
            if rhs == "0.0":
                continue

            lhs = f"    {jac_name}({nvar}, {indp}) = "
            code += lhs + rhs + ";\n"

    return code + "}"

def generate_poly_proc(poly):
    return """
void evpoly(double* x, double* p) {{
{}
}}
""".format(
        "".join([
            "    p[{}] = {};\n".format(ind, generate_poly_expr(p))
            for ind, p in enumerate(poly)
        ])
    )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis_file", required=True, type=str, help="path to file with basis specification [.BAS]")
    parser.add_argument("--generate_poly", required=False, type=str2bool, help="whether to generate the C code for procedure to calculate polynomials")
    parser.add_argument("--generate_jac", required=False, type=str2bool, help="whether to generate the C code for procedure to calculate jacobian of polynomials")
    args = parser.parse_args()

    assert os.path.isfile(args.basis_file)
    poly = load_polynomials(args.basis_file)
    #poly = [poly[ind] for ind in poly_index]

    if args.generate_poly:
        proc = generate_poly_proc(poly)
        print(proc)

    if args.generate_jac:
        proc = generate_jac_proc(poly)
        print(proc)
