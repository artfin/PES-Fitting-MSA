import argparse
import dataclasses
import json
import os
import re

import itertools
import sympy as sp
from sympy.codegen.ast import Assignment

from tqdm import tqdm

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

def save_polynomials_to_bas(fname, poly):
    with open(fname, mode='w') as out:
        for p in poly:
            for id_mono, m in enumerate(p.monomials):
                degrees_str = " ".join([str(d) for d in m.degrees])
                out.write(f"{p.id} {p.degree} {p.nmon} {id_mono} : {degrees_str}\n")

def load_polynomials_from_bas(fname):
    assert fname.endswith('.BAS')

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
    result.coeffs[nvar] = int(coeff * degree)

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
            expr += str(int(coeff))
            if degree > 0:
                expr += "*" + "*".join([f"y[{j}]"] * degree)
        elif coeff > 0 and len(expr) > 0:
            expr += "*" + str(int(coeff))
            if degree > 0:
                expr += "*" + "*".join([f"y[{j}]"] * degree)

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

def make_drdx(natoms):
    ndist = natoms * (natoms - 1) // 2
    drdx = sp.zeros(3 * natoms, ndist)

    ts = sp.zeros(3*natoms, 3*natoms)
    for i, j in itertools.product(range(3*natoms), range(3*natoms)):
        ts[i, j] = sp.Symbol("t[{},{}]".format(i, j))

    k = 0
    for i, j in itertools.combinations(range(natoms), 2):
        drdx[3*i,     k] = sp.Symbol("dydx[{}, {}]".format(3*i, k))     #ts[3*i,     3*j    ]
        drdx[3*i + 1, k] = sp.Symbol("dydx[{}, {}]".format(3*i + 1, k)) # ts[3*i + 1, 3*j + 1]
        drdx[3*i + 2, k] = sp.Symbol("dydx[{}, {}]".format(3*i + 2, k)) # ts[3*i + 2, 3*j + 2]

        drdx[3*j,     k] = -sp.Symbol("dydx[{}, {}]".format(3*i, k)) # ts[3*i    , 3*j    ]
        drdx[3*j + 1, k] = -sp.Symbol("dydx[{}, {}]".format(3*i+1, k)) #ts[3*i + 1, 3*j + 1]
        drdx[3*j + 2, k] = -sp.Symbol("dydx[{}, {}]".format(3*i+2, k)) #ts[3*i + 2, 3*j + 2]

        k += 1


    return drdx

def generate_jac_proc_dpdx_full_cse(poly):
    ndist = len(poly[0].monomials[0].degrees)
    npoly = len(poly)

    natoms = 9
    drdx = make_drdx(natoms)
    dpdr = sp.zeros(ndist, npoly)

    # See issue: https://github.com/sympy/sympy/issues/15348 
    # set up variables to replace IndexedBase->MatrixSymbol
    # for code generation to work properly
    y        = sp.IndexedBase('y')
    y_matrix = sp.MatrixSymbol('y_m', ndist, 1)
    subs_vars = {y[k] : y_matrix[k] for k in range(ndist)}

    # Infinite generator yielding unique sympy.Symbols
    # to be used for labeling common subexpressions
    def symbols():
        n = 0
        while True:
            yield sp.Symbol(f"cse[{n}]")
            n = n + 1

    print("Creating symbolic Jacobian matrix of d(polynomials)/d(morse-variables")

    for indp, p in enumerate(tqdm(poly)):
        for nvar in range(ndist):
            poly_der = diff_poly(p, nvar)
            poly_der_expr = generate_poly_expr(poly_der)

            expr = sp.sympify(poly_der_expr, locals={'y': y})
            dpdr[nvar, indp] = expr

    dpdx = drdx * dpdr
    #dpdx_sym = sp.MatrixSymbol(f"dpdx[{indp}]", 3*natoms, 1)
    dpdx_sym = sp.MatrixSymbol(f"dpdx", 3 * natoms, npoly)

    print("drdx: {}".format(drdx.shape))
    print("dpdr: {}".format(dpdr.shape))
    print("dpdx: {}".format(dpdx.shape))

    # the following approach works but common subexpressions (CSE) are not eliminated
    #   > dpdx_sym = sp.MatrixSymbol('dpdx[0]', 3*natoms, 1)
    #   > print_ccode(dpdx, assign_to=dpdx_sym)

    # Perform common subexpression elimination on the
    # derivatives of the current polynomial
    gensym = symbols()
    replacements, reduced_exprs = sp.cse(dpdx, symbols=gensym)

    cse_code = ""
    for rep in replacements:
        eq = Assignment(rep[0], rep[1].subs(subs_vars))
        line = sp.ccode(eq)
        line = line.replace("y_m", "y")

        cse_code += line + "\n"

    red_expr_code = sp.ccode(reduced_exprs[0], assign_to=dpdx_sym)

    # Converting 1d indexing -> 2d indexing consistent with Eigen syntax
    pattern = r"dpdx\[(\d+)\]"
    red_expr_code = re.sub(pattern, lambda expr: "dpdx({0}, {1})".format(
        int(expr.groups()[0]) % npoly, int(expr.groups()[0]) // npoly
    ), red_expr_code)

    # eliminate zero elements of the jacobian
    red_expr_code = [line for line in red_expr_code.split("\n") if line and line.split('=')[1].strip() != "0;"]
    red_expr_code = "\n".join(red_expr_code)

    c_code = cse_code + red_expr_code

    # change formatting of temporary variables
    pattern = r"dydx\[(\d+), (\d+)\]"
    c_code = re.sub(pattern, lambda expr: "dydx({0}, {1})".format(
        expr.groups()[0], expr.groups()[1]
    ), c_code)

    decl = "void evpoly_jac(Eigen::Ref<Eigen::MatrixXd> dpdx, Eigen::Ref<Eigen::MatrixXd> dydx, double* y, double* cse) {\n"
    return decl + c_code + "\n}\n"

def generate_jac_proc_dpdx_partial_cse(poly):
    ndist = len(poly[0].monomials[0].degrees)
    npoly = len(poly)

    natoms = 9
    drdx = make_drdx(natoms)
    dpdr = sp.zeros(ndist, 1)

    # See issue: https://github.com/sympy/sympy/issues/15348 
    # set up variables to replace IndexedBase->MatrixSymbol
    # for code generation to work properly
    y        = sp.IndexedBase('y')
    y_matrix = sp.MatrixSymbol('y_m', ndist, 1)
    subs_vars = {y[k] : y_matrix[k] for k in range(ndist)}

    # Infinite generator yielding unique sympy.Symbols
    # to be used for labeling common subexpressions
    def symbols():
        n = 0
        while True:
            yield sp.Symbol(f"cse[{n}]")
            n = n + 1

    print("Creating symbolic Jacobian matrix of d(polynomials)/d(morse-variables")

    c_code = ""

    for indp, p in enumerate(tqdm(poly)):
        for nvar in range(ndist):
            poly_der = diff_poly(p, nvar)
            poly_der_expr = generate_poly_expr(poly_der)

            expr = sp.sympify(poly_der_expr, locals={'y': y})
            dpdr[nvar, 0] = expr

        dpdx = drdx * dpdr
        dpdx_sym = sp.MatrixSymbol(f"dpdx[{indp}]", 3*natoms, 1)

        # the following approach works but common subexpressions (CSE) are not eliminated
        #   > dpdx_sym = sp.MatrixSymbol('dpdx[0]', 3*natoms, 1)
        #   > print_ccode(dpdx, assign_to=dpdx_sym)

        # Perform common subexpression elimination on the
        # derivatives of the current polynomial
        gensym = symbols()
        replacements, reduced_exprs = sp.cse(dpdx, symbols=gensym)

        cse_code = ""
        for rep in replacements:
            eq = Assignment(rep[0], rep[1].subs(subs_vars))
            line = sp.ccode(eq)
            line = line.replace("y_m", "y")

            cse_code += line + "\n"

        red_expr_code = sp.ccode(reduced_exprs[0], assign_to=dpdx_sym)

        # Converting 1d indexing -> 2d indexing consistent with Eigen syntax
        pattern = r"dpdx\[(\d+)\]\[(\d+)\]"
        red_expr_code = re.sub(pattern, lambda expr: "dpdx({0}, {1})".format(
            int(expr.groups()[0]), int(expr.groups()[1])
        ), red_expr_code)

        # eliminate zero elements of the jacobian
        red_expr_code = [line for line in red_expr_code.split("\n") if line and line.split('=')[1].strip() != "0;"]
        red_expr_code = "\n".join(red_expr_code)

        c_code_iter = cse_code + red_expr_code

        # change formatting of temporary variables
        pattern = r"dydx\[(\d+), (\d+)\]"
        c_code_iter = re.sub(pattern, lambda expr: "dydx({0}, {1})".format(
            expr.groups()[0], expr.groups()[1]
        ), c_code_iter)

        c_code += c_code_iter + "\n"

    decl = "void evpoly_jac(Eigen::Ref<Eigen::MatrixXd> dpdx, Eigen::Ref<Eigen::MatrixXd> dydx, double* y, double* cse) {\n"
    return decl + c_code + "\n}\n"

def generate_jac_proc_dpdr(poly, jac_func_name, use_eigen_interface, skip_zeros=True):
    """
    Produces code for elements of jacobian matrix with dimensions [ndist, npoly]

    Assumes Eigen format for selecting element of the matrix
    """
    nvars = len(poly[0].monomials[0].degrees)

    if use_eigen_interface:
        decl = "extern \"C\" void {}(Eigen::Ref<Eigen::MatrixXd> jac, double* y) {{\n".format(jac_func_name)
    else:
        decl = "extern \"C\" void {}(double** jac, double* y) {{\n".format(jac_func_name)

    body = ""
    for indp, p in enumerate(poly):
        for nvar in range(nvars):
            pd = diff_poly(p, nvar)

            rhs = generate_poly_expr(pd)
            if rhs == "0.0":
                continue

            # remove annoying multiplications by 1
            rhs = rhs.replace("1*", "")

            if use_eigen_interface:
                lhs = f"    jac({nvar}, {indp}) = "
            else:
                lhs = f"    jac[{nvar}][{indp}] = "

            body += lhs + rhs + ";\n"

    return decl + body + "}"

def generate_poly_proc(poly, poly_func_name, use_eigen_interface):
    if use_eigen_interface:
        decl = "extern \"C\" void {}(double* y, Eigen::Ref<Eigen::RowVectorXd> p) {{\n".format(poly_func_name)
    else:
        decl = "extern \"C\" void {}(double* y, double* p) {{\n".format(poly_func_name)

    body = ""
    for ind, p in enumerate(poly):
        rhs = generate_poly_expr(p)
        rhs = rhs.replace("1*", "")

        if use_eigen_interface:
            body = body + "    p({}) = {};\n".format(ind, rhs)
        else:
            body = body + "    p[{}] = {};\n".format(ind, rhs)

    return decl + body + "}\n"

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
    parser.add_argument("--basis_file",          required=True,  type=str, help="path to file with basis specification [.BAS]")
    parser.add_argument("--use_eigen_interface", required=False, default=False, type=str2bool, help="whether to generate the C code for calculating polynomials using Eigen")
    parser.add_argument("--generate_poly",       required=False, default=False, type=str2bool, help="whether to generate the C code for procedure to calculate polynomials")
    parser.add_argument("--generate_jac",        required=False, default=False, type=str2bool, help="whether to generate the C code for procedure to calculate jacobian of polynomials")
    args = parser.parse_args()

    assert os.path.isfile(args.basis_file)
    poly = load_polynomials_from_bas(args.basis_file)

    stub = args.basis_file.split("MOL_")[1].split(".BAS")[0]

    if args.generate_poly:
        poly_func_name = "evpoly_{}".format(stub)

        print("Generating code for evaluating polynomials...")
        proc_code = generate_poly_proc(poly, poly_func_name, use_eigen_interface=args.use_eigen_interface)
        print("Finished.")

        poly_cc_fname = "c_basis_" + stub + ".cc"
        poly_h_fname  = "c_basis_" + stub + ".h"

        print("Writing generated code to CC={}".format(poly_cc_fname))
        print("Writing corresponding header file H={}".format(poly_h_fname))

        with open(poly_cc_fname, mode='w') as out:
            out.write("#include \"{}\"\n\n".format(poly_h_fname))
            out.write(proc_code)

        include_guard = "c_basis_" + stub + "_h"
        include_guard = include_guard.upper()

        with open(poly_h_fname, mode='w') as out:
            out.write("#ifndef {}\n".format(include_guard))
            out.write("#define {}\n".format(include_guard))
            out.write("\n")

            if args.use_eigen_interface:
                out.write("#include <Eigen/Dense>\n")
                out.write("extern \"C\" void {}(double* y, Eigen::Ref<Eigen::RowVectorXd> p);\n".format(poly_func_name))
            else:
                out.write("extern \"C\" void {}(double* y, double* p);\n".format(poly_func_name))

            out.write("\n")
            out.write("#endif")

    if args.generate_jac:
        jac_func_name = "evpoly_jac_{}".format(stub)

        print("Generating jacobian code...")
        jac_code = generate_jac_proc_dpdr(poly, jac_func_name, args.use_eigen_interface, skip_zeros=True)
        print("Finished.")

        jac_cc_fname = "c_jac_" + stub + ".cc"
        jac_h_fname  = "c_jac_" + stub + ".h"

        print("Writing generated code to CC={}".format(jac_cc_fname))
        print("Writing corresponding header file H={}".format(jac_h_fname))

        with open(jac_cc_fname, mode='w') as out:
            out.write("#include \"{}\"\n\n".format(jac_h_fname))
            out.write(jac_code)

        include_guard = "c_jac_" + stub + "_h"
        include_guard = include_guard.upper()
        with open(jac_h_fname, mode='w') as out:
            out.write("#ifndef {}\n".format(include_guard))
            out.write("#define {}\n".format(include_guard))
            out.write("\n")

            if args.use_eigen_interface:
                out.write("#include <Eigen/Dense>\n")
                out.write("extern \"C\" void {}(Eigen::Ref<Eigen::MatrixXd> jac, double* y);\n".format(jac_func_name))
            else:
                out.write("extern \"C\" void {}(double** jac, double* y);\n".format(jac_func_name))

            out.write("\n")
            out.write("#endif")
