import logging
from pathlib import Path
import os
import torch

from eval_model import load_dataset
from eval_model import retrieve_checkpoint

from genpip import cl

from dataset import PolyDataset

def generate_torchscript(fname, model, NPOLY):
    dummy = torch.rand(1, NPOLY, dtype=torch.float64)
    logging.info("Tracing the model and saving the torchscript to {}".format(fname))

    traced_script_module = torch.jit.trace(model, dummy)
    traced_script_module.save(fname)


def generate_cpp(fname, model_fname, xscaler, yscaler, meta_info):
    logging.info("Generating cpp code to call the model to {}".format(fname))

    NPOLY  = meta_info["NPOLY"]
    NMON   = meta_info["NMON"]
    NATOMS = meta_info["NATOMS"]

    xscaler_mean = ", ".join("{:.16f}".format(xscaler.mean[0][k].item()) for k in range(NPOLY))
    xscaler_std  = ", ".join("{:.16f}".format(xscaler.std[0][k].item()) for k in range(NPOLY))

    yscaler_mean = "{:.16f}".format(yscaler.mean.item())
    yscaler_std  = "{:.16f}".format(yscaler.std.item())

    inf_poly = torch.zeros(1, NPOLY, dtype=torch.double)
    inf_poly[0, 0] = 1.0
    inf_poly = xscaler.transform(inf_poly)
    inf_pred = model(inf_poly)
    inf_pred = inf_pred * yscaler.std + yscaler.mean
    inf_pred = inf_pred.item()

    cpp_template = """
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>

#include <torch/script.h>

#include "lr_pes_ch4_n2.hpp"

const double a0 = 2.0;
const double NN_BOND_LENGTH = 2.078; // a0

const double HTOCM = 2.194746313702e5;

extern "C" {{
    void c_evmono(double* x, double* mono);
    void c_evpoly(double* mono, double* poly);
}}

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

template <typename T>
std::vector<T> linspace(const T start, const T end, const size_t size) {{

    if (size == 1) {
        return std::vector<T>{start};
    }

    const T step = (end - start) / (size - 1);

    std::vector<T> v(size);
    for (size_t k = 0; k < size; ++k) {{
        v[k] = start + step * k;
    }}

    return v;
}}

struct StandardScaler
{{
    StandardScaler() = default;

    void transform(double *x, size_t sz) {{

        assert(sz == mean.size());
        assert(sz == std.size());

        const double EPS = 1e-9;
        for (size_t k = 0; k < sz; ++k) {{
            x[k] = (x[k] - mean[k]) / (std[k] + EPS);
        }}
    }}

public:
    std::vector<double> mean;
    std::vector<double> std;
}};

class NNPIP
{{
public:
    NNPIP(const size_t NATOMS, std::string const& pt_fname);
    ~NNPIP();

    double pes(std::vector<double> const& x);

    const size_t NATOMS;
private:
    void cart2internal(std::vector<double> const& cart, double & R, double & ph1, double & th1, double & ph2, double & th2);

    const size_t NMON = {0};
    const size_t NPOLY = {1};

    const size_t NDIS;

    double *yij;
    double *mono;
    double *poly;

    StandardScaler xscaler;
    StandardScaler yscaler;

    LR_PES_ch4_n2 lr;

    torch::jit::script::Module model;
    at::Tensor t;
}};

NNPIP::NNPIP(const size_t NATOMS, std::string const& pt_fname)
    : NATOMS(NATOMS), NDIS(NATOMS * (NATOMS - 1) / 2)
{{
    lr.init();

    yij = new double [NDIS];
    mono = new double [NMON];
    poly = new double [NPOLY];

    try {{
        model = torch::jit::load(pt_fname);
    }} catch (const c10::Error& e) {{
        std::cerr << ": ERROR: could not load the model\\n";
        exit(1);
    }}

    // analogous to py:with torch.no_grad()
    torch::NoGradGuard no_grad;

    xscaler.mean = {{{2}}};
    xscaler.std  = {{{3}}};

    yscaler.mean = {{{4}}};
    yscaler.std  = {{{5}}};
}}

NNPIP::~NNPIP()
{{
    delete yij;
    delete mono;
    delete poly;
}}

void NNPIP::cart2internal(std::vector<double> const& cart, double & R, double & ph1, double & th1, double & ph2, double & th2)
{{
    assert(cart[0]  ==  1.193587416 && cart[1]  ==  1.193587416 && cart[2]  == -1.193587416); // H1
    assert(cart[3]  == -1.193587416 && cart[4]  == -1.193587416 && cart[5]  == -1.193587416); // H2
    assert(cart[6]  == -1.193587416 && cart[7]  ==  1.193587416 && cart[8]  ==  1.193587416); // H3
    assert(cart[9]  ==  1.193587416 && cart[10] == -1.193587416 && cart[11] ==  1.193587416); // H4
    assert(cart[18] ==  0.000000000 && cart[19] ==  0.000000000 && cart[20] ==  0.000000000); // C

    const double EPS = 1e-9;

    Eigen::Vector3d N1;
    N1 << cart[12], cart[13], cart[14];

    Eigen::Vector3d N2;
    N2 << cart[15], cart[16], cart[17];

    Eigen::Vector3d center = 0.5 * (N1 + N2);
    R = center.norm();

    th1 = std::acos(center(2) / R);
    double sin_th1 = std::sin(th1);

    if (std::abs(sin_th1) < EPS) {{
        ph1 = 0.0;
    }} else {{
        ph1 = std::atan2(center(1) / R / sin_th1, center(0) / R / sin_th1);
    }}

    Eigen::Vector3d delta = 0.5 * (N2 - N1);
    double N2_len = delta.norm();

    th2 = std::acos(delta(2) / N2_len);
    double sin_th2 = std::sin(th2);

    if (std::abs(sin_th2) < EPS) {{
        ph2 = 0.0;
    }} else {{
        ph2 = std::atan2(delta(1) / N2_len / sin_th2, delta(0) / N2_len / sin_th2);
    }}
}}

double NNPIP::pes(std::vector<double> const& x) {{
    double drx, dry, drz;

    size_t k = 0;

    for (size_t i = 0; i < NATOMS; ++i) {{
        for (size_t j = i + 1; j < NATOMS; ++j) {{
            if (i == 0 && j == 1) {{ yij[k] = 0.0; k = k + 1; continue; }} // H1 H2
            if (i == 0 && j == 2) {{ yij[k] = 0.0; k = k + 1; continue; }} // H1 H3
            if (i == 0 && j == 3) {{ yij[k] = 0.0; k = k + 1; continue; }} // H1 H4
            if (i == 1 && j == 2) {{ yij[k] = 0.0; k = k + 1; continue; }} // H2 H3
            if (i == 1 && j == 3) {{ yij[k] = 0.0; k = k + 1; continue; }} // H2 H4
            if (i == 2 && j == 3) {{ yij[k] = 0.0; k = k + 1; continue; }} // H3 H4
            if (i == 0 && j == 6) {{ yij[k] = 0.0; k = k + 1; continue; }} // H1 C
            if (i == 1 && j == 6) {{ yij[k] = 0.0; k = k + 1; continue; }} // H2 C
            if (i == 2 && j == 6) {{ yij[k] = 0.0; k = k + 1; continue; }} // H3 C
            if (i == 3 && j == 6) {{ yij[k] = 0.0; k = k + 1; continue; }} // H4 C
            if (i == 4 && j == 5) {{ yij[k] = 0.0; k = k + 1; continue; }} // N1 N2

            drx = ATOMX(x, i) - ATOMX(x, j);
            dry = ATOMY(x, i) - ATOMY(x, j);
            drz = ATOMZ(x, i) - ATOMZ(x, j);

            yij[k] = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-yij[k]/a0);
            k++;
        }}
    }}

    assert((k == NDIS) && ": ERROR: the morse variables vector is not filled properly.");

    c_evmono(yij, mono);
    c_evpoly(mono, poly);

    xscaler.transform(poly, NPOLY);

    t = torch::from_blob(poly, {{static_cast<long int>(NPOLY)}}, torch::kDouble);
    double ytr = model.forward({{t}}).toTensor().item<double>();

    double INF_PRED = {6};

    double R, ph1, th1, ph2, th2;
    cart2internal(x, R, ph1, th1, ph2, th2);

    double lrval = lr.pes(R, ph1, th1, ph2, th2) * HTOCM;

    double C = 15.0;
    double S = 1.0;
    double WT = 1.0 / (1.0 + std::exp(-S * (R - C)));

    return ytr * yscaler.std[0] + yscaler.mean[0] - INF_PRED + lrval * WT;
}}

double internal_pes(NNPIP & pes, double R, double PH1, double TH1, double PH2, double TH2)
{{
    static std::vector<double> cart(21);

    cart[0] =  1.193587416; cart[1]  =  1.193587416; cart[2]  = -1.193587416; // H1
    cart[3] = -1.193587416; cart[4]  = -1.193587416; cart[5]  = -1.193587416; // H2
    cart[6] = -1.193587416; cart[7]  =  1.193587416; cart[8]  =  1.193587416; // H3
    cart[9] =  1.193587416; cart[10] = -1.193587416; cart[11] =  1.193587416; // H4

    // N1
    cart[12] = R * std::sin(TH1) * std::cos(PH1) - NN_BOND_LENGTH * std::cos(PH2) * std::sin(TH2);
    cart[13] = R * std::sin(TH1) * std::sin(PH1) - NN_BOND_LENGTH * std::sin(PH2) * std::sin(TH2);
    cart[14] = R * std::cos(TH1)                 - NN_BOND_LENGTH * std::cos(TH2);

    // N2
    cart[15] = R * std::sin(TH1) * std::cos(PH1) + NN_BOND_LENGTH * std::cos(PH2) * std::sin(TH2);
    cart[16] = R * std::sin(TH1) * std::sin(PH1) + NN_BOND_LENGTH * std::sin(PH2) * std::sin(TH2);
    cart[17] = R * std::cos(TH1)                 + NN_BOND_LENGTH * std::cos(TH2);

    cart[18] = 0.0; cart[19] = 0.0; cart[20] = 0.0;                           // C

    return pes.pes(cart);
}}

int main()
{{
    std::cout << std::fixed << std::setprecision(16);

    const size_t natoms = 7;
    NNPIP nn_pes(natoms, "model.pt");

    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    std::vector<double> Rv = linspace(4.5, 30.0, 500);

    for (double R : Rv) {{
        double nnval   = internal_pes(nn_pes, R, PH1, TH1, PH2, TH2);
        std::cout << R << " " << nnval << "\\n";
    }}

    return 0;
}}
    """.format(NMON, NPOLY, xscaler_mean, xscaler_std, yscaler_mean, yscaler_std, inf_pred, NATOMS, model_fname)

    with open(fname, mode='w') as out:
        out.write(cpp_template)


def generate_cmake(fname, basis_fname, cmake_prefix_path="/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/"):
    logging.info("Generating CMakeLists.txt")

    cmake_template = """
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-model LANGUAGES CXX Fortran)

enable_language(C)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

set(CMAKE_CXX_FLAGS "-O2")

list(APPEND CMAKE_PREFIX_PATH "{0}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} ${{TORCH_CXX_FLAGS}}")

find_package(Eigen3)
include_directories(${{EIGEN3_INCLUDE_DIR}})

add_executable(
    load-model
    load_model.cpp
    {1}
    lr_pes_ch4_n2.hpp
    lr_pes_ch4_n2.cpp
)

target_link_libraries(load-model "${{TORCH_LIBRARIES}}")
set_property(TARGET load-model PROPERTY CXX_STANDARD 14)
    """.format(cmake_prefix_path, basis_fname)

    with open(fname, mode='w') as out:
        out.write(cmake_template)

def export_model(export_wd, dataset_wd, model, xscaler, yscaler, meta_info):
    logging.info("Export folder={}".format(export_wd))
    Path(export_wd).mkdir(parents=True, exist_ok=True)

    torchscript_fname = os.path.join(export_wd, "model.pt")
    generate_torchscript(torchscript_fname, model, meta_info["NPOLY"])

    cpp_fname = os.path.join(export_wd, "load_model.cpp")
    generate_cpp(cpp_fname, "model.pt", xscaler, yscaler, meta_info)

    symmetry = meta_info["symmetry"]
    order = meta_info["order"]
    basis_fname = "basis_{}_{}.f90".format(symmetry.replace(' ', '_'), order)
    cl(f"cp {dataset_wd}/{basis_fname} {export_wd}")

    LR_CPP = os.path.join("CH4-N2", "long-range", "lr_pes_ch4_n2.cpp")
    LR_HPP = os.path.join("CH4-N2", "long-range", "lr_pes_ch4_n2.hpp")
    cl(f"cp {LR_CPP} {export_wd}")
    cl(f"cp {LR_HPP} {export_wd}")

    cmake_fname = os.path.join(export_wd, "CMakeLists.txt")
    generate_cmake(cmake_fname, basis_fname)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    model, xscaler, yscaler, meta_info = retrieve_checkpoint(folder=".", fname="checkpoint.pt")
    export_model(export_wd="cpp-export", dataset_wd="CH4-N2", model=model, xscaler=xscaler, yscaler=yscaler, meta_info=meta_info)

    X, y = load_dataset("CH4-N2", "dataset.pt")

    X0 = X[-1].view((1, meta_info["NPOLY"]))
    X0 = xscaler.transform(X0)

    with torch.no_grad():
        ytr = model(X0)

    NPOLY = meta_info["NPOLY"]
    inf_poly = torch.zeros(1, NPOLY, dtype=torch.double)
    inf_poly[0, 0] = 1.0
    inf_poly = xscaler.transform(inf_poly)
    inf_pred = model(inf_poly)
    inf_pred = inf_pred * yscaler.std + yscaler.mean

    ytr = ytr * yscaler.std + yscaler.mean - inf_pred
    logging.info("Expected output of the exported model: {}".format(ytr.item()))
