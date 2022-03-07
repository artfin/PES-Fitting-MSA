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

    cpp_template = """
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>

#include <torch/script.h>

const double a0 = 2.0;

extern "C" {{
    void c_evmono(double* x, double* mono);
    void c_evpoly(double* mono, double* poly);
}}

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

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
private:
    const size_t NMON = {0};
    const size_t NPOLY = {1};

    const size_t NATOMS;
    const size_t NDIS;

    double *yij;
    double *mono;
    double *poly;

    StandardScaler xscaler;
    StandardScaler yscaler;

    torch::jit::script::Module model;
    at::Tensor t;
}};

NNPIP::NNPIP(const size_t NATOMS, std::string const& pt_fname)
    : NATOMS(NATOMS), NDIS(NATOMS * (NATOMS - 1) / 2)
{{
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


double NNPIP::pes(std::vector<double> const& x) {{
    double drx, dry, drz;

    size_t k = 0;

    for (size_t i = 0; i < NATOMS; ++i) {{
        for (size_t j = i + 1; j < NATOMS; ++j) {{

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

    return ytr * yscaler.std[0] + yscaler.mean[0];
}}

int main()
{{
    std::cout << std::fixed << std::setprecision(16);

    const size_t natoms = {6};
    NNPIP pes(natoms, "{7}");

    std::vector<double> x{{
        1.1935874160000000,  1.1935874160000000, -1.1935874160000000,
       -1.1935874160000000, -1.1935874160000000, -1.1935874160000000,
       -1.1935874160000000,  1.1935874160000000,  1.1935874160000000,
        1.1935874160000000, -1.1935874160000000,  1.1935874160000000,
        2.5980762113524745,  2.5980762113524740,  0.5200762113550002,
        2.5980762113524745,  2.5980762113524740,  4.6760762113549994,
        0.0000000000000000,  0.0000000000000000,  0.0000000000000000
    }};

    double energy = pes.pes(x);
    std::cout << "(model) y: " << energy << "\\n";

    return 0;
}}
    """.format(NMON, NPOLY, xscaler_mean, xscaler_std, yscaler_mean, yscaler_std, NATOMS, model_fname)

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

add_executable(
    load-model
    load_model.cpp
    {1}
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

    X0 = X[0].view((1, meta_info["NPOLY"]))
    X0 = xscaler.transform(X0)

    with torch.no_grad():
        ytr = model(X0)

    ytr = ytr * yscaler.std + yscaler.mean
    logging.info("Expected output of the exported model: {}".format(ytr.item()))
