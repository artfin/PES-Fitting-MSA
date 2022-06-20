import argparse
import logging
from pathlib import Path
import os
import torch
import yaml

from build_model import build_network_yaml
from dataset import PolyDataset
from genpip import cl
from train_model import load_dataset

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

class StandardScaler_impl(torch.nn.Module):
    def __init__(self, mean_=None, scale_=None):
        super(StandardScaler_impl, self).__init__()
        self.mean_ = mean_
        self.scale_ = scale_

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, xtr):
        return self.scale_ * xtr + self.mean_


class Evaluator(torch.nn.Module):
    def __init__(self, model, xscaler, yscaler, meta_info):
        super(Evaluator, self).__init__()
        self.model = model
        self.xscaler = xscaler
        self.yscaler = yscaler
        self.meta_info = meta_info

    def forward(self, X):
        self.model.eval()
        Xtr = self.xscaler.transform(X)
        ytr = self.model(Xtr)
        return self.yscaler.inverse_transform(ytr)


def retrieve_checkpoint(cfg, chk_path):
    checkpoint = torch.load(chk_path, map_location=torch.device('cpu'))
    meta_info = checkpoint["meta_info"]

    cfg_model = cfg['MODEL']
    model = build_network_yaml(cfg_model, input_features=meta_info["NPOLY"])
    model.load_state_dict(checkpoint["model"])

    xscaler        = StandardScaler_impl()
    xscaler.mean_  = torch.from_numpy(checkpoint['X_mean'])
    xscaler.scale_ = torch.from_numpy(checkpoint['X_std'])

    yscaler        = StandardScaler_impl()
    yscaler.mean_  = torch.from_numpy(checkpoint['y_mean'])
    yscaler.scale_ = torch.from_numpy(checkpoint['y_std'])

    evaluator = Evaluator(model, xscaler, yscaler, meta_info)
    return evaluator

class Export:
    DATASETS_EXTERNAL = "datasets/external"

    def __init__(self, cfg, evaluator, poly_source="CUSTOM"):
        self.evaluator   = evaluator
        self.poly_source = poly_source

        self.export_wd        = cfg['OUTPUT_PATH']
        self.DEFAULT_CPP_PATH = os.path.join(self.export_wd, "load_model.hpp")

        #LR_CPP = os.path.join("CH4-N2", "long-range", "lr_pes_ch4_n2.cpp")
        #LR_HPP = os.path.join("CH4-N2", "long-range", "lr_pes_ch4_n2.hpp")
        #cl(f"cp {LR_CPP} {export_wd}")
        #cl(f"cp {LR_HPP} {export_wd}")

    def run(self):
        self.generate_torchscript()
        self.generate_cpp()

        symmetry    = self.evaluator.meta_info["symmetry"]
        order       = self.evaluator.meta_info["order"]

        basis_fname = "c_basis_{}_{}.cc".format(symmetry.replace(' ', '_'), order) if self.poly_source == "CUSTOM" \
                      else "f_basis_{}_{}.f90".format(symmetry.replace(' ', '_'), order)

        basis_fpath = os.path.join(BASEDIR, self.DATASETS_EXTERNAL, basis_fname)
        assert os.path.isfile(basis_fpath), "basis file {} is not found".format(basis_fpath)

        cl(f"cp {basis_fpath} {self.export_wd}")
        #self.generate_cmake(basis_fname)

    def generate_torchscript(self, torchscript_fname=None):
        if torchscript_fname is None:
            torchscript_fname = "torchscript-model.pt"

        torchscript_fpath = os.path.join(self.export_wd, torchscript_fname)

        logging.info("Tracing the model and saving the torchscript to {}".format(torchscript_fpath))
        NPOLY = self.evaluator.meta_info["NPOLY"]
        dummy = torch.rand(1, NPOLY, dtype=torch.float64)

        traced_script_module = torch.jit.trace(self.evaluator, dummy)
        traced_script_module.save(torchscript_fpath)

    def generate_cpp(self):
        logging.info("Saving generated cpp code to {}".format(self.cpp_path))

        #inf_poly = torch.zeros(1, NPOLY, dtype=torch.double)
        #inf_poly[0, 0] = 1.0
        #inf_poly = xscaler.transform(inf_poly)
        #inf_pred = model(inf_poly)
        #inf_pred = inf_pred * yscaler.std + yscaler.mean
        #inf_pred = inf_pred.item()

        logging.info("[NOTE] Long-range potential is not included in the export code.")
        logging.info("[NOTE] Constant at R=+infinity is not subtracted from the final value.")
        #logging.info("[NOTE] The code sets intermolecular Morse coordinates to zero.")

        NATOMS       = self.evaluator.meta_info["NATOMS"]

        if self.poly_source == "MSA":
            mask         = self.evaluator.meta_info.get("mask", None)
            NPOLY_TOTAL  = None

            if mask is not None:
                mask = mask.tolist()
                NPOLY_TOTAL = len(mask)

            NPOLY = self.evaluator.meta_info["NPOLY"]
            NMON  = self.evaluator.meta_info["NMON"]
            decl_apply_mask = ""
            body_apply_mask = ""
            decl_extern_poly = """
extern "C" {
    void c_evmono(double* x, double* mono);
    void c_evpoly(double* mono, double* poly);
}
"""
            mono_init = """
    mono = new double [NMON];
"""
            var_declarations = """
    const size_t NMON  = {};
    const size_t NPOLY = {};
""".format(NMON, NPOLY)

            poly_masked_init = ""
            free_vars = """
    delete mono;
"""
            poly_eval = """
    c_evmono(yij, mono);
    c_evpoly(mono, poly);
"""
            # decl_apply_mask = "void apply_mask(double* poly, double* poly_masked);"
            # body_apply_mask = """
            # void NNPIP::apply_mask(double* poly, double* poly_masked) {
            # size_t j = 0;
            # for (size_t k = 0; k < NPOLY; ++k) {
            # if (mask[k]) {
            # poly_masked[j] = poly[k];
            # j++;
            # }
            # }
            # }
            # """

            #//apply_mask(poly, poly_masked);
            # //t = torch::from_blob(poly_masked, {static_cast<long int>(NPOLY_MASKED)}, torch::kDouble);
            #poly_masked_init = """
            # poly_masked = new double [NPOLY_MASKED];
            # """
            # free_vars = """
            # delete poly_masked;
            # """

        elif self.poly_source == "CUSTOM":
            NPOLY_TOTAL = self.evaluator.meta_info["NPOLY"]
            NMON = 0
            NPOLY_MASKED = 0
            mask = []

            decl_apply_mask = ""
            body_apply_mask = ""
            decl_extern_poly = """
extern "C" {
    void evpoly(double x[], double p[]);
}
"""
            poly_masked = ""
            free_vars = ""
            poly_eval = """
    evpoly(yij, poly);
"""
            # const size_t NMON = """ + str(NMON) + """;
            # const size_t NPOLY = """ + str(NPOLY_TOTAL) + """;
            # const size_t NPOLY_MASKED = """ + str(NPOLY_MASKED) + """;
            # int mask[""" + str(NPOLY_TOTAL) + """] = {""" + ", ".join(list(map(str, mask))) + """};
            # double *poly_masked;
        else:
            raise ValueError("unreachable")

        cpp_template = """
#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>

#include <Eigen/Dense>

#include <torch/script.h>

const double a0 = 2.0;
const double NN_BOND_LENGTH = 2.078; // a0

const double HTOCM = 2.194746313702e5;
""" + decl_extern_poly + """

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

template <typename T>
std::vector<T> linspace(const T start, const T end, const size_t size) {

    if (size == 1) {
        return std::vector<T>{start};
    }

    const T step = (end - start) / (size - 1);

    std::vector<T> v(size);
    for (size_t k = 0; k < size; ++k) {
        v[k] = start + step * k;
    }

    return v;
}

class NNPIP
{
public:
    NNPIP(const size_t NATOMS, std::string const& pt_fname);
    ~NNPIP();

    double pes(std::vector<double> const& x);

    const size_t NATOMS;
private:""" + decl_apply_mask + """
    void cart2internal(std::vector<double> const& cart, double & R, double & ph1, double & th1, double & ph2, double & th2);
    """ + var_declarations + """
    const size_t NDIS;

    double *yij;
    double *mono;
    double *poly;

    torch::jit::script::Module model;
    at::Tensor t;
};

NNPIP::NNPIP(const size_t NATOMS, std::string const& pt_fname)
    : NATOMS(NATOMS), NDIS(NATOMS * (NATOMS - 1) / 2)
{
    yij = new double [NDIS];""" + \
    mono_init + """
    poly = new double [NPOLY];""" + \
    poly_masked_init + """

    try {
        model = torch::jit::load(pt_fname);
    } catch (const c10::Error& e) {
        std::cerr << ": ERROR: could not load the model\\n";
        exit(1);
    }

    // analogous to py:with torch.no_grad()
    torch::NoGradGuard no_grad;
}

NNPIP::~NNPIP()
{
    delete yij;""" + \
    free_vars + """
    delete poly;
}

void NNPIP::cart2internal(std::vector<double> const& cart, double & R, double & ph1, double & th1, double & ph2, double & th2)
{
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

    if (std::abs(sin_th1) < EPS) {
        ph1 = 0.0;
    } else {
        ph1 = std::atan2(center(1) / R / sin_th1, center(0) / R / sin_th1);
    }

    Eigen::Vector3d delta = 0.5 * (N2 - N1);
    double N2_len = delta.norm();

    th2 = std::acos(delta(2) / N2_len);
    double sin_th2 = std::sin(th2);

    if (std::abs(sin_th2) < EPS) {
        ph2 = 0.0;
    } else {
        ph2 = std::atan2(delta(1) / N2_len / sin_th2, delta(0) / N2_len / sin_th2);
    }
}
""" + body_apply_mask + """
double NNPIP::pes(std::vector<double> const& x) {
    double drx, dry, drz;

    size_t k = 0;

    for (size_t i = 0; i < NATOMS; ++i) {
        for (size_t j = i + 1; j < NATOMS; ++j) {
            if (i == 0 && j == 1) { yij[k] = 0.0; k = k + 1; continue; } // H1 H2
            if (i == 0 && j == 2) { yij[k] = 0.0; k = k + 1; continue; } // H1 H3
            if (i == 0 && j == 3) { yij[k] = 0.0; k = k + 1; continue; } // H1 H4
            if (i == 1 && j == 2) { yij[k] = 0.0; k = k + 1; continue; } // H2 H3
            if (i == 1 && j == 3) { yij[k] = 0.0; k = k + 1; continue; } // H2 H4
            if (i == 2 && j == 3) { yij[k] = 0.0; k = k + 1; continue; } // H3 H4
            if (i == 0 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H1 C
            if (i == 1 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H2 C
            if (i == 2 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H3 C
            if (i == 3 && j == 6) { yij[k] = 0.0; k = k + 1; continue; } // H4 C
            if (i == 4 && j == 5) { yij[k] = 0.0; k = k + 1; continue; } // N1 N2

            drx = ATOMX(x, i) - ATOMX(x, j);
            dry = ATOMY(x, i) - ATOMY(x, j);
            drz = ATOMZ(x, i) - ATOMZ(x, j);

            yij[k] = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-yij[k]/a0);
            k++;
        }
    }

    assert((k == NDIS) && ": ERROR: the morse variables vector is not filled properly.");

    """ + \
    poly_eval + """

    t = torch::from_blob(poly, {static_cast<long int>(NPOLY)}, torch::kDouble);
    return model.forward({t}).toTensor().item<double>();
}

double internal_pes(NNPIP & pes, double R, double PH1, double TH1, double PH2, double TH2)
{
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
}
    """

##### int main()
##### {
#####     std::cout << std::fixed << std::setprecision(16);
##### 
#####     const size_t natoms = """ + str(NATOMS) + """;
##### 
#####     const std::string torchscript_filename = """ + "\"" + self.torchscript_filename + "\"" + """;
#####     NNPIP nn_pes(natoms, torchscript_filename);
##### 
#####     /*
#####     const double deg = M_PI / 180.0;
#####     double PH1 = 47.912 * deg;
#####     double TH1 = 56.167 * deg;
#####     double PH2 = 0.0    * deg;
#####     double TH2 = 135.0  * deg;
#####     */
##### 
#####     double R = 6.25;
#####     const double PH1 = -1.5255882801233351;
#####     const double TH1 = 0.8272103029256817;
#####     const double PH2 = 1.3884766738202114;
#####     const double TH2 = 1.8628777065028299;
##### 
#####     double nnval = internal_pes(nn_pes, R, PH1, TH1, PH2, TH2);
#####     std::cout << R << " " << nnval << "\\n";
##### 
#####     /*
#####     std::vector<double> Rv = linspace(4.5, 30.0, 500);
##### 
#####     for (double R : Rv) {
#####         double nnval   = internal_pes(nn_pes, R, PH1, TH1, PH2, TH2);
#####         std::cout << R << " " << nnval << "\\n";
#####     }
#####     */
##### 
#####     return 0;
##### }

        with open(self.cpp_path, mode='w') as out:
            out.write(cpp_template)


    def generate_cmake(self, basis_fname):
        cmake_fname = os.path.join(self.export_wd, "CMakeLists.txt")
        logging.info("Generating CMakeLists.txt")

        CMAKE_PREFIX_PATH = "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/"

        cmake_template = """
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-model LANGUAGES CXX Fortran)

enable_language(C)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

set(CMAKE_CXX_FLAGS "-O2 -ggdb")

list(APPEND CMAKE_PREFIX_PATH """ + "\"" + CMAKE_PREFIX_PATH + "\"" + """)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(
    load-model
    load_model.hpp
    """ + basis_fname + """
)

target_link_libraries(load-model "${TORCH_LIBRARIES}")
set_property(TARGET load-model PROPERTY CXX_STANDARD 14)
    """

        with open(cmake_fname, mode='w') as out:
            out.write(cmake_template)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, type=str, help="path to folder with YAML configuration file")
    parser.add_argument("--model_name",   required=True, type=str, help="the name of the YAML configuration file [without extension]")
    args = parser.parse_args()

    MODEL_FOLDER = os.path.join(BASEDIR, args.model_folder)
    MODEL        = args.model_name

    assert os.path.isdir(MODEL_FOLDER), "Path to folder is invalid: {}".format(MODEL_FOLDER)

    cfg_path = os.path.join(MODEL_FOLDER, MODEL + ".yaml")
    assert os.path.isfile(cfg_path), "YAML configuration file does not exist at {}".format(cfg_path)

    with open(cfg_path, mode="r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    logging.info("loaded configuration file from {}".format(cfg_path))

    chk_path = os.path.join(MODEL_FOLDER, MODEL + ".pt")
    evaluator = retrieve_checkpoint(cfg, chk_path)

    torchscript_fname = MODEL + "-torchscript.pt"
    Export(cfg, evaluator, poly_source="MSA").generate_torchscript(torchscript_fname=torchscript_fname)

    cfg_dataset = cfg['DATASET']
    train, val, test = load_dataset(cfg_dataset)

    X0 = train.X[0].view((1, train.NPOLY))

    y0 = evaluator(X0)
    logging.info("Expected output of the exported model on the first configuration: {}".format(y0.item()))
    logging.info("Dataset energy value: {:.10f}".format(train.y[0].item()))


    #inf_poly = torch.zeros(1, NPOLY, dtype=torch.double)
    #inf_poly[0, 0] = 1.0
    #inf_poly = xscaler.transform(inf_poly)
    #inf_pred = model(inf_poly)
    #inf_pred = inf_pred * yscaler.std + yscaler.mean
