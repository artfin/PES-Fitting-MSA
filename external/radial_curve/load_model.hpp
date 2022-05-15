#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>

#include <Eigen/Dense>

#include <torch/script.h>

const double a0 = 2.0;
const double HTOCM = 2.194746313702e5;

extern "C" {
    void c_evmono(double* x, double* mono);
    void c_evpoly(double* mono, double* poly);
}


#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

class NNPIP
{
public:
    NNPIP(const size_t NATOMS, std::string const& pt_fname);
    ~NNPIP();

    double pes(std::vector<double> const& x);

    const size_t NATOMS;
private:
    void cart2internal(std::vector<double> const& cart, double & R, double & ph1, double & th1, double & ph2, double & th2);
    
    const size_t NMON  = 2892;
    const size_t NPOLY = 650;

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
    yij = new double [NDIS];
    mono = new double [NMON];

    poly = new double [NPOLY];

    try {
        model = torch::jit::load(pt_fname);
    } catch (const c10::Error& e) {
        std::cerr << ": ERROR: could not load the model\n";
        exit(1);
    }

    // analogous to py:with torch.no_grad()
    torch::NoGradGuard no_grad;
}

NNPIP::~NNPIP()
{
    delete yij;
    delete mono;

    delete poly;
}

double NNPIP::pes(std::vector<double> const& x) {
    double drx, dry, drz;

    size_t k = 0;

    for (size_t i = 0; i < NATOMS; ++i) {
        for (size_t j = i + 1; j < NATOMS; ++j) {
            drx = ATOMX(x, i) - ATOMX(x, j);
            dry = ATOMY(x, i) - ATOMY(x, j);
            drz = ATOMZ(x, i) - ATOMZ(x, j);

            yij[k] = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-yij[k]/a0);
            k++;
        }
    }

    assert((k == NDIS) && ": ERROR: the morse variables vector is not filled properly.");
    
    c_evmono(yij, mono);
    c_evpoly(mono, poly);

    t = torch::from_blob(poly, {static_cast<long int>(NPOLY)}, torch::kDouble);
    return model.forward({t}).toTensor().item<double>();
}
