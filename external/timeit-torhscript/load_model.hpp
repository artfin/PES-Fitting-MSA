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
    void evpoly(double x[], double p[]);
}

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
private:
    void cart2internal(std::vector<double> const& cart, double & R, double & ph1, double & th1, double & ph2, double & th2);

    const size_t NPOLY = 79;
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
    delete poly;
}

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

    evpoly(yij, poly);

    //for (size_t k = 0; k < 79; ++k) {
    //    std::cout << poly[k] << "\n";
    //}

    t = torch::from_blob(poly, {static_cast<long int>(NPOLY)}, torch::kDouble);
    return model.forward({t}).toTensor().item<double>();
}
