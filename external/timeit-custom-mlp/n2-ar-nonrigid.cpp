#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>

#include "c_basis_2_1_4_purify.h"
#include "c_jac_2_1_4_purify.h"
const int natoms = 3;
const int ndist = natoms * (natoms - 1) / 2;
const int npoly = 18; 

const double a0 = 2.0;

double sw(double x) {
    double x_i = 6.0;
    double x_f = 20.0;

    if (x < x_i) {
        return 0.0;
    } else if (x < x_f) {
        double r = (x - x_i) / (x_f - x_i);
        double r3 = r * r * r;
        double r4 = r3 * r;
        double r5 = r4 * r;

        return 10.0 * r3 - 15.0 * r4 + 6.0 * r5;
    } else {
        return 1.0;
    }
}

void make_yij_2_1_4_purify(const double * x, double* yij, int natoms)
{
    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            
            drx = x[3*i    ] - x[3*j    ]; 
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];
            
            double dst = std::sqrt(drx*drx + dry*dry + drz*drz);
            
            if (i == 0 && j == 1) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // N1 N2 

            double dst6 = dst * dst * dst * dst * dst * dst;
            double s = sw(dst);
            yij[k] = (1.0 - s) * std::exp(-dst / 2.0) + s * 1e4 / dst6;

            k++;
        }
    }
}

// probably a terrible hack but seems to be working for now...
#define EVPOLY     evpoly_2_1_4_purify
#define EVPOLY_JAC evpoly_jac_2_1_4_purify
#define MAKE_YIJ   make_yij_2_1_4_purify
#include "mlp.hpp"

double internal_pes_n2_ar(MLPModel & model, double R, double TH, double NN_BOND_LENGTH)
{
    static std::vector<double> cart(9);

    // N1
    cart[0] = NN_BOND_LENGTH/2.0 * std::sin(TH);
    cart[1] = 0.0; 
    cart[2] = NN_BOND_LENGTH/2.0 * std::cos(TH);
   
    // N2 
    cart[3] = -NN_BOND_LENGTH/2.0 * std::sin(TH);
    cart[4] = 0.0; 
    cart[5] = -NN_BOND_LENGTH/2.0 * std::cos(TH);

    // Ar 
    cart[6] = 0.0; 
    cart[7] = 0.0; 
    cart[8] = R;

    return model.forward(cart);
}

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

void min_crossection()
{
    std::cout << std::fixed;
    auto model = build_model_from_npz("models/n2-ar-nonrigid-18-32-1.npz");
    
    const double deg = M_PI / 180.0;
    double TH = 90.0 * deg;
    
    double NN_BOND_LENGTH = 2.078;
    double INFVAL = internal_pes_n2_ar(model, 1000.0, TH, NN_BOND_LENGTH); 
   
    auto rr = linspace(4.0, 30.0, 1000);

    for (size_t k = 0; k < rr.size(); ++k) {
        double R     = rr[k];
        double nnval = internal_pes_n2_ar(model, R, TH, NN_BOND_LENGTH) - INFVAL;

        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(12) << std::setprecision(5) << nnval << "\n";
    }
}

void min_crossection_qc_table()
{
    auto model = build_model_from_npz("models/n2-ar-nonrigid-18-32-1.npz");
    
    const double deg = M_PI / 180.0;
    double TH = 90.0 * deg;

    // minimal cross-section
    // CCSD(T)-F12a, aug-cc-pVQZ, N-N 2.078 a0
    std::vector<std::pair<double, double>> qc_cs = {
        std::make_pair(  4.50,    6780.6163 ),
        std::make_pair(  4.75,    4086.6217 ),
        std::make_pair(  5.00,    2393.4013 ), 
        std::make_pair(  5.25,    1345.3775 ),
        std::make_pair(  5.50,     708.6346 ), 
        std::make_pair(  5.75,     330.8768 ), 
        std::make_pair(  6.00,     113.9343 ), 
        std::make_pair(  6.25,      -4.8055 ), 
        std::make_pair(  6.50,     -64.8119 ), 
        std::make_pair(  6.75,     -90.6401 ), 
        std::make_pair(  7.00,     -97.3346 ), 
        std::make_pair(  7.25,     -93.9383 ), 
        std::make_pair(  7.50,     -85.7549 ), 
        std::make_pair(  7.75,     -75.7954 ), 
        std::make_pair(  8.00,     -65.6894 ), 
        std::make_pair(  8.25,     -56.2536 ), 
        std::make_pair(  8.50,     -47.8374 ), 
        std::make_pair(  8.75,     -40.5330 ), 
        std::make_pair(  9.00,     -34.2994 ), 
        std::make_pair(  9.25,     -29.0336 ), 
        std::make_pair(  9.50,     -24.6117 ), 
        std::make_pair( 10.00,     -17.8126 ), 
        std::make_pair( 10.50,     -13.0496 ), 
        std::make_pair( 11.00,      -9.6906 ), 
        std::make_pair( 12.00,      -5.5648 ), 
        std::make_pair( 14.00,      -2.1092 ), 
        std::make_pair( 16.00,      -0.9204 ), 
        std::make_pair( 18.00,      -0.4456 ), 
        std::make_pair( 20.00,      -0.2338 ), 
        std::make_pair( 25.00,      -0.0601 ), 
        std::make_pair( 30.00,      -0.0199 ), 
    };

    double NN_BOND_LENGTH = 2.078;
    double INFVAL = internal_pes_n2_ar(model, 1000.0, TH, NN_BOND_LENGTH); 

    std::cout << std::fixed << std::setprecision(12); 
    std::cout << "INFVAL: " << INFVAL << "\n"; 

    std::cout << "  (N-N 2.078 a0 [equilibrium])\n";
    std::cout << "  R \t\t NN \t\t QC \t     NN ERROR\n";

    for (size_t k = 0; k < qc_cs.size(); ++k) {
        double R      = qc_cs[k].first;
        double qc    = qc_cs[k].second;
        double nnval = internal_pes_n2_ar(model, R, TH, NN_BOND_LENGTH) - INFVAL;

        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(12) << std::setprecision(3) << nnval
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(3) << qc
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(2) << (nnval - qc) / qc * 100.0 << "%\n";
    }
}

void test_configs()
{
    auto model = build_model_from_npz("models/n2-ar-nonrigid-18-32-1.npz");
    
    std::vector<double> cc1 = {
        0.6213774562, 0.0000000000,  0.8330670804,
       -0.6213774562, 0.0000000000, -0.8330670804,
        0.0000000000, 0.0000000000,  5.0000000000,
    };
  

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Expected:  8148.1626; model: " << model.forward(cc1) << "\n";
}

int main()
{
    //test_configs();
    min_crossection();
    //min_crossection_qc_table();

    return 0;
}
