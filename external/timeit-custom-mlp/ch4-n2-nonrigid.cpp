#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>

#include <Eigen/Dense>

#include "ai_pes_ch4_n2_opt1.hpp"

#include "c_basis_4_2_1_4_purify.h"
#include "c_jac_4_2_1_4_purify.h"
const int natoms = 7;
const int ndist = natoms * (natoms - 1) / 2;
const int npoly = 524; 

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

void make_yij_4_2_1_4_purify(const double * x, double* yij, int natoms)
{
    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            
            drx = x[3*i    ] - x[3*j    ]; 
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];
            
            double dst = std::sqrt(drx*drx + dry*dry + drz*drz);
            
            if (i == 0 && j == 1) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 H2
            if (i == 0 && j == 2) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 H3
            if (i == 0 && j == 3) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 H4
            if (i == 1 && j == 2) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H2 H3
            if (i == 1 && j == 3) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H2 H4
            if (i == 2 && j == 3) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H3 H4
            if (i == 0 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 C
            if (i == 1 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H2 C
            if (i == 2 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H3 C
            if (i == 3 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H4 C
            if (i == 4 && j == 5) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // N1 N2

            double dst6 = dst * dst * dst * dst * dst * dst;
            double s = sw(dst);
            yij[k] = (1.0 - s) * std::exp(-dst / 2.0) + s * 1e4 / dst6;

            k++;
        }
    }
}

// probably a terrible hack but seems to be working for now...
#define EVPOLY     evpoly_4_2_1_4_purify
#define EVPOLY_JAC evpoly_jac_4_2_1_4_purify
#define MAKE_YIJ   make_yij_4_2_1_4_purify
#include "mlp.hpp"

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

struct vec3 {
    double x, y, z;
};

struct XYZConfig {
    std::vector<vec3> coords;
    std::vector<vec3> qc_forces;

    char symbols[natoms];
    double qc_energy; 
};

std::vector<double> flatten_vec3(std::vector<vec3> const& coords) {
    size_t natoms = coords.size();
    std::vector<double> r(3*natoms);

    for (size_t j = 0; j < natoms; ++j) {
        r[3*j    ] = coords[j].x;
        r[3*j + 1] = coords[j].y;
        r[3*j + 2] = coords[j].z;
    }

    return r;
}

void forces_qc_comparison()
{
    auto model = build_model_from_npz("models/ch4-n2-nonrigid.npz");
    
    XYZConfig cc = {
        .coords = std::vector<vec3>{ 
            vec3{ 1.193587416000,  1.193587416000, -1.193587416000},
            vec3{-1.193587416000, -1.193587416000, -1.193587416000},
            vec3{-1.193587416000,  1.193587416000,  1.193587416000},
            vec3{ 1.193587416000, -1.193587416000,  1.193587416000},
            vec3{ 2.049166615657,  3.082245949322,  3.518554625153},
            vec3{ 3.518534506963,  3.082245949322,  2.049186733847},
            vec3{ 0.000000000000,  0.000000000000,  0.000000000000},
        },
        // 2nd-order stencil; CCSD(T)-F12a
        .qc_forces = std::vector<vec3>{
            vec3{  486.33132,   383.72498,   145.80773},
            vec3{   79.61198,   219.65616,    79.61133},
            vec3{  145.81320,   383.73622,   486.34633},
            vec3{  259.82105,    81.19282,   259.82041},
            vec3{-1536.66510, -1450.29512, -1250.09702},
            vec3{-1250.10429, -1450.31260, -1536.68443},
            vec3{ 1815.20228,  1832.28170,  1815.19702},
        },
        .symbols = {'H', 'H', 'H', 'H', 'N', 'N', 'C'},
        .qc_energy = 2071.75359,
    };

    int prec, width;

    prec = 6;
    width = prec + 3;
    std::cout << std::fixed << std::setprecision(prec);

    for (int j = 0; j < natoms; ++j) {
        std::cout << cc.symbols[j] << "  " << std::right << std::setw(width) << cc.coords[j].x 
                                   << " "  << std::right << std::setw(width) << cc.coords[j].y 
                                   << " "  << std::right << std::setw(width) << cc.coords[j].z << "\n";
    }
    std::cout << "\n";

    std::vector<double> coords = flatten_vec3(cc.coords);
    std::vector<double> forces = flatten_vec3(cc.qc_forces);

    double nn_energy = model.forward(coords);

    std::cout << ">> (QC; CCSD(T)-F12a)           energy: " << cc.qc_energy << " cm-1\n";
    std::cout << ">> (NN; 524-32-1, purify, exp6) energy: " << nn_energy << " cm-1\n\n"; 

    const double dd = 1e-3;
    std::vector<double> ccd(coords);
    double nnp, nnm, nnd;

    std::vector<double> nn_forces(3*natoms);

    for (int j = 0; j < 3 * natoms; ++j) {
        ccd[j] = coords[j] + dd;
        nnp = model.forward(ccd);
        
        ccd[j] = coords[j] - dd;
        nnm = model.forward(ccd);

        ccd[j] = coords[j];
        nn_forces[j] = (nnp - nnm) / (2.0 * dd);
    }
    
    prec = 3;
    width = prec + 7;
    std::cout << std::fixed << std::setprecision(prec);
    std::cout << "(QC) 2-point stencil; CCSD(T)-F12a\n";
    std::cout << "(NN) 2-point stencil\n";

    for (int j = 0; j < natoms; ++j) {
        std::cout << cc.symbols[j] << " (QC) " << std::right << std::setw(width) << cc.qc_forces[j].x
                                   << "      " << std::right << std::setw(width) << cc.qc_forces[j].y 
                                   << "      " << std::right << std::setw(width) << cc.qc_forces[j].z << "  cm-1/bohr\n"; 
        std::cout <<                 "  (NN) " << std::right << std::setw(width) << nn_forces[3*j    ]
                                   << "      " << std::right << std::setw(width) << nn_forces[3*j + 1] 
                                   << "      " << std::right << std::setw(width) << nn_forces[3*j + 2] << "  cm-1/bohr\n"; 
    }
}

double internal_pes_ch4_n2(MLPModel & model, double R, double PH1, double TH1, double PH2, double TH2, double NN_BOND_LENGTH)
{
    static std::vector<double> cart(21);

    cart[0] =  1.193587416; cart[1]  =  1.193587416; cart[2]  = -1.193587416; // H1
    cart[3] = -1.193587416; cart[4]  = -1.193587416; cart[5]  = -1.193587416; // H2
    cart[6] = -1.193587416; cart[7]  =  1.193587416; cart[8]  =  1.193587416; // H3
    cart[9] =  1.193587416; cart[10] = -1.193587416; cart[11] =  1.193587416; // H4

    // N1
    cart[12] = R * std::sin(TH1) * std::cos(PH1) - NN_BOND_LENGTH/2.0 * std::cos(PH2) * std::sin(TH2);
    cart[13] = R * std::sin(TH1) * std::sin(PH1) - NN_BOND_LENGTH/2.0 * std::sin(PH2) * std::sin(TH2);
    cart[14] = R * std::cos(TH1)                 - NN_BOND_LENGTH/2.0 * std::cos(TH2);

    // N2
    cart[15] = R * std::sin(TH1) * std::cos(PH1) + NN_BOND_LENGTH/2.0 * std::cos(PH2) * std::sin(TH2);
    cart[16] = R * std::sin(TH1) * std::sin(PH1) + NN_BOND_LENGTH/2.0 * std::sin(PH2) * std::sin(TH2);
    cart[17] = R * std::cos(TH1)                 + NN_BOND_LENGTH/2.0 * std::cos(TH2);

    cart[18] = 0.0; cart[19] = 0.0; cart[20] = 0.0;  // C

    return model.forward(cart);
}


void min_crossection_qc_table()
{
    auto model = build_model_from_npz("models/ch4-n2-nonrigid.npz");
    
    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    // minimal cross-section
    // CCSD(T)-F12a, aug-cc-pVTZ, N-N 2.078 a0
    std::vector<std::pair<double, double>> qc_cs = {
        std::make_pair(  4.50,    6109.53939    ),
        std::make_pair(  4.55,    5522.09013),
        std::make_pair(  4.60,    4984.72456),
        std::make_pair(  4.70,    4044.78150),
        std::make_pair(  4.75,    3635.16784), 
        std::make_pair(  5.00,    2071.86905),
        std::make_pair(  5.25,    1104.09456), 
        std::make_pair(  5.50,    519.967380), 
        std::make_pair(  5.75,    179.046967), 
        std::make_pair(  6.00,    -10.531815), 
        std::make_pair(  6.25,    -108.00501), 
        std::make_pair(  6.50,    -150.96692), 
        std::make_pair(  6.75,    -162.85652), 
        std::make_pair(  7.00,    -157.99741), 
        std::make_pair(  7.25,    -144.94302), 
        std::make_pair(  7.50,    -128.65527), 
        std::make_pair(  7.75,    -111.89408), 
        std::make_pair(  8.00,    -96.089158), 
        std::make_pair(  8.25,    -81.885364), 
        std::make_pair(  8.50,    -69.484145), 
        std::make_pair(  8.75,    -58.850515), 
        std::make_pair(  9.00,    -49.835928), 
        std::make_pair(  9.25,    -42.248063), 
        std::make_pair(  9.50,    -35.887667), 
        std::make_pair( 10.00,    -26.117203), 
        std::make_pair( 11.00,    -14.408236), 
        std::make_pair( 12.00,    -8.385055 ), 
        std::make_pair( 14.00,    -3.240298 ), 
        std::make_pair( 16.00,    -1.432077 ), 
        std::make_pair( 18.00,    -0.699601 ), 
        std::make_pair( 20.00,    -0.369563 ), 
        std::make_pair( 25.00,    -0.096164 ), 
        std::make_pair( 30.00,    -0.032139 ), 
    };

    // CCSD(T)-F12a, aug-cc-pVTZ, N-N 2.2 a0
    std::vector<std::pair<double, double>> qc_cs2 = {   
        std::make_pair(   4.50,  6116.8768839374 ),  
        std::make_pair(   4.55,  5533.1686585765 ),
        std::make_pair(   4.60,  4998.7421629067 ),
        std::make_pair(   4.70,  4062.7035095111 ),
        std::make_pair(   4.75,  3654.2409552208 ),
        std::make_pair(   5.00,  2091.5050183477 ),
        std::make_pair(   5.25,  1119.9042409639 ),
        std::make_pair(   5.50,   530.9121881913 ),
        std::make_pair(   5.75,   185.5690910430 ),
        std::make_pair(   6.00,    -7.4761446184 ),
        std::make_pair(   6.25,  -107.4087109801 ),
        std::make_pair(   6.50,  -151.9649222801 ),
        std::make_pair(   6.75,  -164.7815997947 ),
        std::make_pair(   7.00,  -160.3728124362 ),
        std::make_pair(   7.25,  -147.4507363204 ),
        std::make_pair(   7.50,  -131.0983818985 ),
        std::make_pair(   7.75,  -114.1617767496 ),
        std::make_pair(   8.00,   -98.1281170367 ),
        std::make_pair(   8.25,   -83.6792955846 ),
        std::make_pair(   8.50,   -71.0387987274 ),
        std::make_pair(   8.75,   -60.1844793082 ),
        std::make_pair(   9.00,   -50.9734071675 ),
        std::make_pair(   9.25,   -43.2148959217 ),
        std::make_pair(   9.50,   -36.7087055296 ),
        std::make_pair(  10.00,   -26.7110309890 ),
        std::make_pair(  11.00,   -14.7269878365 ),
        std::make_pair(  12.00,    -8.5642357820 ),
        std::make_pair(  14.00,    -3.3062821900 ),
        std::make_pair(  16.00,    -1.4601983585 ),
        std::make_pair(  18.00,    -0.7129183957 ),
        std::make_pair(  20.00,    -0.3764109050 ),
        std::make_pair(  25.00,    -0.0978750344 ),
        std::make_pair(  30.00,    -0.0326902645 ),
    };

    double NN1 = 2.078;
    double NN2 = 2.2;

    double INFVAL1 = internal_pes_ch4_n2(model, 1000.0, PH1, TH1, PH2, TH2, NN1); 
    double INFVAL2 = internal_pes_ch4_n2(model, 1000.0, PH1, TH1, PH2, TH2, NN2);

    std::cout << std::fixed << std::setprecision(12); 
    std::cout << "INFVAL1: " << INFVAL1 << "; INFVAL2: " << INFVAL2 << std::endl;  

    std::cout << "  (first => N-N 2.078 a0 [equilibrium]; second => N-N 2.2 a0)\n";
    std::cout << "  R \t\t NN \t\t QC \t     NN ERROR \t\t QC \t\t NN \t\t NN ERROR\n";

    for (size_t k = 0; k < qc_cs.size(); ++k) {
        double R      = qc_cs[k].first;
        double qc1    = qc_cs[k].second;
        double nnval1 = internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2, NN1) - INFVAL1;

        double qc2    = qc_cs2[k].second;
        double nnval2 = internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2, NN2) - INFVAL2;

        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(12) << std::setprecision(3) << nnval1
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(3) << qc1
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(2) << (nnval1 - qc1) / qc1 * 100.0 << "%"
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(3) << nnval2
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(3) << qc2
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(2) << (nnval2 - qc2) / qc2 * 100.0 << "%\n";

    }
}

void test_configs()
{
    auto model = build_model_from_npz("models/ch4-n2-nonrigid.npz");
    
    std::vector<double> cc1 = {
 	  1.7346750730,   0.1539914340,  1.0212816020, 
 	 -0.0339562100,   0.0671856230, -2.0245088950,
 	 -1.1583902060,  -1.4766887200,  0.8236432680,
 	 -1.2115972240,   1.5207603710,  0.6773417370,
 	  1.5657732230,   0.7225551450, -6.3406147620,
 	 -0.0826966530,  -0.5116781290, -6.6883467980,
 	  0.0000000000,   0.0000000000,  0.0000000000,
    };

    std::cout << "Expected: 327.551; model: " << model.forward(cc1) << "\n";
   
    std::vector<double> cc2 = { 
 	  1.0619305090, -1.4978434750,  0.8988444250, 
 	 -0.4152058120, -0.1674049690, -2.0021679000,
 	 -1.5570330730,  0.7562955150,  1.0753934450,
 	  1.1006478950,  1.7132141150, -0.1226711960,
 	  3.2540594550, -3.3631139460,  5.0431203240,
 	  1.5487040380, -2.2789411300,  4.7016146590,
 	  0.0000000000,  0.0000000000,  0.0000000000,
    };

    std::cout << "Expected: 1343.658; model: " << model.forward(cc2) << "\n";
}

void timeit() {
    auto model = build_model_from_npz("models/ch4-n2-nonrigid.npz");
    
    std::vector<double> cc = {
 	  1.1935874160,	  1.1935874160, -1.1935874160,
 	 -1.1935874160,	 -1.1935874160, -1.1935874160,
 	 -1.1935874160,	  1.1935874160,  1.1935874160,
 	  1.1935874160,	 -1.1935874160,  1.1935874160,
 	  2.5980762114,   2.5980762114,  1.5590762114,
 	  2.5980762114,	  2.5980762114,  3.6370762114,
 	  0.0000000000,	  0.0000000000,  0.0000000000,
    };
  
    std::random_device rd;
    std::mt19937 mt(rd()); 
    std::uniform_real_distribution<> dist(0, 1); 
    
    size_t ncycles = 1000000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    for (size_t k = 0; k < ncycles; ++k) {
        for (size_t j = 0; j < 21; ++j) {
            cc[j] += 0.001 * dist(mt); 
        }

        start = std::chrono::high_resolution_clock::now();
        double out = model.forward(cc);
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    std::cout << "Elapsed nanoseconds:   " << elapsed << std::endl;
    std::cout << "Nanoseconds per cycle: " << elapsed / ncycles << std::endl;
}

int main()
{
    forces_qc_comparison();

    min_crossection_qc_table();

    test_configs();

    timeit();

    return 0;
}
