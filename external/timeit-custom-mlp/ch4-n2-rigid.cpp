#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#include "ai_pes_ch4_n2_opt1.hpp"

// rigid case
#include "c_basis_4_2_1_4_intermolecular.h"
#include "c_jac_4_2_1_4_intermolecular.h"
const int natoms = 7;
const int ndist = natoms * (natoms - 1) / 2;
//const int npoly = 79; // rigid case, `old model` silu.npz
const int npoly = 78;

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

void make_yij_4_2_1_4_intermolecular(const double * x, double* yij, int natoms)
{
    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
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
            
            drx = x[3*i    ] - x[3*j    ]; 
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];
            
            double dst  = std::sqrt(drx*drx + dry*dry + drz*drz);
            double dst6 = dst * dst * dst * dst * dst * dst;
            double s    = sw(dst);

            yij[k] = (1.0 - s) * std::exp(-dst / 2.0) + s * 1e4 / dst6;
            k++;
        }
    }
}

// probably a terrible hack but seems to be working for now...
#define EVPOLY     evpoly_4_2_1_4_intermolecular
#define EVPOLY_JAC evpoly_jac_4_2_1_4_intermolecular
#define MAKE_YIJ   make_yij_4_2_1_4_intermolecular
#define MAKE_DYDR  void();
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

void print_molpro_style(std::vector<double> const& cc) {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "1, H1,, " << cc[0]  << ", " << cc[1]  << ", " << cc[2]  << "\n";
    std::cout << "2, H2,, " << cc[3]  << ", " << cc[4]  << ", " << cc[5]  << "\n";
    std::cout << "3, H3,, " << cc[6]  << ", " << cc[7]  << ", " << cc[8]  << "\n";
    std::cout << "4, H4,, " << cc[9]  << ", " << cc[10] << ", " << cc[11] << "\n";
    std::cout << "5, N1,, " << cc[12] << ", " << cc[13] << ", " << cc[14] << "\n";
    std::cout << "6, N2,, " << cc[15] << ", " << cc[16] << ", " << cc[17] << "\n";
    std::cout << "7, C1,, " << cc[18] << ", " << cc[19] << ", " << cc[20] << "\n";
}

double internal_pes_ch4_n2(MLPModel & model, double R, double PH1, double TH1, double PH2, double TH2)
{
    const double NN_BOND_LENGTH = 2.078; // a0
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

    //print_molpro_style(cart);

    return model.forward(cart);
}

void long_range_ch4_n2_morse()
{
    auto model = build_model_from_npz("models/ch4-n2-rigid-79-32-1-silu.npz");

    AI_PES_ch4_n2 symm_pes;
    symm_pes.init();
    
    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    const double INFVAL = internal_pes_ch4_n2(model, 1000.0, PH1, TH1, PH2, TH2); 
    std::cout << "INFVAL: " << INFVAL << std::endl;  

    std::vector<double> Rv = linspace(4.5, 30.0, 300);

    std::cout << std::fixed;

    for (size_t k = 0; k < Rv.size(); ++k) {
        double R = Rv[k];
        
        double symmval = symm_pes.pes(R, PH1, TH1, PH2, TH2);

        double nnval = internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2) - INFVAL;


        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(12) << std::setprecision(7) << nnval
                                                       << "\t" << std::right << std::setw(20) << std::setprecision(7) << symmval << "\n";
    }
}

void long_range_ch4_n2_to_plot()
{
    auto model = build_model_from_npz("models/ch4-n2-rigid-78-32-1-y=exp6.npz");
    
    AI_PES_ch4_n2 symm_pes;
    symm_pes.init();
    
    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;
    
    const double INFVAL = internal_pes_ch4_n2(model, 1000.0, PH1, TH1, PH2, TH2); 
    std::cout << "INFVAL: " << INFVAL << std::endl;  
    
    std::vector<double> Rv = linspace(4.5, 30.0, 300);
    
    for (size_t k = 0; k < Rv.size(); ++k) {
        double R = Rv[k];
        
        double symmval = symm_pes.pes(R, PH1, TH1, PH2, TH2);
        double nnval   = internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2) - INFVAL;

        //std::cout << std::fixed << std::setprecision(4) << R << " " << std::setprecision(6) << nnval << " " << symmval << "\n";
        std::cout << std::fixed << std::setprecision(4) << R << " " << std::setprecision(6) << nnval << "\n";
    }
}

void ch4_n2_derivatives_comparison()
{
    auto model = build_model_from_npz("models/ch4-n2-rigid-y=exp6.npz");
   
    AI_PES_ch4_n2 symm_pes;
    symm_pes.init();
    
    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;
    
    std::vector<double> Rv{5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0};

    const double INFVAL = internal_pes_ch4_n2(model, 1000.0, PH1, TH1, PH2, TH2); 
    std::cout << "INFVAL: " << INFVAL << std::endl;  

    std::vector<std::pair<double, double>> qcv_dr = {
        std::make_pair(5.00, -4891.258),
        std::make_pair(5.25, -2989.840),
        std::make_pair(5.5,  -1775.066),
    };

    std::vector<std::pair<double, double>> qcv_dph1 = {
        std::make_pair(5.00, 514.689),
    };

    std::cout << std::fixed;
    std::cout << "             DR                  DPH1                DTH1                DPH2               DTH2\n"; 
    
    for (size_t k = 0; k < Rv.size(); ++k) {
        double R = Rv[k];

        double symmval = symm_pes.pes(R, PH1, TH1, PH2, TH2);
        double nnval   = internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2) - INFVAL;
        
        const double dd = 1e-3;
        double nn_dR   = (internal_pes_ch4_n2(model, R + dd, PH1, TH1, PH2, TH2) - internal_pes_ch4_n2(model, R - dd, PH1, TH1, PH2, TH2)) / (2.0 * dd);
        double nn_dPH1 = (internal_pes_ch4_n2(model, R, PH1 + dd, TH1, PH2, TH2) - internal_pes_ch4_n2(model, R, PH1 - dd, TH1, PH2, TH2)) / (2.0 * dd);
        double nn_dTH1 = (internal_pes_ch4_n2(model, R, PH1, TH1 + dd, PH2, TH2) - internal_pes_ch4_n2(model, R, PH1, TH1 - dd, PH2, TH2)) / (2.0 * dd);
        double nn_dPH2 = (internal_pes_ch4_n2(model, R, PH1, TH1, PH2 + dd, TH2) - internal_pes_ch4_n2(model, R, PH1, TH1, PH2 - dd, TH2)) / (2.0 * dd);
        double nn_dTH2 = (internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2 + dd) - internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2 - dd)) / (2.0 * dd);

        double symm_dR   = (symm_pes.pes(R + dd, PH1, TH1, PH2, TH2) - symm_pes.pes(R - dd, PH1, TH1, PH2, TH2)) / (2.0 * dd);
        double symm_dPH1 = (symm_pes.pes(R, PH1 + dd, TH1, PH2, TH2) - symm_pes.pes(R, PH1 - dd, TH1, PH2, TH2)) / (2.0 * dd);
        double symm_dTH1 = (symm_pes.pes(R, PH1, TH1 + dd, PH2, TH2) - symm_pes.pes(R, PH1, TH1 - dd, PH2, TH2)) / (2.0 * dd);
        double symm_dPH2 = (symm_pes.pes(R, PH1, TH1, PH2 + dd, TH2) - symm_pes.pes(R, PH1, TH1, PH2 - dd, TH2)) / (2.0 * dd);
        double symm_dTH2 = (symm_pes.pes(R, PH1, TH1, PH2, TH2 + dd) - symm_pes.pes(R, PH1, TH1, PH2, TH2 - dd)) / (2.0 * dd);

        std::cout << " R = " << std::setprecision(2) << R << std::endl;     
       
        if (k == 0) {
            std::cout << "  (QC)   " << std::right << std::setw(10) << std::setprecision(5) << qcv_dr[k].second 
                                     << std::right << std::setw(19) << std::setprecision(5) << qcv_dph1[k].second << "\n";
        } else if (k < 3) { 
            std::cout << "  (QC)   " << std::right << std::setw(10) << std::setprecision(5) << qcv_dr[k].second << "\n";
        }

        std::cout << "  (NN)   " << std::right << std::setw(10) << std::setprecision(5) << nn_dR
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << nn_dPH1
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << nn_dTH1
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << nn_dPH2
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << nn_dTH2 << "\n";

        std::cout << "  (SYMM) " << std::right << std::setw(10) << std::setprecision(5) << symm_dR 
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << symm_dPH1
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << symm_dTH1
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << symm_dPH2
                  << "         " << std::right << std::setw(10) << std::setprecision(5) << symm_dTH2 << "\n";
    }
}

void long_range_ch4_n2_qc_table()
{
    auto model = build_model_from_npz("models/ch4-n2-rigid-78-32-1-y=exp.npz");
   
    AI_PES_ch4_n2 symm_pes;
    symm_pes.init();
    
    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    // minimal cross-section
    std::vector<std::pair<double, double>> qc_cs = {
        std::make_pair(  4.50,    6109.53939),
        std::make_pair(  4.55,    5522.09013),
        std::make_pair(  4.60,    4984.72456),
        std::make_pair(  4.65,    4493.50556),
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
    
    const double INFVAL = internal_pes_ch4_n2(model, 1000.0, PH1, TH1, PH2, TH2); 
    std::cout << "INFVAL: " << INFVAL << std::endl;  

    std::vector<double> Rv{4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0};

    std::cout << std::fixed;
    std::cout << "  R \t\t NN \t\t QC \t\t NN ERROR \t SYMMETRY-ADAPTED \t SYMMETRY-ADAPTED ERROR\n";


    for (size_t k = 0; k < Rv.size(); ++k) {
        double R = Rv[k];
        
        double qc      = qc_cs[k].second;
        double symmval = symm_pes.pes(R, PH1, TH1, PH2, TH2);
        double nnval   = internal_pes_ch4_n2(model, R, PH1, TH1, PH2, TH2) - INFVAL;

        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(12) << std::setprecision(3) << nnval
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(3) << qc
                                                       << "\t" << std::right << std::setw(12) << std::setprecision(2) << (nnval - qc) / qc * 100.0 << "%"
                                                       << "\t" << std::right << std::setw(20) << std::setprecision(3) << symmval
                                                       << "\t" << std::right << std::setw(20) << std::setprecision(2) << (symmval -qc) / qc * 100.0 << "%\n";

    }
}

void timeit()
{
    auto model = build_model_from_npz("models/ch4-n2-rigid-y=exp6.npz");
    
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
    
    size_t ncycles = 10000000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    for (size_t k = 0; k < ncycles; ++k) {
        cc[12] = 3.0 + dist(mt); cc[13] = 3.0 + dist(mt); cc[14] = 3.0 + dist(mt);
        cc[15] = 3.0 + dist(mt); cc[16] = 3.0 + dist(mt); cc[17] = 3.0 + dist(mt);

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
    // needs corrections to mlp.hpp to get this working 
    // since this model expects 79 polys instead of 78 (additional constant p(0)=1.0 is expected)
    //long_range_ch4_n2_morse();

    long_range_ch4_n2_to_plot();

    //long_range_ch4_n2_qc_table();

    //ch4_n2_derivatives_comparison();

    //timeit();

    return 0;
}

