#include <omp.h>
#include <unistd.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

// Ethanol [group: 1 1 1 2 3 1]
#include "c_jac_1_1_1_2_3_1_3.h"
#include "c_basis_1_1_1_2_3_1_3.h"
const int natoms = 9;
const int ndist  = natoms * (natoms - 1) / 2;
const int npoly = 1898;

const double a0 = 2.0;

void make_yij_1_1_1_2_3_1(const double * x, double* yij, int natoms)
{
    double dst;
    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            drx = x[3*i    ] - x[3*j    ]; 
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];
            
            dst  = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-dst/a0); 
            k++;
        }
    }
}

// probably a terrible hack but seems to be working for now...
#define EVPOLY     evpoly_1_1_1_2_3_1_3
#define EVPOLY_JAC evpoly_jac_1_1_1_2_3_1_3
#define MAKE_YIJ   make_yij_1_1_1_2_3_1
#include "mlp.hpp"

void time_ethanol_energy_and_forces_omp() {

    const int maxNumThreads = omp_get_max_threads();
    omp_set_num_threads(maxNumThreads);

    int np = 20000;
    double st = omp_get_wtime();
    
    #pragma omp parallel
    {
        auto model = build_model_from_npz("models/ethanol-wf-32-ntrain=50000.npz");
    
        std::random_device rd;
        std::mt19937 mt(rd()); 
        std::uniform_real_distribution<> dist(0, 1); 

        std::vector<double> c = { 
            -0.48656453, -0.89968471,  0.33532281,
            -1.75840118,  1.72792291,  0.23343237,
             2.06343414, -0.81795535, -0.44480756,
            -0.62151551, -1.24660767,  2.59927346,
            -1.42294266, -2.532772  , -0.64166572,
            -3.71060312,  1.32700928,  1.02877824,
            -0.62999519,  3.16756379,  1.12270384,
            -2.01849358,  2.04032063, -1.81168631,
             2.39924183,  0.35599741, -2.01414636
        };

        int natoms = 9;
        std::vector<double> forces(3 * natoms);

        #pragma omp for 
        for (int i = 0; i < np; i++) {
            int tid = omp_get_thread_num();
            
            for (size_t natom = 0; natom < natoms; ++natom) {
                c[natom] += 0.001 * dist(mt);
            }

            double en = model.forward(c);
            model.backward(c, forces);
            //std::cout << "tid: " << tid << "; en: " << en << std::endl;
        }
    }

    double end = omp_get_wtime();

    double elapsed = end - st;
    double per_call = elapsed / np * 1e6;
    std::cout << "Elapsed " << elapsed << "s; per call: " <<  per_call  << " mcs \n";
}

void time_ethanol_forces() {
    auto model = build_model_from_npz("models/ethanol-32.npz");

    // QC energy: [5122.87413416]
    std::vector<double> c = { 
        -0.48656453, -0.89968471,  0.33532281,
        -1.75840118,  1.72792291,  0.23343237,
         2.06343414, -0.81795535, -0.44480756,
        -0.62151551, -1.24660767,  2.59927346,
        -1.42294266, -2.532772  , -0.64166572,
        -3.71060312,  1.32700928,  1.02877824,
        -0.62999519,  3.16756379,  1.12270384,
        -2.01849358,  2.04032063, -1.81168631,
         2.39924183,  0.35599741, -2.01414636
    };
    

    std::random_device rd;
    std::mt19937 mt(rd()); 
    std::uniform_real_distribution<> dist(0, 1); 
    
    size_t ncycles;
    double en;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    std::vector<double> d(3 * natoms);
    
    ncycles = 10000;
    for (size_t k = 0; k < ncycles; ++k) {
        for (size_t natom = 0; natom < natoms; ++natom) {
            c[natom] += 0.01 * dist(mt);
        }

        start = std::chrono::high_resolution_clock::now();
        model.backward(c, d);
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    std::cout << std::fixed << std::setprecision(3); 
    std::cout << "> Force evaluation: " << elapsed / ncycles * 1e-3 << " mcs [total=" << elapsed * 1e-9 << "s]" << std::endl;
}

void time_ethanol_energy()
{
    auto model = build_model_from_npz("models/ethanol-32.npz");

    // QC energy: [5122.87413416]
    std::vector<double> c = { 
        -0.48656453, -0.89968471,  0.33532281,
        -1.75840118,  1.72792291,  0.23343237,
         2.06343414, -0.81795535, -0.44480756,
        -0.62151551, -1.24660767,  2.59927346,
        -1.42294266, -2.532772  , -0.64166572,
        -3.71060312,  1.32700928,  1.02877824,
        -0.62999519,  3.16756379,  1.12270384,
        -2.01849358,  2.04032063, -1.81168631,
         2.39924183,  0.35599741, -2.01414636
    };
    
    std::random_device rd;
    std::mt19937 mt(rd()); 
    std::uniform_real_distribution<> dist(0, 1); 
    
    size_t ncycles;
    double en;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    ncycles = 50000;
    for (size_t k = 0; k < ncycles; ++k) {
        for (size_t natom = 0; natom < natoms; ++natom) {
            c[natom] += 0.01 * dist(mt);
        }

        start = std::chrono::high_resolution_clock::now();
        en = model.forward(c);
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
   
    std::cout << std::fixed << std::setprecision(3); 
    std::cout << "> Energy evaluation: " << elapsed / ncycles * 1e-3 << " mcs [total=" << elapsed * 1e-9 << "s]" << std::endl;
}

void check_ethanol_model()
{
    const std::string model_name = "ethanol-wf-120.npz";
    std::cout << "MODEL NAME: " << model_name << "\n";
    auto model = build_model_from_npz("models/" + model_name);
   
    std::vector<double> c = {
        -0.4761826 ,  0.45898619,  0.73695564,  
        -1.73022068,  0.05322867, -1.94216265,
         2.0259864 , -0.4448517 ,  1.15635829,
        -0.85455832,  2.47620421,  1.12702922,
        -1.45137251, -0.65367861,  2.24923022,
        -3.46611077,  1.1125507 , -1.5203589 ,
        -0.46206592,  0.59411205, -3.37373338,
        -2.69442806, -1.85576319, -2.18146213,
         3.05913588, -0.71856286, -0.2946519
    };
    
    std::vector<double> fqc = { 
        -2901.01160391, -6328.93587572, -1304.75909643,  
        -7776.67168787, -7720.1621886 , 14827.21338247,
          287.36616892,  1584.63050094,  4245.61757989,
         3894.82333932,   552.50839805,  1016.73137371,
        -1976.36425186,  2246.94402647, -2434.620439  ,
          302.86573437,  3651.27764088, -4925.545509  ,
         1788.0732345 ,  2882.63321238, -6272.12761022,
         5936.95457564,  2132.36290733,  -596.58068577,
          478.52534213,   988.35711758, -4599.71958949,
    };

    double en_nn = model.forward(c);
    double en_qc = 5337.1027;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << ">> (QC; DFT)              energy: " << en_qc << " cm-1\n";
    std::cout << ">> (NN; 1898-120-1, en+f) energy: " << en_nn << " cm-1\n\n"; 

    std::vector<double> auto_d(3 * natoms);
    model.backward(c, auto_d); 
  
    double num_d; 
    double dx = 1.0e-3;
    std::vector<double> cm(c);
    std::vector<double> cp(c);

    int width = 12;  
    std::cout << std::fixed << std::setprecision(3); 
    std::cout << "#nvar   automatic    numeric          QC" << std::endl;

    for (int nvar = 0; nvar < 3 * natoms; ++nvar) {

        cm[nvar] -= dx;
        cp[nvar] += dx;
        num_d = (model.forward(cp) - model.forward(cm)) / (2.0 * dx);

        std::cout <<        std::right << std::setw(3)     << nvar
                  << " " << std::right << std::setw(width) << auto_d[nvar] 
                  << " " << std::right << std::setw(width) << num_d 
                  << " " << std::right << std::setw(width) << -fqc[nvar] << "\n"; 

        cm[nvar] = c[nvar];
        cp[nvar] = c[nvar]; 
    }
}

int main()
{
    time_ethanol_energy_and_forces_omp();

    //check_ethanol_model();
    //time_ethanol_energy();
    //time_ethanol_forces();

    return 0;
}
