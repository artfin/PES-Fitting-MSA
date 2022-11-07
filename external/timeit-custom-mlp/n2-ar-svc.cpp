#include <mpi.h>

#include <cassert>
#include <cstdint>
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
#define MAKE_DYDR  void();
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

const double BohrToAng = 0.529177210903;
const double Boltzmann = 1.380649e-23;             // SI: J * K^(-1)
const double Hartree   = 4.3597447222071e-18;      // SI: J
const double HkT       = Hartree/Boltzmann;        // to use as:  -V[a.u.]*`HkT`/T
const double HTOCM     = 2.194746313702 * 1E5;     // convert Hartree to cm^-1
const double VkT       = HkT / HTOCM;              // to use as:  -V[cm-1]*`VkT`/T
const double ALU       = 5.29177210903e-11;          // SI: m
const double ALU3      = ALU * ALU * ALU;
const double AVOGADRO  = 6.022140857 * 1e23;

const double LightSpeed_cm = 2.99792458e10;            // cm/s
const double EVTOCM        = 8065.73;
const double DALTON        = 1.66054e-27;              // kg
const double EVTOJ         = 1.602176565e-19;
const double ANGTOBOHR     = 1.0 / 0.529177249;

#define ARR_LEN(x)  (sizeof(x) / sizeof((x)[0]))

double pot_N2(double r)
/*
 * returns N2 potential [cm-1] approximated as a Morse curve
 * the parameters are derived from experiment
 * accepts the distance in bohr 
 */ 
{
    double rA = r * BohrToAng;

    // https://doi.org/10.1098/rspa.1956.0135 
    const double De    = 9.91; // eV
    const double omega = 2358.57; // cm-1
    const double nu    = 2.0 * M_PI * LightSpeed_cm * omega; // 1/s
    const double mu    = 14.003074004460 / 2.0 * DALTON;

    const double a  = sqrt(mu / (2.0 * De * EVTOJ)) * nu * 1e-10; // A
    const double re = 1.09768; // A

    return (De * EVTOCM) * (1 - exp(-a * (rA - re))) * (1 - exp(-a * (rA - re))); 
} 

double density_N2(double r, double T) {
    return exp(-pot_N2(r) * VkT / T);
}

double sample_N2(double T) 
/*
 * sample density function of N2 using rejection approach
 */
{
    static std::mt19937 generator; 
    static std::uniform_real_distribution<double> distr(0.0, 1.0);

    double EMIN_N2 = 0.0;    // cm-1
    double EMAX_N2 = 2000.0; // cm-1
                                   
    double x1 = 1.9, x2 = 2.2; // bohr
    double y1 = 0.0, y2 = 1.0; 

    double x, y, E;
    while (true) {
        x = distr(generator) * (x2 - x1) + x1;
        y = distr(generator) * (y2 - y1) + y1;

        if (y < density_N2(x, T)) {
            return x;
            //E = pot_N2(x);
            //if ((E > EMIN_N2) && (E < EMAX_N2)) return x;
        }
    }
}

#define NWALKERS 512 
#define DIM 6 
static double ensemble[NWALKERS][DIM];

#define BURNIN_LEN 10000
#define MAX_NPOINTS 250000000
#define THINNING 100
    
static std::random_device rd;
static std::mt19937 gen(rd()); 
static std::uniform_real_distribution<double> uni_distr(0.0, 1.0);
static std::uniform_int_distribution<int> int_distr(0, NWALKERS - 1);

void generate_initial_geom(double x[DIM]) {

    double l_N2 = 2.078;
    double phi   = uni_distr(gen) * 2.0 * M_PI;
    double theta = acos(uni_distr(gen) * 2.0 - 1.0);
    
    x[0] =  l_N2/2*cos(phi)*sin(theta); x[1] =  l_N2/2*sin(phi)*sin(theta); x[2] =  l_N2/2*cos(theta);
    x[3] = -l_N2/2*cos(phi)*sin(theta); x[4] = -l_N2/2*sin(phi)*sin(theta); x[5] = -l_N2/2*cos(theta);
}

int select_index_except_for(int k) {
    int ind;

    while (true) {
        ind = int_distr(gen); 
        if (ind != k) return ind;
    }
}

double propose_stretch_move(double dst[DIM], double walker[DIM], double c_walker[DIM]) 
// c_walker: complementary walker
{
    // Goodman & Weare (2010): adjustable scale parameter set to 2
    double a = 1.05; 

    double t = (a - 1.0) * uni_distr(gen) + 1;
    double z = t * t / a;

    for (int k = 0; k < DIM; ++k) {
        dst[k] = c_walker[k] + z * (walker[k] - c_walker[k]);  
    }

    return z;
}

double density(double x[DIM], double T)  {
    double l_N2 = sqrt((x[0]-x[3])*(x[0]-x[3]) + (x[1]-x[4])*(x[1]-x[4]) + (x[2]-x[5])*(x[2]-x[5]));
    double V = pot_N2(l_N2);
    return exp(-V * VkT / T);
}

double make_ensemble_step(double T) {
    // candidate position

    double c[DIM];

    int acc = 0;
    
    for (int k = 0; k < NWALKERS; ++k) {
        int j = select_index_except_for(k);
        double z = propose_stretch_move(c, ensemble[k], ensemble[j]);
        
        double q = pow(z, DIM - 1) * density(c, T) / density(ensemble[k], T);
   
        double u = uni_distr(gen); 
        if (u < q) {
            memcpy(ensemble[k], c, sizeof(double) * DIM);
            acc++;
        }
    }

    //for (int k = 0; k < NWALKERS; ++k) {
    //    printf("0: %.6f %.6f %.6f\n", ensemble[k][0], ensemble[k][1], ensemble[k][2]);
    //    printf("1: %.6f %.6f %.6f\n", ensemble[k][3], ensemble[k][4], ensemble[k][5]);
    //    
    //    double l_N2 = sqrt((ensemble[k][0]-ensemble[k][3])*(ensemble[k][0]-ensemble[k][3]) + \
    //                       (ensemble[k][1]-ensemble[k][4])*(ensemble[k][1]-ensemble[k][4]) + \
    //                       (ensemble[k][2]-ensemble[k][5])*(ensemble[k][2]-ensemble[k][5]));
    //    printf("len: %.6f\n\n", l_N2); 
    //}

    return acc / (double) NWALKERS;
}

double burnin(double T) {
    double acc = 0.0;

    for (int k = 0; k < BURNIN_LEN; ++k) {
        acc += make_ensemble_step(T);
    }

    return acc / BURNIN_LEN;
}

void mixed_ensemble_nonrigid_svc() {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto model = build_model_from_npz("models/n2-ar-nonrigid-18-32-1.npz");
    double INFVAL = internal_pes_n2_ar(model, 1000.0, 0.0, 2.078); 

    double RMAX_INT = 40.0;
    double V = 4.0 / 3.0 * M_PI * RMAX_INT * RMAX_INT * RMAX_INT * ALU3;
    
    double Tref[] = {510.0, 410.0, 310.0, 210.0, 160.0, 130.0, 110.0};
    int NTR = ARR_LEN(Tref);

    double TT[] = {100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 
                          210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
                          310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 
                          410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0};
    int NT = ARR_LEN(TT);
        
    double f[NT];
    double f2[NT];
    double nstar[NT];
    memset(f, 0, NT * sizeof(double));
    memset(f2, 0, NT * sizeof(double));
    memset(nstar, 0, NT * sizeof(double));
 
    std::vector<double> x(9);

    for (int l = 0; l < NTR; ++l) {
        for (int k = 0; k < NWALKERS; ++k) {
            generate_initial_geom(ensemble[k]);
        }

        double acc_rate = burnin(Tref[l]);

        if (world_rank == 0) {
            printf(" Tref = %.2f\n", Tref[l]);
            printf(" Run burn-in for %d steps with stretch move (GW10).\n", BURNIN_LEN);
            printf("      acceptance rate: %f\n", acc_rate);
        }

        uint64_t acc = 0; 
        double en;

        for (uint64_t i = 0; acc < MAX_NPOINTS; i++) {
            make_ensemble_step(Tref[l]); 

            if (i % THINNING == 0) {
                for (int k = 0; k < NWALKERS; ++k) {
                    double R3 = uni_distr(gen) * RMAX_INT * RMAX_INT * RMAX_INT;
                    double R  = pow(R3, 1.0/3.0);
                        
                    double Phi   = uni_distr(gen) * 2.0 * M_PI;
                    double Theta = acos(uni_distr(gen) * 2.0 - 1.0);

                    x[0] = ensemble[k][0];        x[1] = ensemble[k][1];        x[2] = ensemble[k][2];
                    x[3] = ensemble[k][3];        x[4] = ensemble[k][4];        x[5] = ensemble[k][5];
                    x[6] = R*cos(Phi)*sin(Theta); x[7] = R*sin(Phi)*sin(Theta); x[8] =  R*cos(Theta);
                    
                    double l_N2  = sqrt((x[0]-x[3])*(x[0]-x[3]) + (x[1]-x[4])*(x[1]-x[4]) + (x[2]-x[5])*(x[2]-x[5]));
                    double en_N2 = pot_N2(l_N2);

                    double fi, w;
                    for (int j = 0; j < NT; ++j) {
                        if (TT[j] < Tref[l] + 1.0) {
                            if (R < 4.5) {
                                fi = 1.0;
                            } else {
                                en = model.forward(x) - INFVAL;
                                fi = (1.0 - exp(-en * VkT / TT[j]));
                            }

                            w = exp(-en_N2 * VkT / TT[j]) / exp(-en_N2 * VkT / Tref[l]);

                            nstar[j] += w;
                            f[j]     += fi * w;
                            f2[j]    += (fi * w) * (fi * w);
                        }
                    }

                    acc++;
                }
            }
        }

        if (world_rank == 0) {
            printf("Tref = %.2f done. [0] Collected %d points.\n", Tref[l], acc);
        }

        double f_t[NT];
        double f2_t[NT];
        double nstar_t[NT];
        memset(f_t, 0, NT * sizeof(double));
        memset(f2_t, 0, NT * sizeof(double));
        memset(nstar_t, 0, NT * sizeof(double));

        for (int j = 0; j < NT; ++j) {
            f_t[j]     = f[j] *  V / 2.0 * AVOGADRO * 1e6;
            f2_t[j]    = f2[j] * (V / 2.0 * AVOGADRO * 1e6) * (V / 2.0 * AVOGADRO * 1e6);
            nstar_t[j] = nstar[j];
        }
        
        double f_t_root[NT];
        double f2_t_root[NT];
        double nstar_t_root[NT];
        memset(f_t_root, 0, NT * sizeof(double));
        memset(f2_t_root, 0, NT * sizeof(double));
        memset(nstar_t_root, 0, NT * sizeof(double));

        MPI_Reduce(f_t,     f_t_root,     NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(f2_t,    f2_t_root,    NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nstar_t, nstar_t_root, NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            for (int j = 0; j < NT; ++j) {
                f_t_root[j] /= nstar_t_root[j];
                f2_t_root[j] /= nstar_t_root[j];

                double sigma = sqrt((f2_t_root[j] - f_t_root[j]*f_t_root[j]) / (nstar_t_root[j] - 1));
                printf("> T = %.2f: nstar = %.2f, svc = %.6f +- %.6f\n", TT[j], nstar_t_root[j], f_t_root[j], sigma);
            }
        }
    }

    double f_root[NT];
    double f2_root[NT];
    double nstar_root[NT];
    memset(f_root, 0, NT * sizeof(double));
    memset(f2_root, 0, NT * sizeof(double));
    memset(nstar_root, 0, NT * sizeof(double));

    MPI_Reduce(f,     f_root,     NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(f2,    f2_root,    NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nstar, nstar_root, NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\n\n");
        printf("Results over multiple reference temperatures:\n");

        for (int j = 0; j < NT; ++j) {
            f_root[j]  *=  V / 2.0 * AVOGADRO * 1e6 / nstar_root[j];
            f2_root[j] *= (V / 2.0 * AVOGADRO * 1e6) * (V / 2.0 * AVOGADRO * 1e6) / nstar_root[j];
        
            double sigma = sqrt((f2_root[j] - f_root[j]*f_root[j]) / (nstar_root[j] - 1));
   
            printf("> T = %.2f: nstar = %.2f, svc = %.6f +- %.6f\n", TT[j], nstar_root[j], f_root[j], sigma);
        }
    }
}

void mixed_mc_nonrigid_svc() {
    auto model = build_model_from_npz("models/n2-ar-nonrigid-18-32-1.npz");
    double INFVAL = internal_pes_n2_ar(model, 1000.0, 0.0, 2.078); 

    std::vector<double> x(9);
    double RMAX_INT = 40.0; 
    
    int NPOINTS = 1000000;

    double svc = 0.0;
    double TT = 200.0;
   
    FILE* fd = fopen("nn.txt", "w");

    for (int k = 0; k < NPOINTS; ++k) {
        double R3 = uni_distr(gen) * RMAX_INT * RMAX_INT * RMAX_INT; 
        double R  = pow(R3, 1.0/3.0);
        
        if (R < 4.5) {
            svc += 1.0;
            continue;
        }

        double Phi   = uni_distr(gen) * 2.0 * M_PI;
        double Theta = acos(uni_distr(gen) * 2.0 - 1.0);

        double phi = uni_distr(gen) * 2.0 * M_PI;
        double theta = acos(uni_distr(gen) * 2.0 - 1.0);

        double l_N2 = sample_N2(TT);

        x[0] =  l_N2/2*cos(phi)*sin(theta); x[1] =  l_N2/2*sin(phi)*sin(theta); x[2] =  l_N2/2*cos(theta);
        x[3] = -l_N2/2*cos(phi)*sin(theta); x[4] = -l_N2/2*sin(phi)*sin(theta); x[5] = -l_N2/2*cos(theta);
        x[6] =  R*cos(Phi)*sin(Theta);      x[7] =  R*sin(Phi)*sin(Theta);      x[8] =  R*cos(Theta);

        double en    = model.forward(x) - INFVAL;
        double en_N2 = pot_N2(l_N2);

        {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            fprintf(fd, "%.6f %.6f\n", l_N2, en_N2);
        }

        svc += (1.0 - exp(-en * VkT / TT)); 
    }
    
    double V = 4.0 / 3.0 * M_PI * RMAX_INT * RMAX_INT * RMAX_INT * ALU3;
    svc *= V / 2.0 * AVOGADRO * 1e6 / NPOINTS;
   
    printf("  (npoints=%d) svc = %.6f\n", NPOINTS, svc);
    fclose(fd); 
}

void plain_mc_rigid_svc() {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double l_N2 = 2.078;

    auto model = build_model_from_npz("models/n2-ar-nonrigid-18-32-1.npz");
    double INFVAL = internal_pes_n2_ar(model, 1000.0, 0.0, l_N2); 

    std::vector<double> x(9);
    uint64_t NPOINTS = 500000000; 
    double RMAX_INT = 40.0; 
    
    double TT[] = {100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 
                          210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
                          310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 
                          410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0};
    int NT = ARR_LEN(TT);

    double f[NT]  = {0.0};
    double f2[NT] = {0.0};

    for (uint64_t k = 0; k < NPOINTS; ++k) {
        double R3 = uni_distr(gen) * RMAX_INT * RMAX_INT * RMAX_INT; 
        double R  = pow(R3, 1.0/3.0);
        
        if (R < 4.5) {
            for (int j = 0; j < NT; ++j) {
                f[j]  += 1.0;
                f2[j] += 1.0; 
            }
            continue;
        }

        double Phi   = uni_distr(gen) * 2.0 * M_PI;
        double Theta = acos(uni_distr(gen) * 2.0 - 1.0);

        double phi = uni_distr(gen) * 2.0 * M_PI;
        double theta = acos(uni_distr(gen) * 2.0 - 1.0);

        x[0] =  l_N2/2*cos(phi)*sin(theta); x[1] =  l_N2/2*sin(phi)*sin(theta); x[2] =  l_N2/2*cos(theta);
        x[3] = -l_N2/2*cos(phi)*sin(theta); x[4] = -l_N2/2*sin(phi)*sin(theta); x[5] = -l_N2/2*cos(theta);
        x[6] =  R*cos(Phi)*sin(Theta);      x[7] =  R*sin(Phi)*sin(Theta);      x[8] =  R*cos(Theta);

        double en = model.forward(x) - INFVAL;

        for (int j = 0; j < NT; ++j) { 
            double fi = 1.0 - exp(-en * VkT / TT[j]);
 
            f[j]  += fi;
            f2[j] += fi * fi; 
        }
    }

    for (int j = 0; j < NT; ++j) {
        double V = 4.0 / 3.0 * M_PI * RMAX_INT * RMAX_INT * RMAX_INT * ALU3;
        f[j]  *= V / 2.0 * AVOGADRO * 1e6 / NPOINTS;
        f2[j] *= (V / 2.0 * AVOGADRO * 1e6) * (V / 2.0 * AVOGADRO * 1e6) / NPOINTS;
    }

    double f_root[NT] = {0};
    double f2_root[NT] = {0};
    MPI_Reduce(f,  f_root,  NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(f2, f2_root, NT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        uint64_t npoints_total = NPOINTS * world_size; 
        printf("  (npoints=%lu)\n", npoints_total);

        double sigma;
        for (int j = 0; j < NT; ++j) {
            f_root[j]  /= world_size;
            f2_root[j] /= world_size;
            
            sigma = sqrt((f2_root[j] - f_root[j]*f_root[j]) / (npoints_total - 1));
            printf("> T = %.2f: svc = %.6f +- %.6f\n", TT[j], f_root[j], sigma);
        } 
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    //plain_mc_rigid_svc();
    mixed_mc_nonrigid_svc();
    //mixed_ensemble_nonrigid_svc();
    
    MPI_Finalize();
    
    //plain_mc_rigid_svc();
    //mixed_mc_nonrigid_svc();
    //mixed_ensemble_nonrigid_svc();
    
    // NOTE: RMAX=40.0
    //
    // T = 200 K
    // rigid:    (npoints=50,000,000) svc = -40.41
    //          (npoints=100,000,000) svc = -40.45
    //          (npoints=100,000,000) svc = -40.329859 +- 0.100738
    //          (npoints=2,000,000,000) svc = -40.399476 +- 0.022535
    //
    // nonrigid: (npoinst=100,000,000) svc = -40.37
  

    // T = 200 K; ensemble mcmc 
    // [mix_ensemble_nonrigid_svc; a = 1.0 for stretch move] 

    return 0;
}


