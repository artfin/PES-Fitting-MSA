#include <mpi.h>
#include <stdio.h>

#include <chrono>
#include <random>

extern "C" {
    void potinit();
    void poten_xy4(double *q, double* res);
}

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

    //double EMIN_N2 = 0.0;    // cm-1
    //double EMAX_N2 = 2000.0; // cm-1
                                   
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

void unit_vector(double x, double y, double z, double u[3]) {
    double l = sqrt(x*x + y*y + z*z); 
    u[0] = x / l; u[1] = y / l; u[2] = z / l; 
}

double dot_product(double u[3], double v[3]) {
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

double pot_CH4(double x[15])  
/*
 * x: 
 *    xC  = x[0]  yC  = x[1]  zC  = x[2]
 *    xH1 = x[3]  yH1 = x[4]  zH1 = x[5]
 *    xH2 = x[6]  yH2 = x[7]  zH2 = x[8]
 *    xH3 = x[9]  yH3 = x[10] zH3 = x[11]
 *    xH4 = x[12] yH4 = x[13] zH4 = x[14]
 * q: [0] r1 -- d(C-H1) 
 *    [1] r2 -- d(C-H2)
 *    [2] r3 -- d(C-H3)
 *    [3] r4 -- d(C-H4)
 *    [4] alpha12 -- ang(H1-C-H2)
 *    [5] alpha13 -- ang(H1-C-H3)
 *    [6] alpha14 -- ang(H1-C-H4)
 *    [7] alpha23 -- ang(H2-C-H3)
 *    [8] alpha24 -- ang(H2-C-H4)
 *    [9] alpha34 -- ang(H3-C-H4)
 */
{
    static double q[10];

    q[0] = sqrt(( x[3]-x[0])*( x[3]-x[0]) + ( x[4]-x[1])*( x[4]-x[1]) + ( x[5]-x[2])*( x[5]-x[2])) * BohrToAng;
    q[1] = sqrt(( x[6]-x[0])*( x[6]-x[0]) + ( x[7]-x[1])*( x[7]-x[1]) + ( x[8]-x[2])*( x[8]-x[2])) * BohrToAng;
    q[2] = sqrt(( x[9]-x[0])*( x[9]-x[0]) + (x[10]-x[1])*(x[10]-x[1]) + (x[11]-x[2])*(x[11]-x[2])) * BohrToAng;
    q[3] = sqrt((x[12]-x[0])*(x[12]-x[0]) + (x[13]-x[1])*(x[13]-x[1]) + (x[14]-x[2])*(x[14]-x[2])) * BohrToAng;

    int k = 4;
    static double u[3], v[3];

    for (int i = 1; i <= 3; ++i) {
        for (int j = i + 1; j <= 4; ++j) {
            unit_vector(x[3*i] - x[0], x[3*i + 1] - x[1], x[3*i + 2] - x[2], u);
            unit_vector(x[3*j] - x[0], x[3*j + 1] - x[1], x[3*j + 2] - x[2], v);
            q[k] = acos(dot_product(u, v));

            k++;
        } 
    } 
    
    // q: distances in Angstrom, angles in radian 
    static double V;
    poten_xy4(q, &V);

    double ang;
    for (int k = 4; k < 10; ++k) {
        ang = q[k] / M_PI * 180.0;
        if ((ang < 90.0) || (ang > 130.0)) {
             return 100000.0;
            //std::cout << "q[" << k << "]: " << q[k] << "; ang: " << ang << std::endl; 
        
            //std::cout << "V: " << V << "\n";
            //printf("C %.6f %.6f %.6f\n", x[0]  * BohrToAng,  x[1] * BohrToAng, x[2]  * BohrToAng);
            //printf("H %.6f %.6f %.6f\n", x[3]  * BohrToAng,  x[4] * BohrToAng, x[5]  * BohrToAng);
            //printf("H %.6f %.6f %.6f\n", x[6]  * BohrToAng,  x[7] * BohrToAng, x[8]  * BohrToAng);
            //printf("H %.6f %.6f %.6f\n", x[9]  * BohrToAng,  x[10]* BohrToAng, x[11] * BohrToAng);
            //printf("H %.6f %.6f %.6f\n", x[12] * BohrToAng,  x[13]* BohrToAng, x[14] * BohrToAng);
            //
            //exit(1); 
        } 
    }

    return V; 
}

double density_CH4(double x[15], double T) {
    double V = pot_CH4(x);
    return exp(-V * VkT / T);
}

#define NWALKERS 64 
#define DIM 15 
static double ensemble[NWALKERS][DIM];
   
#define BURNIN_LEN  100
#define MAX_NPOINTS 1000 
#define THINNING    10 

#define ARR_LEN(x)  (sizeof(x) / sizeof((x)[0]))

static std::random_device rd;
static std::mt19937 gen(rd()); 
static std::uniform_real_distribution<double> uni_distr(0.0, 1.0);
static std::uniform_int_distribution<int> int_distr(0, NWALKERS - 1);

Eigen::Matrix3d random_S() {
    static Eigen::Matrix3d m;

    double phi   = uni_distr(gen) * 2.0 * M_PI;
    double theta = acos(uni_distr(gen) * 2.0 - 1.0);
    double psi   = uni_distr(gen) * 2.0 * M_PI;

    m(0, 0) = cos(psi)*cos(phi) - cos(theta)*sin(phi)*sin(psi);
    m(0, 1) = cos(psi)*sin(phi) + cos(theta)*cos(phi)*sin(psi);
    m(0, 2) = sin(theta)*sin(psi);

    m(1, 0) = -sin(psi)*cos(phi) - cos(theta)*sin(phi)*cos(psi);
    m(1, 1) = -sin(psi)*sin(phi) + cos(theta)*cos(phi)*cos(psi);
    m(1, 2) =  sin(theta)*cos(psi);

    m(2, 0) = sin(theta)*sin(phi);
    m(2, 1) = -sin(theta)*cos(phi);
    m(2, 2) = cos(theta);

    return m;
}

void calc_center_of_mass(double x[DIM], double com[3]) {
    double mH = 1.00782503223;
    double mC = 12.00000000;

    com[0] = (mC * x[0] + mH * x[3] + mH * x[6] + mH * x[9]  + mH * x[12]) / (mC + mH * 4);
    com[1] = (mC * x[1] + mH * x[4] + mH * x[7] + mH * x[10] + mH * x[13]) / (mC + mH * 4);
    com[2] = (mC * x[2] + mH * x[5] + mH * x[8] + mH * x[11] + mH * x[14]) / (mC + mH * 4);
}

void generate_initial_geom(double x[DIM]) {
    static Eigen::Vector3d t;
    static Eigen::Matrix3d S;
    static Eigen::Vector3d H1, H2, H3, H4;
    H1 <<  1.193587416,  1.193587416, -1.193587416;
    H2 << -1.193587416, -1.193587416, -1.193587416; 
    H3 << -1.193587416,  1.193587416,  1.193587416;
    H4 <<  1.193587416, -1.193587416,  1.193587416;

    S = random_S();

    double CC = 0.01;
    x[0] = uni_distr(gen) * CC; 
    x[1] = uni_distr(gen) * CC; 
    x[2] = uni_distr(gen) * CC;

    t = S * H1;
    x[3] = t(0); x[4] = t(1); x[5] = t(2);

    t = S * H2;
    x[6] = t(0); x[7] = t(1); x[8] = t(2);

    t = S * H3;
    x[9] = t(0); x[10] = t(1); x[11] = t(2);

    t = S * H4;
    x[12] = t(0); x[13] = t(1); x[14] = t(2);

    static double com[3];
    calc_center_of_mass(x, com);

    x[0]  -= com[0]; x[1]  -= com[1]; x[2]  -= com[2];
    x[3]  -= com[0]; x[4]  -= com[1]; x[5]  -= com[2];
    x[6]  -= com[0]; x[7]  -= com[1]; x[8]  -= com[2];
    x[9]  -= com[0]; x[10] -= com[1]; x[11] -= com[2];
    x[12] -= com[0]; x[13] -= com[1]; x[14] -= com[2];
}

void timeit_ch4_potential() {
    size_t ncycles = 1000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    double x[15];

    for (int k = 0; k < ncycles; ++k) {
        generate_initial_geom(x);

        start = std::chrono::high_resolution_clock::now();
        double en = pot_CH4(x);
        end = std::chrono::high_resolution_clock::now();
        
        elapsed += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    printf("Elapsed: %.2f mcs; per cycle: %.3f\n", elapsed, elapsed / ncycles);
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


double make_ensemble_step(double T) {
    // candidate position

    double c[DIM];

    int acc = 0;
    
    for (int k = 0; k < NWALKERS; ++k) {
        int j = select_index_except_for(k);
        double z = propose_stretch_move(c, ensemble[k], ensemble[j]);
        
        double q = pow(z, DIM - 1) * density_CH4(c, T) / density_CH4(ensemble[k], T);
   
        double u = uni_distr(gen); 
        if (u < q) {
            memcpy(ensemble[k], c, sizeof(double) * DIM);
            acc++;
        }
    }

    /*
    for (int k = 0; k < NWALKERS; ++k) {
        printf("0: %.6f %.6f %.6f\n", ensemble[k][0],  ensemble[k][1],  ensemble[k][2] );
        printf("1: %.6f %.6f %.6f\n", ensemble[k][3],  ensemble[k][4],  ensemble[k][5] );
        printf("2: %.6f %.6f %.6f\n", ensemble[k][6],  ensemble[k][7],  ensemble[k][8] );
        printf("3: %.6f %.6f %.6f\n", ensemble[k][9],  ensemble[k][10], ensemble[k][11]);
        printf("4: %.6f %.6f %.6f\n", ensemble[k][12], ensemble[k][13], ensemble[k][14]);

        double com[3];
        calc_center_of_mass(ensemble[k], com);
        printf("com: %.6f %.6f %.6f\n\n", com[0], com[1], com[2]);
    }
    */

    return acc / (double) NWALKERS;
}

double burnin(double T) {
    double acc = 0.0;

    for (int k = 0; k < BURNIN_LEN; ++k) {
        acc += make_ensemble_step(T);
    }

    return acc / BURNIN_LEN;
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

    return model.forward(cart);
}

#include <math.h>
bool is_denormal(double f) {
    return isinf(f) || isnan(f); 
}

void ensemble_nonrigid_svc() {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const double deg = M_PI / 180.0;
    auto model = build_model_from_npz("models/ch4-n2-nonrigid.npz");
    const double INFVAL = internal_pes_ch4_n2(model, 1000.0, 47.912*deg, 56.167*deg, 0.0, 135.0*deg); 
    //printf("INFVAL: %.6f\n", INFVAL);  
    
    double RMAX_INT = 40.0;
    double V = 4.0 / 3.0 * M_PI * RMAX_INT * RMAX_INT * RMAX_INT * ALU3;
    
    double Tref[] = {510.0, 410.0};
    int NTR = ARR_LEN(Tref);

    double TT[] = {100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 
                          210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
                          310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0}; 
                          
    int NT = ARR_LEN(TT);

    double en, en_N2, en_CH4, sigma;
    std::vector<double> x(21);

    double f[NT]     = {0};
    double f2[NT]    = {0};
    double nstar[NT] = {0};

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
        for (uint64_t i = 0; acc < MAX_NPOINTS; i++) {
            make_ensemble_step(Tref[l]); 

            if (i % THINNING == 0) {
                for (int k = 0; k < NWALKERS; ++k) {
                    x[0]  = ensemble[k][3];  x[1]  = ensemble[k][4];  x[2]  = ensemble[k][5];  // H1
                    x[3]  = ensemble[k][6];  x[4]  = ensemble[k][7];  x[5]  = ensemble[k][8];  // H2
                    x[6]  = ensemble[k][9];  x[7]  = ensemble[k][10]; x[8]  = ensemble[k][11]; // H3
                    x[9]  = ensemble[k][12]; x[10] = ensemble[k][13]; x[11] = ensemble[k][14]; // H4
                    x[18] = ensemble[k][0];  x[19] = ensemble[k][1];  x[20] = ensemble[k][2];  // C

                    double l_N2  = sample_N2(Tref[l]);
                    double phi   = uni_distr(gen) * 2.0 * M_PI;
                    double theta = acos(uni_distr(gen) * 2.0 - 1.0);
                    x[12] =  l_N2/2*cos(phi)*sin(theta); x[13] =  l_N2/2*sin(phi)*sin(theta); x[14] =  l_N2/2*cos(theta); // N1
                    x[15] = -l_N2/2*cos(phi)*sin(theta); x[16] = -l_N2/2*sin(phi)*sin(theta); x[17] = -l_N2/2*cos(theta); // N2

                    double R3    = uni_distr(gen) * RMAX_INT * RMAX_INT * RMAX_INT;
                    double R     = pow(R3, 1.0/3.0);
                    double Phi   = uni_distr(gen) * 2.0 * M_PI;
                    double Theta = acos(uni_distr(gen) * 2.0 - 1.0);

                    double trsl[3];
                    trsl[0] = R*cos(Phi)*sin(Theta); 
                    trsl[1] = R*sin(Phi)*sin(Theta);
                    trsl[2] = R*cos(Theta);

                    x[12] += trsl[0]; x[13] += trsl[1]; x[14] += trsl[2];
                    x[15] += trsl[0]; x[16] += trsl[1]; x[17] += trsl[2]; 

                    en_N2  = pot_N2(l_N2);
                    en_CH4 = pot_CH4(ensemble[k]);

                    if (is_denormal(en_N2)) {
                        fprintf(stderr, "l_N2: %.6f\n", l_N2);
                        assert(!is_denormal(en_N2) && "en_N2 is denormal");
                    }
                    if (is_denormal(en_CH4)) {
                        fprintf(stderr, "C1 %.6f %.6f %.6f\n", ensemble[k][0],  ensemble[k][1],  ensemble[k][2] );
                        fprintf(stderr, "H1 %.6f %.6f %.6f\n", ensemble[k][3],  ensemble[k][4],  ensemble[k][5] );
                        fprintf(stderr, "H2 %.6f %.6f %.6f\n", ensemble[k][6],  ensemble[k][7],  ensemble[k][8] );
                        fprintf(stderr, "H3 %.6f %.6f %.6f\n", ensemble[k][9],  ensemble[k][10], ensemble[k][11]);
                        fprintf(stderr, "H4 %.6f %.6f %.6f\n", ensemble[k][12], ensemble[k][13], ensemble[k][14]);
                        assert(!is_denormal(en_CH4) && "en_CH4 is denormal");
                    }

                    double fi, w;   
                    for (int j = 0; j < NT; ++j) {
                        if (TT[j] < Tref[l] + 0.1) {
                            if (R < 4.5) {
                                fi = 1.0;
                            } else {
                                en = model.forward(x) - INFVAL;
                                fi = (1.0 - exp(-en * VkT / TT[j]));

                                if (is_denormal(fi)) {
                                    fprintf(stderr, "H1 %.6f %.6f %.6f\n", x[0],  x[1],  x[2] );
                                    fprintf(stderr, "H2 %.6f %.6f %.6f\n", x[3],  x[4],  x[5] );
                                    fprintf(stderr, "H3 %.6f %.6f %.6f\n", x[6],  x[7],  x[8] );
                                    fprintf(stderr, "H4 %.6f %.6f %.6f\n", x[9],  x[10], x[11]);
                                    fprintf(stderr, "N1 %.6f %.6f %.6f\n", x[12], x[13], x[14]);
                                    fprintf(stderr, "N2 %.6f %.6f %.6f\n", x[15], x[16], x[17]);
                                    fprintf(stderr, "C1 %.6f %.6f %.6f\n", x[18], x[19], x[20]);
                                    assert(!is_denormal(fi) && "fi is denormal");
                                }
                            }

                            w = exp(-(en_N2 + en_CH4)*VkT / TT[j]) / exp(-(en_N2 + en_CH4) * VkT / Tref[l]);

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
            printf("Tref = %.2f done. [0] Collected %lu points.\n", Tref[l], acc);
        }

        double f_t[NT]     = {0};
        double f2_t[NT]    = {0};
        double nstar_t[NT] = {0};

        for (int j = 0; j < NT; ++j) {
            f_t[j]     = f[j] *  V / 2.0 * AVOGADRO * 1e6;
            f2_t[j]    = f2[j] * (V / 2.0 * AVOGADRO * 1e6) * (V / 2.0 * AVOGADRO * 1e6);
            nstar_t[j] = nstar[j];
        }

        double f_t_root[NT] = {0};
        double f2_t_root[NT] = {0};
        double nstar_t_root[NT] = {0};

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
    
    double f_root[NT]     = {0};
    double f2_root[NT]    = {0};
    double nstar_root[NT] = {0};

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

void safe_potinit_ch4() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    potinit();
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    safe_potinit_ch4();
    ensemble_nonrigid_svc(); 
    
    MPI_Finalize();

    return 0;
}
