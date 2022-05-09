#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mtwist.h"

const double deg = M_PI / 180.0;

const double BohrToAng     = 0.529177210903;
const double Boltzmann     = 1.380649e-23;             // SI: J * K^(-1)
const double Hartree       = 4.3597447222071e-18;      // SI: J
const double HkT           = Hartree/Boltzmann;        // to use as:  -V[a.u.]*`HkT`/T
const double HTOCM         = 2.1947463136320e5;        // 1 Hartree in cm-1
const double VkT           = HkT / HTOCM;              // to use as:  -V[cm-1]*`VkT`/T

const double LightSpeed_cm = 2.99792458e10;            // cm/s
const double EVTOCM        = 8065.73;
const double DALTON        = 1.66054e-27;              // kg
const double EVTOJ         = 1.602176565e-19;
const double ANGTOBOHR     = 1.0 / 0.529177249;

#define SHOW_MCMC_STEPS_COUNTER
//#undef SHOW_MCMC_STEPS_COUNTER

const int DIM = 15;
const double GENSTEP[DIM] = {5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3};

const double Teff_CH4 = 50.0; // K 
const double Teff_N2  = 1000.0; // K

const double EMIN_CH4 = 500.0; // cm-1
const double EMAX_CH4 = 1000.0; // cm-1

const double EMIN_N2 = 500.0;
const double EMAX_N2 = 1000.0; // cm-1

const int BURNIN = 10000;
const int THINNING = 500;

extern "C" {
    void potinit();
    void poten_xy4(double *q, double* res);
}

static double INITIAL_CH4_GEOM[15] = {
     0.0000000000,  0.0000000000,  0.0000000000, 
     1.0860100000,  0.0000000000,  0.0000000000,
    -0.3620033333,  0.0000000000, -1.0239000473,
    -0.3620033333, -0.8867234520,  0.5119500235,
    -0.3620033333,  0.8867234517,  0.5119500240,
};

double generate_normal(double sigma) 
/*
 * Generate normally distributed variable using Box-Muller method
 * Uniformly distributed variables are generated using Mersenne Twister
 *
 * double mt_drand(void)
 *   Return a pseudorandom double in [0,1) with 32 bits of randomness
 */
{
    double U = mt_drand();
    double V = mt_drand();
    return sigma * sqrt(-2 * log(U)) * cos(2.0 * M_PI * V);
}

void unit_vector(double x, double y, double z, double u[3]) {
    double l = sqrt(x*x + y*y + z*z); 
    u[0] = x / l; u[1] = y / l; u[2] = z / l; 
}

double dot_product(double u[3], double v[3]) {
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

void xyz_to_internal(double x[15], double q[10]) 
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
    q[0] = sqrt((x[3]-x[0])*(x[3]-x[0]) + (x[4]-x[1])*(x[4]-x[1]) + (x[5]-x[2])*(x[5]-x[2]));
    q[1] = sqrt((x[6]-x[0])*(x[6]-x[0]) + (x[7]-x[1])*(x[7]-x[1]) + (x[8]-x[2])*(x[8]-x[2]));
    q[2] = sqrt((x[9]-x[0])*(x[9]-x[0]) + (x[10]-x[1])*(x[10]-x[1]) + (x[11]-x[2])*(x[11]-x[2]));
    q[3] = sqrt((x[12]-x[0])*(x[12]-x[0]) + (x[13]-x[1])*(x[13]-x[1]) + (x[14]-x[2])*(x[14]-x[2]));

    int k = 4;
    static double u[3], v[3];
   
    for (int i = 1; i <= 3; ++i) {
        for (int j = i + 1; j <= 4; ++j) {
            unit_vector(x[3*i], x[3*i + 1], x[3*i + 2], u);
            unit_vector(x[3*j], x[3*j + 1], x[3*j + 2], v);
            q[k] = acos(dot_product(u, v));
            k++;
        } 
    } 
}

void center_CH4(double x[15]) {
    for (int i = 1; i < 5; i++) {
        x[3*i] -= x[0]; x[3*i + 1] -= x[1]; x[3*i + 2] -= x[2];
    }
    x[0] = x[1] = x[2] = 0.0;
}

double pot_CH4(double x[15]) {
    static double q[10];
    xyz_to_internal(x, q);

    //for (int k = 0; k < 10; ++k) {
    //    printf("q[%d] = %.10lf\n", k, q[k]);
    //}

    double V;
    poten_xy4(q, &V);

    //printf("[pot_CH4] V: %.5lf\n", V);

    return V;
}

double density_CH4(double x[15]) {
    return exp(- pot_CH4(x) * VkT / Teff_CH4);
}

void make_step(double* src, double* dest) {
   for (int i = 0; i < DIM; ++i) {
       dest[i] = src[i] + generate_normal(GENSTEP[i]);
   } 
}

double pot_N2(double r)
/*
 * returns N2 potential [cm-1] approximated as a Morse curve
 * the parameters are derived from experiment
 * accepts the distance in A
 */ 
{
    // https://doi.org/10.1098/rspa.1956.0135 
    const double De    = 9.91; // eV
    const double omega = 2358.57; // cm-1
    const double nu    = 2.0 * M_PI * LightSpeed_cm * omega; // 1/s
    const double mu    = 14.003074004460 / 2.0 * DALTON;

    const double a  = sqrt(mu / (2.0 * De * EVTOJ)) * nu * 1e-10; // A
    const double re = 1.09768; // A

    return (De * EVTOCM) * (1 - exp(-a * (r - re))) * (1 - exp(-a * (r - re))); 
} 

double density_N2(double r) {
    return exp(-pot_N2(r) * VkT / Teff_N2);
}

double sample_N2() 
/*
 * sample density function of N2 using rejection approach
 */
{
    const double x1 = 0.9, x2 = 1.3; // A
    const double y1 = 0.0, y2 = 1.0; 

    double x, y, E;
    while (true) {
        x = mt_drand() * (x2 - x1) + x1;
        y = mt_drand() * (y2 - y1) + y1;

        if (y < density_N2(x)) {
            E = pot_N2(x);
            if ((E > EMIN_N2) && (E < EMAX_N2)) return x;
        }
    }
}

void burnin(double x[15]) {
    double c[15];
    double alpha, u;
    int acc = 0;

    for (int i = 0; i < BURNIN; ++i) {
        make_step(x, c);
        alpha = density_CH4(c) / density_CH4(x);
        u = mt_drand();

        if (u < alpha) {
            memcpy(x, c, sizeof(double) * DIM);
            ++acc; 
        }
    }     
    printf("Percentage of accepted points: %.3lf\n", (double) acc/BURNIN * 100.0);
}

void print_CH4_coords(FILE * fd, double x_CH4[15]) {
    fprintf(fd, "6 %.10lf %.10lf %.10lf\n", x_CH4[0] , x_CH4[1] , x_CH4[2] );
    fprintf(fd, "1 %.10lf %.10lf %.10lf\n", x_CH4[3] , x_CH4[4] , x_CH4[5] );
    fprintf(fd, "1 %.10lf %.10lf %.10lf\n", x_CH4[6] , x_CH4[7] , x_CH4[8] );
    fprintf(fd, "1 %.10lf %.10lf %.10lf\n", x_CH4[9] , x_CH4[10], x_CH4[11]);
    fprintf(fd, "1 %.10lf %.10lf %.10lf\n", x_CH4[12], x_CH4[13], x_CH4[14]);
}

bool check_geometry_CH4(double x[15])
/* 
 * x: 
 *    xC  = x[0]  yC  = x[1]  zC  = x[2]
 *    xH1 = x[3]  yH1 = x[4]  zH1 = x[5]
 *    xH2 = x[6]  yH2 = x[7]  zH2 = x[8]
 *    xH3 = x[9]  yH3 = x[10] zH3 = x[11]
 *    xH4 = x[12] yH4 = x[13] zH4 = x[14]
 */
{
    const double HH_LOWLIM = 1.0;
    const double HH_UPLIM  = 2.2;
        
    double H12, H13, H14, H23, H24, H34;

    H12 = sqrt((x[3] - x[6]) * (x[3] - x[6]) + (x[4] - x[7]) * (x[4] - x[7]) + (x[5] - x[8]) * (x[5] - x[8])); 
    H13 = sqrt((x[3] - x[9]) * (x[3] - x[9]) + (x[4] - x[10]) * (x[4] - x[10]) + (x[5] - x[11]) * (x[5] - x[11])); 
    H14 = sqrt((x[3] - x[12]) * (x[3] - x[12]) + (x[4] - x[13]) * (x[4] - x[13]) + (x[5] - x[14]) * (x[5] - x[14])); 
    H23 = sqrt((x[6] - x[9]) * (x[6] - x[9]) + (x[7] - x[10]) * (x[7] - x[10]) + (x[8] - x[11]) * (x[8] - x[11])); 
    H24 = sqrt((x[6] - x[12]) * (x[6] - x[12]) + (x[7] - x[13]) * (x[7] - x[13]) + (x[8] - x[14]) * (x[8] - x[14])); 
    H34 = sqrt((x[9] - x[12]) * (x[9] - x[12]) + (x[10] - x[13]) * (x[10] - x[13]) + (x[11] - x[14]) * (x[11] - x[14])); 

    bool status = true;
    if ((H12 > HH_UPLIM) || (H12 < HH_LOWLIM)) status = false;
    if ((H13 > HH_UPLIM) || (H13 < HH_LOWLIM)) status = false;
    if ((H14 > HH_UPLIM) || (H14 < HH_LOWLIM)) status = false;
    if ((H23 > HH_UPLIM) || (H23 < HH_LOWLIM)) status = false;
    if ((H24 > HH_UPLIM) || (H24 < HH_LOWLIM)) status = false;
    if ((H34 > HH_UPLIM) || (H34 < HH_LOWLIM)) status = false;

    return status;
}

void sample_CH4(double x[15], int thinning=THINNING) {

    double c[15];
    double alpha, u;

    int acc = 0;

    for (int counter = 0; ; counter++) {
        make_step(x, c);
        alpha = density_CH4(c) / density_CH4(x);
        u = mt_drand();

        if (u < alpha) {
            memcpy(x, c, sizeof(double) * DIM);
            acc++;
        }
        if (counter % thinning == 0 && counter != 0) {
            bool st = check_geometry_CH4(x);

            if (!st) {
                print_CH4_coords(stdout, x);
                printf("geometry rejected\n");
                memcpy(x, INITIAL_CH4_GEOM, sizeof(double) * DIM);
                burnin(x); 
            }
#ifdef SHOW_MCMC_STEPS_COUNTER
            printf("Steps made: %d\n", acc);
#endif
            break; 
        } 
    }
}

void sample_inter(double inter[3]) {

    const double rmin = 3.0, rmax = 6.0;
    inter[0] = mt_drand() * (rmax - rmin) + rmin; // R
    inter[1] = mt_drand() * 2.0 * M_PI;           // Phi
    inter[2] = acos(mt_drand() * 2.0 - 1.0);      // Theta
}


int main()
{
    mt_goodseed();
    potinit();

    double x_CH4[15];
    memcpy(x_CH4, INITIAL_CH4_GEOM, sizeof(double) * DIM);

    printf("INITIAL ENERGY: %lf\n", pot_CH4(x_CH4));

    burnin(x_CH4);
    center_CH4(x_CH4);

    for (int n = 25000; n < 30000; ) {
        sample_CH4(x_CH4);
        center_CH4(x_CH4);
        double V_CH4 = pot_CH4(x_CH4);
        
        double l_N2 = sample_N2();
        double V_N2 = pot_N2(l_N2);

        printf("[CH4] V: %.3lf cm-1\n", V_CH4);
        printf("[N2]  V: %.3lf cm-1\n", V_N2);

        if ((V_CH4 > EMAX_CH4) || (V_CH4 < EMIN_CH4)) continue;
        
        double phi_N2 = mt_drand() * 2.0 * M_PI;
        double theta_N2 = acos(mt_drand() * 2.0 - 1.0);
        
        double inter[3];
        sample_inter(inter);

        double x_N2[6];
        x_N2[0] = -l_N2/2 * cos(phi_N2) * sin(theta_N2);
        x_N2[1] = -l_N2/2 * sin(phi_N2) * sin(theta_N2);
        x_N2[2] = -l_N2/2 * cos(theta_N2);
        x_N2[3] = l_N2/2 * cos(phi_N2) * sin(theta_N2);
        x_N2[4] = l_N2/2 * sin(phi_N2) * sin(theta_N2);
        x_N2[5] = l_N2/2 * cos(theta_N2);

        x_N2[0] += inter[0] * cos(inter[1]) * sin(inter[2]);
        x_N2[1] += inter[0] * sin(inter[1]) * sin(inter[2]);
        x_N2[2] += inter[0] * cos(inter[2]);
        x_N2[3] += inter[0] * cos(inter[1]) * sin(inter[2]);
        x_N2[4] += inter[0] * sin(inter[1]) * sin(inter[2]);
        x_N2[5] += inter[0] * cos(inter[2]);

        char xyz_fname[256];
        snprintf(xyz_fname, sizeof(xyz_fname), "INP-25000-30000/%03d.xyz", n);
        printf("Writing xyz configuration to %s\n", xyz_fname); 

        FILE * fd_xyz = fopen(xyz_fname, "w");
        fprintf(fd_xyz, "# length units: A\n");
        fprintf(fd_xyz, "# V [CH4]: %.10lf\n", V_CH4);
        fprintf(fd_xyz, "# V [N2] : %.10lf\n", V_N2);
        fprintf(fd_xyz, "6 %.10lf %.10lf %.10lf\n", x_CH4[0] , x_CH4[1] , x_CH4[2] );
        fprintf(fd_xyz, "1 %.10lf %.10lf %.10lf\n", x_CH4[3] , x_CH4[4] , x_CH4[5] );
        fprintf(fd_xyz, "1 %.10lf %.10lf %.10lf\n", x_CH4[6] , x_CH4[7] , x_CH4[8] );
        fprintf(fd_xyz, "1 %.10lf %.10lf %.10lf\n", x_CH4[9] , x_CH4[10], x_CH4[11]);
        fprintf(fd_xyz, "1 %.10lf %.10lf %.10lf\n", x_CH4[12], x_CH4[13], x_CH4[14]);
        fprintf(fd_xyz, "7 %.10lf %.10lf %.10lf\n", x_N2[0]  , x_N2[1]  , x_N2[2]  );
        fprintf(fd_xyz, "7 %.10lf %.10lf %.10lf\n", x_N2[3]  , x_N2[4]  , x_N2[5]  );
        fclose(fd_xyz);

        char molpro_fname[256];
        snprintf(molpro_fname, sizeof(molpro_fname), "INP-25000-30000/%03d.inp", n);
        printf("Writing MOLPRO input to %s\n", molpro_fname);

        FILE * fd_molpro = fopen(molpro_fname, "w");

        char C_coords[64], H1_coords[64], H2_coords[64], H3_coords[64], H4_coords[64], N1_coords[64], N2_coords[64];
        snprintf(C_coords,  sizeof(C_coords),  "1, C1,,%.10lf, %.10lf, %.10lf\n", x_CH4[0] * ANGTOBOHR, x_CH4[1]  * ANGTOBOHR, x_CH4[2]  * ANGTOBOHR); 
        snprintf(H1_coords, sizeof(H1_coords), "2, H1,,%.10lf, %.10lf, %.10lf\n", x_CH4[3] * ANGTOBOHR, x_CH4[4]  * ANGTOBOHR, x_CH4[5]  * ANGTOBOHR); 
        snprintf(H2_coords, sizeof(H2_coords), "3, H2,,%.10lf, %.10lf, %.10lf\n", x_CH4[6] * ANGTOBOHR, x_CH4[7]  * ANGTOBOHR, x_CH4[8]  * ANGTOBOHR); 
        snprintf(H3_coords, sizeof(H3_coords), "4, H3,,%.10lf, %.10lf, %.10lf\n", x_CH4[9] * ANGTOBOHR, x_CH4[10] * ANGTOBOHR, x_CH4[11] * ANGTOBOHR); 
        snprintf(H4_coords, sizeof(H4_coords), "5, H4,,%.10lf, %.10lf, %.10lf\n", x_CH4[12]* ANGTOBOHR, x_CH4[13] * ANGTOBOHR, x_CH4[14] * ANGTOBOHR); 
        snprintf(N1_coords, sizeof(N1_coords), "6, N1,,%.10lf, %.10lf, %.10lf\n", x_N2[0]  * ANGTOBOHR, x_N2[1]   * ANGTOBOHR, x_N2[2]   * ANGTOBOHR); 
        snprintf(N2_coords, sizeof(N2_coords), "7, N2,,%.10lf, %.10lf, %.10lf\n", x_N2[3]  * ANGTOBOHR, x_N2[4]   * ANGTOBOHR, x_N2[5]   * ANGTOBOHR); 
        
        char buf[2048];
        snprintf(buf, sizeof(buf), "%s%s%s%s%s%s%s%s%s", 
                 "***,CH4-N2\n"
                 "memory,160,m;\n"
                 "gthresh,energy=1.d-11;\n"
                 "basis=avtz\n"
                 "symmetry,nosym\n"
                 "GEOMETRY={\n",
                 C_coords,
                 H1_coords,
                 H2_coords,
                 H3_coords,
                 H4_coords,
                 N1_coords,
                 N2_coords,
                 "}\n"
                 "{hf,maxit=300}\n"
                 "{ccsd(t)-f12,SINGLES=0,df_basis=avtz/mp2fit,ri_basis=avtz/optri,\\\n"
                 "df_basis_exch=avtz/mp2fit,SCALE_TRIP=1,thrvar=1.e-14,gem_beta=1.3,\\\n"
                 "maxit=200}\n"
                 "eABa=energy(1)\n"
                 "eABb=energy(2)\n"
                 "\n"
                 "dummy, N1,N2\n"
                 "{hf,maxit=300}\n"
                 "{ccsd(t)-f12,SINGLES=0,df_basis=avtz/mp2fit,ri_basis=avtz/optri,\\\n"
                 "df_basis_exch=avtz/mp2fit,SCALE_TRIP=1,thrvar=1.e-14,gem_beta=1.3,\\\n"
                 "maxit=200}\n"
                 "eAa=energy(1)\n"
                 "eAb=energy(2)\n"
                 "\n"
                 "dummy, C1,H1,H2,H3,H4\n"
                 "{hf,maxit=300}\n"
                 "{ccsd(t)-f12,SINGLES=0,df_basis=avtz/mp2fit,ri_basis=avtz/optri,\\\n"
                 "df_basis_exch=avtz/mp2fit,SCALE_TRIP=1,thrvar=1.e-14,gem_beta=1.3,\\\n"
                 "maxit=200}\n"
                 "eBa=energy(1)\n"
                 "eBb=energy(2)\n"
                 "\n"
                 "eCPa=(eABa-eAa-eBa)*tocm\n"
                 "eCPb=(eABb-eAb-eBb)*tocm\n"
                 "\n"
                 "{table,eCPa,eCPb;digits,10,10}\n"
        );

        fprintf(fd_molpro, "%s", buf);

        fclose(fd_molpro);

        printf("\n\n");
        n++;
    }

    return 0;
}


/*
void to_xyz(double* x, double* xyz) {

    double r1, r2, r3, r4;
    double alpha12, alpha13, alpha14, alpha23, alpha24, alpha34;

    r1 = x[0]; r2 = x[1]; r3 = x[2]; r4 = x[3];

    alpha12 = x[4]; 
    alpha13 = x[5]; 
    alpha14 = x[6]; 
    alpha23 = x[7]; 
    alpha24 = x[8]; 
    alpha34 = x[9]; 

    xyz[0] = 0.0; xyz[1] = 0.0; xyz[2] = 0.0; // C

    // H1
    xyz[3] = 0.0; xyz[4] = 0.0; xyz[5] = r1;  

    // H2
    xyz[6] = r2 * sin(alpha12); xyz[7] = 0.0; xyz[8] = r2 * cos(alpha12); 
   
    // H3
    xyz[9] = - r3 * (cos(alpha13) * cos(alpha12) - cos(alpha23)) / sin(alpha12); 
    xyz[10] = - r3 * (cos(alpha12) * cos(alpha24) * cos(alpha13) + cos(alpha12) * cos(alpha23) * cos(alpha14) + cos(alpha34) * sin(alpha12) * sin(alpha12) - cos(alpha24) * cos(alpha23) - cos(alpha14) * cos(alpha13)) / sqrt(2.0 * cos(alpha12) * cos(alpha14) * cos(alpha24) - cos(alpha24)*cos(alpha24) - cos(alpha14) *cos(alpha14)+ sin(alpha12)*sin(alpha12)) / sin(alpha12);    
    xyz[11] = r3 * cos(alpha13); 

    // H4
    xyz[12] = - r4 * (cos(alpha12) * cos(alpha14) - cos(alpha24)) / sin(alpha12);
    xyz[13] = - r4 * sqrt(2.0 * cos(alpha12) * cos(alpha14) * cos(alpha24) - cos(alpha14)*cos(alpha14) - cos(alpha24) * cos(alpha24) + sin(alpha12) * sin(alpha12)) / sin(alpha12);
    xyz[14] = r4 * cos(alpha14);
}
*/


