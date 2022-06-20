#include <mpi.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cstdio>
#include <mutex>

#include "mtwist.h"
#include <torch/script.h>

#include "bind_ch4.h"

const double a0 = 2.0;

const double EVTOCM        = 8065.73;
const double DALTON        = 1.66054e-27;              // kg
const double LightSpeed_cm = 2.99792458e10;            // cm/s
const double EVTOJ         = 1.602176565e-19;

const double AVOGADRO  = 6.022140857 * 1e23;
const double BohrToAng = 0.529177210903;
const double Boltzmann = 1.380649e-23;             // SI: J * K^(-1)
const double Hartree   = 4.3597447222071e-18;      // SI: J
const double UGAS      = 8.31446261815324;         // SI: J * K^(-1) * mol^(-1)
const double HkT       = Hartree/Boltzmann;        // to use as:  -V[a.u.]*`HkT`/T
const double ALU       = 5.29177210903e-11;          // SI: m
const double HTOCM     = 2.1947463136320e5;        // 1 Hartree in cm-1
const double VkT       = HkT / HTOCM;              // to use as:  -V[cm-1]*`VkT`/T
const double ALU3      = ALU * ALU * ALU;

// maximum value of intermolecular distance accesible for a Markov chain
const double RMAX = 40.0;

const int DIM = 21;
const double GENSTEP[DIM] = {
    4e-2, 4e-2, 4e-2, // H1 
    4e-2, 4e-2, 4e-2, // H2
    4e-2, 4e-2, 4e-2, // H3 
    4e-2, 4e-2, 4e-2, // H4
    5e-3, 5e-3, 5e-3, // N1
    5e-3, 5e-3, 5e-3, // N2
    1e-3, 1e-3, 1e-3, // C
};

// DEFAULT UNITS: BOHR
static double INITIAL_GEOM[21] = {
    1.1935874160000000,  1.1935874160000000, -1.1935874160000000, // H 
   -1.1935874160000000, -1.1935874160000000, -1.1935874160000000, // H
   -1.1935874160000000,  1.1935874160000000,  1.1935874160000000, // H
    1.1935874160000000, -1.1935874160000000,  1.1935874160000000, // H
    3.4057873350028769, -3.6137325474466353, -0.0733398115913803, // N
    3.2585864240799634, -5.7013348353020371, -0.0079233056336203, // N
    0.0000000000000000,  0.0000000000000000,  0.0000000000000000, // C
};
 
const int BURNIN = 1000;

extern "C" {
    // monomial and polynomial evaluation for intermolecular potential
    void c_evmono(double* x, double* mono);
    void c_evpoly(double* mono, double* poly);
}

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

struct NNPIP
{
public:
    NNPIP();
    NNPIP(const size_t NATOMS, std::string const& pt_fname);

    ~NNPIP();

    void set_params(size_t NATOMS, std::string const& pt_fname);
    double pes(double x[DIM]);

private:
    size_t NATOMS;
    
    const size_t NMON  = 2892;
    const size_t NPOLY = 650;
    size_t NDIS;

    double *yij;
    double *mono;
    double *poly;

    torch::jit::script::Module model;
    at::Tensor t;
};

NNPIP::NNPIP() { }

void NNPIP::set_params(size_t NATOMS_, std::string const& pt_fname) {

    NATOMS = NATOMS_; 
    NDIS = NATOMS * (NATOMS - 1) / 2;

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

double NNPIP::pes(double x[DIM]) {
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

// Global neural network PES
NNPIP nn_pes;

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

void print_geometry(double x[DIM]) 
/*
 * printing xyz configuration in Angstrom to display in Chemcraft
 */
{
    std::cout << std::fixed << std::setprecision(6);
    std::string symbol = "HHHHNNC";
    for (int k = 0; k < 21; k += 3) {
        std::cout << symbol[k / 3] << " " << x[k] * BohrToAng << " " << x[k + 1] * BohrToAng << " " << x[k + 2] * BohrToAng << "\n";
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

double V1(double x[DIM])
/*
 * input:
 *     x[21] -- configuration of the complex
 * output:
 *     E(CH4) -- energy of the CH4 monomer
 *
 * H1: x[0],  x[1],  x[2]
 * H2: x[3],  x[4],  x[5]
 * H3: x[6],  x[7],  x[8]
 * H4: x[9],  x[10], x[11]
 * N1: x[12], x[13], x[14]
 * N2: x[15], x[16], x[17]
 * C : x[18], x[19], x[20]
 */ 
{
    // x_CH4: (C, H1, H2, H3, H4)
    static double x_CH4[15];
    x_CH4[0] =  x[18] * BohrToAng; x_CH4[1]  = x[19] * BohrToAng; x_CH4[2] =  x[20] * BohrToAng;
    x_CH4[3] =  x[0]  * BohrToAng; x_CH4[4]  = x[1]  * BohrToAng; x_CH4[5] =  x[2]  * BohrToAng;
    x_CH4[6] =  x[3]  * BohrToAng; x_CH4[7]  = x[4]  * BohrToAng; x_CH4[8] =  x[5]  * BohrToAng;
    x_CH4[9] =  x[6]  * BohrToAng; x_CH4[10] = x[7]  * BohrToAng; x_CH4[11] = x[8]  * BohrToAng;
    x_CH4[12] = x[9]  * BohrToAng; x_CH4[13] = x[10] * BohrToAng; x_CH4[14] = x[11] * BohrToAng;  
    
    return pot_CH4(x_CH4);
}

double V2(double x[DIM]) 
/*
 * input:
 *     x[21] -- configuration of the complex
 * output:
 *     E(N2) -- energy of the N2 monomer
 */
{
    double l_N2 = sqrt((x[12] - x[15]) * (x[12] - x[15]) + (x[13] - x[16]) * (x[13] - x[16]) + (x[14] - x[17]) * (x[14] - x[17])) * BohrToAng;
    return pot_N2(l_N2); 
}

double calc_interm_distance(double x[DIM])
{
    static double C[3], NN[3];
    
    C[0] = x[18];
    C[1] = x[19];
    C[2] = x[20];

    NN[0] = 0.5 * (x[12] + x[15]);
    NN[1] = 0.5 * (x[13] + x[16]);
    NN[2] = 0.5 * (x[14] + x[17]);

    return sqrt((NN[0] - C[0]) * (NN[0] - C[0]) + (NN[1] - C[1]) * (NN[1] - C[1]) + (NN[2] - C[2]) * (NN[2] - C[2]));
}


double density(double x[DIM], double T) 
{
    double V = V1(x) + V2(x);
    return exp(-V * VkT / T);
}

void make_step(double* src, double* dest) {
   for (int i = 0; i < DIM; ++i) {
       dest[i] = src[i] + generate_normal(GENSTEP[i]);
   } 
}

double burnin(double x[DIM], double T) {
    double c[DIM];
    double alpha, u;
    int acc = 0;

    for (int i = 0; i < BURNIN; ++i) {
        make_step(x, c);
        alpha = density(c, T) / density(x, T);
        u = mt_drand();

        if (u < alpha) {
            memcpy(x, c, sizeof(double) * DIM);
            ++acc; 
        }
    }    

    return (double) acc / BURNIN; 
}

struct History
{
    History(int npoints) 
    {
        interm_dist   = new double [npoints]; 
        CH4_energy    = new double [npoints]; 
        N2_energy     = new double [npoints]; 
        interm_energy = new double [npoints];

        counter = 0;
    }

    void record(double interm_dist_v, double CH4_enrg_v, double N2_enrg_v, double interm_enrg_v) {
        interm_dist[counter]   = interm_dist_v;
        CH4_energy[counter]    = CH4_enrg_v;
        N2_energy[counter]     = N2_enrg_v;
        interm_energy[counter] = interm_enrg_v;
        counter++;
    }

    void save(std::string const& fname, int npoints, double* v, std::ios_base::openmode mode)
    /*
     * thread safe save of the records 
     */ 
    {
        // mutex to protect file access
        static std::mutex mutex;

        std::lock_guard<std::mutex> lock(mutex);

        std::ofstream ofs;
        ofs.open(fname, mode);

        if (!ofs.is_open()) {
            throw std::runtime_error("unable to open file: " + fname);
        }

        ofs << std::fixed << std::setprecision(10);
        
        for (int k = 0; k < npoints; ++k) {
            ofs << v[k] << "\n";
        }
    }
    
    void save_tmp()
    {
        save("tmp_r.txt",      counter, interm_dist,   std::ios_base::out);
        save("tmp_CH4.txt",    counter, CH4_energy,    std::ios_base::out);
        save("tmp_N2.txt",     counter, N2_energy,     std::ios_base::out);
        save("tmp_interm.txt", counter, interm_energy, std::ios_base::out);
    }

    void save_all()
    {
        save("history_r.txt",      counter, interm_dist  , std::ios_base::app);  
        save("history_CH4.txt",    counter, CH4_energy   , std::ios_base::app);
        save("history_N2.txt",     counter, N2_energy    , std::ios_base::app);
        save("history_interm.txt", counter, interm_energy, std::ios_base::app);
    }

    ~History()
    {
        delete [] interm_dist; 
        delete [] CH4_energy; 
        delete [] N2_energy;
        delete [] interm_energy;
    }

    double * interm_dist; 
    double * CH4_energy;
    double * N2_energy;
    double * interm_energy;

    int counter; 
};


void init_worker()
{
    mt_goodseed();
    potinit();
}

void worker(int rank, int max_npoints, int thinning)
{
    
    const size_t NATOMS = 7;

    const std::string BASEDIR = "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/";
    const std::string torchscript_filename = BASEDIR + "models/nonrigid/nr-best-model/silu-ratio-clipped-torchscript.pt"; 
    //std::cout << "fname: " << torchscript_filename << "\n";
    
    nn_pes.set_params(NATOMS, torchscript_filename);

    double CH4_enrg, N2_enrg, interm_enrg, interm_dist;
    /*
    CH4_enrg    = V1(INITIAL_GEOM);
    N2_enrg     = V2(INITIAL_GEOM);
    interm_enrg = nn_pes.pes(INITIAL_GEOM);
    interm_dist = calc_interm_distance(INITIAL_GEOM);

    std::cout << "Initial energy:\n";
    std::cout << "   CH4: " << CH4_enrg << "\n";
    std::cout << "   N2:  " << N2_enrg << "\n";
    std::cout << "   interm_enrg: " << interm_enrg << "\n"; 
    std::cout << "   interm_dist: " << interm_dist << "\n";
    */
    
    double x[DIM];
    memcpy(x, INITIAL_GEOM, sizeof(double) * DIM); 
   
    const double T = 300.0; // K
    const double p0 = 101325.0; // 1atm -> Pa

    double acc_rate = burnin(x, T); 
    std::cout << std::setprecision(3); 
    std::cout << "[worker=" << rank << "] acceptance rate: " << acc_rate << std::endl;

    int acc = 0;
    double svc_worker = 0.0;

    double alpha, u;
    double c[DIM];

    History hh(max_npoints);

    for (int i = 0; acc < max_npoints; i++) {
        make_step(x, c);
        alpha = density(c, T) / density(x, T);
        u = mt_drand();

        if (u < alpha)  {
            memcpy(x, c, sizeof(double) * DIM);
        }

        interm_dist = calc_interm_distance(x);
        if (interm_dist > RMAX) {
            std::cout << "[worker=" << rank << "] maximum intermolecular distance reached." << std::endl;
            break;
        }

        if (i % thinning == 0) {
            interm_enrg = nn_pes.pes(x);
            svc_worker = svc_worker + (1.0 - exp(-interm_enrg * VkT / T)) * UGAS * T / p0 * 1.0e6; // cm3/mol 
            
            CH4_enrg    = V1(x);
            N2_enrg     = V2(x);
            interm_enrg = nn_pes.pes(x);
            interm_dist = calc_interm_distance(x);            
        
            hh.record(interm_dist, CH4_enrg, N2_enrg, interm_enrg);
            acc++;

            if (acc % 1000 == 0) {
                std::cout << "[worker=" << rank << "] collected: " << acc << std::endl;
                hh.save_tmp(); 
            }
        }
    }

    svc_worker = - svc_worker / acc;

    //std::cout << std::fixed << std::setprecision(3);
    //std::cout << "[worker=" << rank << "] sending SVC (npoints=" << acc << ") = " << svc_worker << std::endl;

    MPI_Send(&svc_worker, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&acc, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    hh.save_all();
}

void clear_history()
{
    std::cout << "[master] clearing history..." << std::endl;

    std::remove("history_r.txt");
    std::remove("history_CH4.txt");
    std::remove("history_N2.txt");
    std::remove("history_interm.txt");
    
    std::remove("tmp_r.txt");
    std::remove("tmp_CH4.txt");
    std::remove("tmp_N2.txt");
    std::remove("tmp_interm.txt");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int niterations = 3;
    int npoints_max = 100000;
    int thinning = 100;

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);

        std::cout << std::endl << std::endl;
        std::cout << "----------- START MCMC WORKER POOL --------------" << std::endl;
        std::cout << "   SIZE OF WORKER POOL: " << world_size << std::endl;

        clear_history();

        int src, npoints_worker, npoints_total;
        double svc_worker, svc_total;

        npoints_total = 0;
        svc_total = 0;

        MPI_Status status;

        for (int iter = 0; iter < niterations; ++iter) {
            for (int k = 0; k < world_size - 1; ++k) {
                MPI_Recv(&svc_worker, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                src = status.MPI_SOURCE;
                MPI_Recv(&npoints_worker, 1, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
                std::cout << "[master] received svc(npoints=" << npoints_worker << ") = " << svc_worker 
                          << "; worker=" << status.MPI_SOURCE << std::endl; 
            
                svc_total += npoints_worker * svc_worker;
                npoints_total += npoints_worker;
            }

            std::cout << "\n";
            std::cout << "[master] >> FINISHED ITERATION " << iter << "\n"; 
            std::cout << "[master] >> CURRENT SVC_TOTAL (npoints=" << npoints_total << ") = " 
                      << svc_total / npoints_total << std::endl << std::endl;
        }  
    
        std::cout << "\n";
        std::cout << "[master] >> FINAL SVC_TOTAL (npoints=" << npoints_total << ") = " << svc_total / npoints_total << std::endl; 

    } else {
        init_worker();

        for (int iter = 0; iter < niterations; ++iter) { 
            worker(rank, npoints_max, thinning);
        }
    }
    
    MPI_Finalize();

    return 0;
}


/*
double total_pot(double x[DIM], bool show=false) 
{
    //print_geometry(x);
    
    double intermolecular_energy = nn_pes.pes(x);

    // x_CH4: (C, H1, H2, H3, H4)
    static double x_CH4[15];
    x_CH4[0] =  x[18] * BohrToAng; x_CH4[1]  = x[19] * BohrToAng; x_CH4[2] =  x[20] * BohrToAng;
    x_CH4[3] =  x[0]  * BohrToAng; x_CH4[4]  = x[1]  * BohrToAng; x_CH4[5] =  x[2]  * BohrToAng;
    x_CH4[6] =  x[3]  * BohrToAng; x_CH4[7]  = x[4]  * BohrToAng; x_CH4[8] =  x[5]  * BohrToAng;
    x_CH4[9] =  x[6]  * BohrToAng; x_CH4[10] = x[7]  * BohrToAng; x_CH4[11] = x[8]  * BohrToAng;
    x_CH4[12] = x[9]  * BohrToAng; x_CH4[13] = x[10] * BohrToAng; x_CH4[14] = x[11] * BohrToAng;  

    double CH4_energy = pot_CH4(x_CH4);

    double l_N2 = sqrt((x[12] - x[15]) * (x[12] - x[15]) + (x[13] - x[16]) * (x[13] - x[16]) + (x[14] - x[17]) * (x[14] - x[17])) * BohrToAng;
    double N2_energy = pot_N2(l_N2);

    if (show) {
        std::cout << "# intermolecular energy: " << intermolecular_energy << "\n";
        std::cout << "# CH4 energy:            " << CH4_energy << "\n";
        std::cout << "# N2 energy:             " << N2_energy << "\n";
    }

    double total = intermolecular_energy + CH4_energy + N2_energy;

    return total; 
}
*/


