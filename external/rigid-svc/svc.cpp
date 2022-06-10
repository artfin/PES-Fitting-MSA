#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <thread>

#include <hep/mc-mpi.hpp>
#include "ai_pes_ch4_n2_opt1.hpp"

#include "load_model.hpp"

const double Boltzmann = 1.380649e-23;               // SI: J * K^(-1)
const double Hartree   = 4.3597447222071e-18;        // SI: J
const double HkT       = Hartree/Boltzmann;          // to use as:  -V[a.u.]*`HkT`/T
const double ALU       = 5.29177210903e-11;          // SI: m
const double AVOGADRO  = 6.022140857 * 1e23;
const double NN_BOND_LENGTH = 2.078; // a0

const double ALU3      = ALU * ALU * ALU; 

template <typename Container>
void print(std::string const& name, Container c) {
    std::cout << "[" << name << "]: ";
    for (auto el : c) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

// R PHI1 THETA1 PHI2 THETA2
const int DIM = 5;

template <typename T>
class Integrand
{
public:
    Integrand(T & pes) : pes(pes) {
        size_t ncartesian = 3 * 7;  
        cart.resize(ncartesian);
    } 

    void setTemperature(double t) {
        Temperature = t;
    }

    double internal_pes(double R, double PH1, double TH1, double PH2, double TH2);
    double operator()(hep::mc_point<double> const& x); 

private:
    T & pes;
    double Temperature;
    
    std::vector<double> cart;
    double RMAX = 30.0; 
};

template <typename T>
double Integrand<T>::internal_pes(double R, double PH1, double TH1, double PH2, double TH2)
// internal coordinates -> cartesian coordinates to call NNPIP.pes(cartesian coordinates)
{
    cart[0] =  1.193587416; cart[1]  =  1.193587416; cart[2]  = -1.193587416; // H1 
    cart[3] = -1.193587416; cart[4]  = -1.193587416; cart[5]  = -1.193587416; // H2
    cart[6] = -1.193587416; cart[7]  =  1.193587416; cart[8]  =  1.193587416; // H3 
    cart[9] =  1.193587416; cart[10] = -1.193587416; cart[11] =  1.193587416; // H4 

    cart[12] = R * std::sin(TH1) * std::cos(PH1) - NN_BOND_LENGTH/2.0 * std::cos(PH2) * std::sin(TH2);
    cart[13] = R * std::sin(TH1) * std::sin(PH1) - NN_BOND_LENGTH/2.0 * std::sin(PH2) * std::sin(TH2);
    cart[14] = R * std::cos(TH1)                 - NN_BOND_LENGTH/2.0 * std::cos(TH2);
    
    cart[15] = R * std::sin(TH1) * std::cos(PH1) + NN_BOND_LENGTH/2.0 * std::cos(PH2) * std::sin(TH2);
    cart[16] = R * std::sin(TH1) * std::sin(PH1) + NN_BOND_LENGTH/2.0 * std::sin(PH2) * std::sin(TH2);
    cart[17] = R * std::cos(TH1)                 + NN_BOND_LENGTH/2.0 * std::cos(TH2);

    cart[18] = 0.0; cart[19] = 0.0; cart[20] = 0.0;                           // C  

    return pes.pes(cart);
}

template <typename T>
double Integrand<T>::operator()(hep::mc_point<double> const& x) {
    double R   = std::tan(M_PI / 2.0 * x.point()[0]); 
    double PH1 = x.point()[1] * 2.0 * M_PI;
	double TH1 = x.point()[2] * M_PI;
    double PH2 = x.point()[3] * 2.0 * M_PI;
    double TH2 = x.point()[4] * M_PI;
    
	double RJ = M_PI / 2.0 * (1.0 + R * R);
    double PH1J = 2.0 * M_PI;
    double TH1J = M_PI;
    double PH2J = 2.0 * M_PI;
    double TH2J = M_PI;
	double DPJC = RJ * TH1J * PH1J * TH2J * PH2J;
    
    double in_braces, potval;
        
    if (R < 5.0) {
        in_braces = 1.0; 
    } else if (R < RMAX) {
        potval = internal_pes(R, PH1, TH1, PH2, TH2) / HTOCM; // Hartree
        //std::cout << "potval: " << potval << "\n";
        //potval = pes.pes(R, PH1, TH1, PH2, TH2) / HTOCM; // Hartree
        in_braces = 1.0 - std::exp(-potval * HkT / Temperature); 
    } else {
        in_braces = 0.0;
    }

    double d = DPJC * in_braces * R * R * std::sin(TH1) * std::sin(TH2);
    //std::cout << "d: " << d << "\n";

    return d; 
}

int main(int argc, char* argv[])
{
/*
    const size_t NATOMS = 7;
    const std::string torchscript_filename = "../../models/rigid/best-model/torchscript-model.pt"; 
    NNPIP nn_pes(NATOMS, torchscript_filename);
    Integrand<NNPIP> integrand(nn_pes);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    AI_PES_ch4_n2 symmpes;
    symmpes.init();
    //Integrand<AI_PES_ch4_n2> integrand(symmpes);

    std::cout << std::setprecision(10);
        
    const size_t NTIMES = 1;
    for (size_t k = 0; k < NTIMES; ++k) {
        double R   =  5.0;
        double PH1 = -2.6150449069645267;
        double TH1 =  0.4537317373406838;
        double PH2 = -1.9334253312268805;
        double TH2 =  0.8715875577518384; 
        double p = integrand.internal_pes(R, PH1, TH1, PH2, TH2);
        std::cout << "NN: " << p << "\n";

        double s = symmpes.pes(R, PH1, TH1, PH2, TH2);
        std::cout << "SYMM: " << s << "\n";
    }

    std::chrono::steady_clock::time_point end   = std::chrono::steady_clock::now();
    std::cout << "Expected answer [QC]: -0.3949590642\n";

    std::cout << "PES call time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / NTIMES << " [mus]" << std::endl;
    */ 
    std::cout << std::fixed << std::setprecision(16);
	MPI_Init(&argc, &argv); 
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const size_t NATOMS = 7;
    const std::string torchscript_filename = "../../models/rigid/best-model-2/torchscript-model.pt"; 
    NNPIP nn_pes(NATOMS, torchscript_filename);
    Integrand<NNPIP> integrand(nn_pes);

    //AI_PES_ch4_n2 symmpes;
    //symmpes.init();
    //Integrand<AI_PES_ch4_n2> integrand(symmpes);

	hep::mpi_vegas_callback<double>( hep::mpi_vegas_verbose_callback<double> );

    std::vector<double> temperatures = linspace(150.0, 500.0, 36);
    //std::vector<double> temperatures = linspace(300.0, 300.0, 1);
  
    std::ofstream ofs("nnpip.out");
    ofs << std::fixed << std::setprecision(10); 

    for (double T : temperatures) {
        integrand.setTemperature(T);
   
        auto results = hep::mpi_vegas(
            MPI_COMM_WORLD,
            hep::make_integrand<double>(integrand, DIM),
            std::vector<size_t>(20, 500000)
        );

        auto mean = hep::accumulate<hep::weighted_with_variance>(results.begin() + 2, results.end());
        double SVC = mean.value() / (8.0 * M_PI) * AVOGADRO * ALU3 * 1e6; // cm3/mol 
        
        if (rank == 0) {
            std::cout << T << " " << SVC << std::endl; 
            std::cout << "REFERENCE VALUE: " << std::endl;
            std::cout << "T=300.0000000000 SVC=-15.9569895499" << std::endl;
            ofs << T << " " << SVC << std::endl; 
        }
    }

    MPI_Finalize();

    return 0;
}
