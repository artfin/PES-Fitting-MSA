#include <cassert>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <thread>

#include <hep/mc-mpi.hpp>

#include "load_model.hpp"
#include "ai_pes_ch4_n2_opt1.hpp"

const double Boltzmann = 1.380649e-23;               // SI: J * K^(-1)
const double Hartree   = 4.3597447222071e-18;        // SI: J
const double HkT       = Hartree/Boltzmann;          // to use as:  -V[a.u.]*`HkT`/T
const double ALU       = 5.29177210903e-11;          // SI: m
const double AVOGADRO  = 6.022140857 * 1e23;

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

class Integrand
{
public:
    Integrand(NNPIP & pes) : pes(pes) {
        size_t ncartesian = 3 * pes.NATOMS;  
        cart.resize(ncartesian);

        cart[0] =  1.193587416; cart[1]  =  1.193587416; cart[2]  = -1.193587416; // H1 
        cart[3] = -1.193587416; cart[4]  = -1.193587416; cart[5]  = -1.193587416; // H2
        cart[6] = -1.193587416; cart[7]  =  1.193587416; cart[8]  =  1.193587416; // H3 
        cart[9] =  1.193587416; cart[10] = -1.193587416; cart[11] =  1.193587416; // H4 
    
        cart[18] = 0.0; cart[19] = 0.0; cart[20] = 0.0;                           // C  
    } 

    void setTemperature(double t) {
        Temperature = t;
    }

    double internal_pes(double R, double PH1, double TH1, double PH2, double TH2);
    double operator()(hep::mc_point<double> const& x); 

private:
    NNPIP & pes;
    double Temperature;

    std::vector<double> cart;

    double RMIN = 5.0;
    double RMAX = 30.0; 
};

double Integrand::internal_pes(double R, double PH1, double TH1, double PH2, double TH2)
// internal coordinates -> cartesian coordinates
// call NNPIP.pes(cartesian coordinates)
{
    cart[12] = R * std::sin(TH1) * std::cos(PH1) - NN_BOND_LENGTH * std::cos(PH2) * std::sin(TH2);
    cart[13] = R * std::sin(TH1) * std::sin(PH1) - NN_BOND_LENGTH * std::sin(PH2) * std::sin(TH2);
    cart[14] = R * std::cos(TH1)                 - NN_BOND_LENGTH * std::cos(TH2);
    
    cart[15] = R * std::sin(TH1) * std::cos(PH1) + NN_BOND_LENGTH * std::cos(PH2) * std::sin(TH2);
    cart[16] = R * std::sin(TH1) * std::sin(PH1) + NN_BOND_LENGTH * std::sin(PH2) * std::sin(TH2);
    cart[17] = R * std::cos(TH1)                 + NN_BOND_LENGTH * std::cos(TH2);

    return pes.pes(cart);
}

double Integrand::operator()(hep::mc_point<double> const& x) {
	double R   = x.point()[0] * (RMAX - RMIN) + RMIN; 
	double TH1 = x.point()[1] * M_PI;
    double PH1 = x.point()[2] * 2.0 * M_PI;
    double TH2 = x.point()[3] * M_PI;
    double PH2 = x.point()[4] * 2.0 * M_PI;
    
    double RJ   = (RMAX - RMIN); 
	double TH1J = M_PI;
    double PH1J = 2.0 * M_PI;
    double TH2J = M_PI;
    double PH2J = 2.0 * M_PI;

	double DPJC = RJ * TH1J * PH1J * TH2J * PH2J;
    
    double in_braces, potval;
        
    if (R < 4.75) {
        in_braces = 1.0; 
    } else {
        potval = internal_pes(R, PH1, TH1, PH2, TH2) / HTOCM; // Hartree
        in_braces = 1.0 - std::exp(-potval * HkT / Temperature); 
    }

    double d = DPJC * in_braces * R * R * std::sin(TH1) * std::sin(TH2);
    
    /* 
    std::cout << "INTERNAL:  " << R << " " << PH1 << " " << TH1 << " " << PH2 << " " << TH2 << "\n";
    std::cout << "POTVAL:    " << potval << " Hartree\n";
    std::cout << "in_braces: " << in_braces << "\n";
    std::cout << "DPJC:      " << DPJC << "\n";
    std::cout << "d:         " << d << "\n\n";
    */

    return d; 
}

int main(int argc, char* argv[])
{
    std::cout << std::fixed << std::setprecision(16);
	MPI_Init(&argc, &argv); 
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t NATOMS = 7;
    NNPIP pes(NATOMS, "model.pt");

    Integrand integrand(pes);

	hep::mpi_vegas_callback<double>( hep::mpi_vegas_verbose_callback<double> );

    std::ofstream out;   
    if (rank == 0) {
        out.open("out.txt", std::fstream::out | std::fstream::app);
        out << std::fixed << std::setprecision(10);
    }

    //std::vector<double> temperatures = linspace(70.0, 500.0, 44);
    std::vector<double> temperatures = linspace(200.0, 200.0, 1);
    for (double T : temperatures) {
        integrand.setTemperature(T);
   
        auto results = hep::mpi_vegas(
            MPI_COMM_WORLD,
            hep::make_integrand<double>(integrand, DIM),
            std::vector<size_t>(20, 100000)
        );

        auto mean = hep::accumulate<hep::weighted_with_variance>(results.begin() + 2, results.end());
        double SVC = mean.value() / (8.0 * M_PI) * AVOGADRO * ALU3 * 1e6; // cm3/mol 
        
        if (rank == 0) {
            out       << T << " " << SVC << "\n"; 
            std::cout << T << " " << SVC << std::endl; 
        }
    }

    MPI_Finalize();

    return 0;
}
