#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <random>
#include <mutex>

#include "scaler.hpp"

#include "c_basis_1_1_1_1_2_1_4_intermolecular.h"
#include "c_basis_4_1_1_1_4_intermolecular.h"
#include "c_basis_4_2_1_4_intermolecular.h"
const int natoms = 7;
const int ndist = natoms * (natoms - 1) / 2;

template <typename Container>
void print(std::string const& name, Container c) {
    std::cout << "[" << name << "]: ";
    for (auto el : c) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

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

void make_yij_intermolecular_exp6(const double * x, double* yij, int natoms)
{
    const double a0 = 2.0;
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

            yij[k] = (1.0 - s) * std::exp(-dst / a0) + s * 1e4 / dst6;
            k++;
        }
    }
}

void fill_poly_dipoleq_ch4_n2(std::vector<size_t> const& npolys, std::vector<double> const& x, Eigen::Ref<Eigen::RowVectorXd> p) 
{
    static std::vector<double> yij(ndist);  
    make_yij_intermolecular_exp6(&x[0], &yij[0], natoms);

    evpoly_1_1_1_1_2_1_4_intermolecular(&yij[0], p.segment(0, npolys[0]));
    evpoly_4_1_1_1_4_intermolecular(&yij[0], p.segment(npolys[0], npolys[1]));
    evpoly_4_2_1_4_intermolecular(&yij[0], p.segment(npolys[0] + npolys[1], npolys[2]));
}

#define FILL_POLY fill_poly_dipoleq_ch4_n2
#include "qmodel.hpp"

/*

void min_crossection()
{
    std::cout << std::fixed << std::setprecision(12);

    double NN_BOND_LENGTH = 2.078;
    auto qmodel = build_qmodel_from_npz("models/dipoleq-sym-ord4-exp7.npz");
    
    const double deg = M_PI / 180.0;
    double TH = 90.0 * deg;
    
    std::vector<double> xyz;
    xyz = xyz_from_internal(1000.0, TH, NN_BOND_LENGTH);
    Eigen::RowVectorXd infq = qmodel.forward(xyz);
    std::cout << "INF q: " << infq << "\n"; 
   
    auto rr = linspace(4.0, 30.0, 261);

    for (size_t k = 0; k < rr.size(); ++k) {
        double R = rr[k];
        xyz = xyz_from_internal(R, TH, NN_BOND_LENGTH);

        Eigen::RowVectorXd q     = qmodel.forward(xyz);
        Eigen::RowVectorXd qcorr = q - infq;

        Eigen::Matrix3d coords(xyz.data()); 
        Eigen::RowVector3d nndip = qcorr * coords.transpose(); 
    
        std::cout << R << "\t" << nndip << "\n";
    }
}
*/

std::vector<double> xyz_from_internal(double R, double PH1, double TH1, double PH2, double TH2, double NN_BOND_LENGTH)
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

    return cart; 
}

void min_crossection_qc_table()
{
    std::cout << std::fixed << std::setprecision(12); 
    
    QModel qmodel; 
    qmodel.init("models/ch4-n2-dipoleq-sym-ord4.npz");
    
    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    // minimal cross-section
    // CCSD(T), aug-cc-pVTZ, N-N 2.078 a0
    std::vector<std::pair<double, Eigen::Vector3d>> qc_cs = {
        std::make_pair(4.50 , Eigen::Vector3d(0.0026633900,  0.0000000000, -0.1439756800)), 
        std::make_pair(4.75 , Eigen::Vector3d(0.0024798600,  0.0000000000, -0.1121835000)), 
        std::make_pair(5.00 , Eigen::Vector3d(0.0022653300,  0.0000000000, -0.0914185500)), 
        std::make_pair(5.25 , Eigen::Vector3d(0.0020083800,  0.0000000000, -0.0760516300)),
        std::make_pair(5.50 , Eigen::Vector3d(0.0017391500,  0.0000000000, -0.0638263100)),
        std::make_pair(5.75 , Eigen::Vector3d(0.0014815600,  0.0000000000, -0.0537495600)),
        std::make_pair(6.00 , Eigen::Vector3d(0.0012489000,  0.0000000000, -0.0453176300)),
        std::make_pair(6.25 , Eigen::Vector3d(0.0010464700,  0.0000000000, -0.0382260800)),
        std::make_pair(6.50 , Eigen::Vector3d(0.0008747200,  0.0000000000, -0.0322597300)),
        std::make_pair(6.75 , Eigen::Vector3d(0.0007314900,  0.0000000000, -0.0272500300)),  
        std::make_pair(7.00 , Eigen::Vector3d(0.0006135100,  0.0000000000, -0.0230563900)),
        std::make_pair(7.25 , Eigen::Vector3d(0.0005171500,  0.0000000000, -0.0195570000)),
        std::make_pair(7.50 , Eigen::Vector3d(0.0004387500,  0.0000000000, -0.0166439800)),
        std::make_pair(7.75 , Eigen::Vector3d(0.0003748900,  0.0000000000, -0.0142218900)),
        std::make_pair(8.00 , Eigen::Vector3d(0.0003225600,  0.0000000000, -0.0122073600)),
        std::make_pair(8.25 , Eigen::Vector3d(0.0002793300,  0.0000000000, -0.0105291600)),
        std::make_pair(8.50 , Eigen::Vector3d(0.0002432700,  0.0000000000, -0.0091274700)),
        std::make_pair(8.75 , Eigen::Vector3d(0.0002129600,  0.0000000000, -0.0079525300)),
        std::make_pair(9.00 , Eigen::Vector3d(0.0001872900,  0.0000000000, -0.0069633200)),
        std::make_pair(9.25 , Eigen::Vector3d(0.0001656000,  0.0000000000, -0.0061263000)),
        std::make_pair(9.50 , Eigen::Vector3d(0.0001468700,  0.0000000000, -0.0054145300)),
        std::make_pair(10.00, Eigen::Vector3d(0.0001172200,  0.0000000000, -0.0042839100)),
        std::make_pair(11.00, Eigen::Vector3d(0.0000777600,  0.0000000000, -0.0028093700)),
        std::make_pair(12.00, Eigen::Vector3d(0.0000538300,  0.0000000000, -0.0019348100)),
        std::make_pair(14.00, Eigen::Vector3d(0.0000282000,  0.0000000000, -0.0010115100)),
        std::make_pair(16.00, Eigen::Vector3d(0.0000162100,  0.0000000000, -0.0005820500)),
        std::make_pair(18.00, Eigen::Vector3d(0.0000099800,  0.0000000000, -0.0003588300)),
        std::make_pair(20.00, Eigen::Vector3d(0.0000064900,  0.0000000000, -0.0002333700)),
        std::make_pair(25.00, Eigen::Vector3d(0.0000026200,  0.0000000000, -0.0000943800)),
        std::make_pair(30.00, Eigen::Vector3d(0.0000012500,  0.0000000000, -0.0000452200)),
    };

    double NN_BOND_LENGTH = 2.078567491;

    std::vector<double> xyz;
    xyz = xyz_from_internal(1000.0, PH1, TH1, PH2, TH2, NN_BOND_LENGTH);
    Eigen::RowVectorXd infq = qmodel.forward(xyz);
    std::cout << "INF q: " << infq << "\n"; 

    std::cout << "  [equilibrium]\n";
    std::cout << "  R \t\t\t NN \t\t\t\t\t QC\n";

    for (size_t k = 0; k < qc_cs.size(); ++k) {
        double R              = qc_cs[k].first;
        Eigen::RowVector3d qc = qc_cs[k].second;

        xyz = xyz_from_internal(R, PH1, TH1, PH2, TH2, NN_BOND_LENGTH);
        Eigen::MatrixXd coords = Eigen::Map<Eigen::Matrix<double, natoms, 3, Eigen::RowMajor>>(xyz.data()); 

        std::cout << "coords:\n" << coords << "\n";
        assert(false);

        Eigen::RowVectorXd q = qmodel.forward(xyz);
        Eigen::RowVectorXd qcorr = q - infq;

        Eigen::RowVector3d nndip = qcorr * coords; 
        
        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(3) << std::setprecision(8) << nndip
                                                       << "\t" << std::right << std::setw(3) << std::setprecision(8) << qc << "\n";
    }
}

void test_configs()
{
    std::cout << std::fixed << std::setprecision(16);
    
    QModel qmodel; 
    qmodel.init("models/ch4-n2-dipoleq-sym-ord4.npz");

    std::vector<double> cc = {
        1.193587416000000,  1.193587416000000, -1.193587416000000,
       -1.193587416000000, -1.193587416000000, -1.193587416000000,
       -1.193587416000000,  1.193587416000000,  1.193587416000000,
        1.193587416000000, -1.193587416000000,  1.193587416000000,
       -6.261753700814096, -2.545862406713278, -4.057956290064579,
       -5.448560860109539, -1.624478785536506, -2.382291160495420,
        0.000000000000000,  0.000000000000000,  0.000000000000000, 
    };

    Eigen::RowVectorXd q = qmodel.forward(cc);  
    Eigen::MatrixXd coords = Eigen::Map<Eigen::Matrix<double, natoms, 3, Eigen::RowMajor>>(cc.data()); 
    Eigen::RowVector3d dip = q * coords;

    std::cout << "Model: " << dip << "\n";
    std::cout << "QC: -0.03015737, -0.00127294, -0.00135275\n";
}

int main()
{
    //test_configs();
    min_crossection_qc_table();

    return 0;
}

