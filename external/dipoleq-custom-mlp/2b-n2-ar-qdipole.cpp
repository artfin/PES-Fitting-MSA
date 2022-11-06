#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <random>
#include <mutex>

#include "scaler.hpp"

#include "c_basis_1_1_1_4_purify.h"
#include "c_basis_2_1_4_purify.h"
const int natoms = 3;
const int ndist = natoms * (natoms - 1) / 2;
const int npoly = 18; 

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

void make_yij_2_1_4_purify_exp5(const double * x, double* yij, int natoms)
{
    const double a0 = 2.0;
    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            
            drx = x[3*i    ] - x[3*j    ]; 
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];
            
            double dst = std::sqrt(drx*drx + dry*dry + drz*drz);
            
            if (i == 0 && j == 1) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // N1 N2 

            double dst5 = dst * dst * dst * dst * dst;
            double s = sw(dst);
            yij[k] = (1.0 - s) * std::exp(-dst / a0) + s * 1e3 / dst5;

            k++;
        }
    }
}

void make_yij_2_1_4_purify_exp6(const double * x, double* yij, int natoms)
{
    const double a0 = 2.0;
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
            yij[k] = (1.0 - s) * std::exp(-dst / a0) + s * 1e4 / dst6;

            k++;
        }
    }
}

void make_yij_2_1_4_purify_exp7(const double * x, double* yij, int natoms)
{
    const double a0 = 2.0;
    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            
            drx = x[3*i    ] - x[3*j    ]; 
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];
            
            double dst = std::sqrt(drx*drx + dry*dry + drz*drz);
            
            if (i == 0 && j == 1) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // N1 N2 

            double dst7 = dst * dst * dst * dst * dst * dst * dst;
            double s = sw(dst);
            yij[k] = (1.0 - s) * std::exp(-dst / a0) + s * 1e5 / dst7;

            k++;
        }
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


void fill_poly_dipoleq_n2_ar(std::vector<size_t> const& npolys, std::vector<double> const& x, Eigen::Ref<Eigen::RowVectorXd> p) 
{
    static std::vector<double> yij(ndist);  
    make_yij_2_1_4_purify_exp7(&x[0], &yij[0], natoms);

    evpoly_1_1_1_4_purify(&yij[0], p.segment(0, npolys[0]));
    evpoly_2_1_4_purify(&yij[0], p.segment(npolys[0], npolys[1]));
}

#define FILL_POLY fill_poly_dipoleq_n2_ar
#include "qmodel.hpp"

std::vector<double> xyz_from_internal(double R, double TH, double NN_BOND_LENGTH) {
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

    return cart;
}

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

void min_crossection_qc_table()
{
    std::cout << std::fixed << std::setprecision(12); 
    //auto qmodel = build_qmodel_from_npz("models/dipoleq-sym-ord4-exp7-noscale.npz");
    //auto qmodel = build_qmodel_from_npz("models/dipoleq-sym-ord4-exp5.npz");
    //auto qmodel = build_qmodel_from_npz("models/dipoleq-sym-ord4-exp6.npz");
    auto qmodel = build_qmodel_from_npz("models/dipoleq-sym-ord4-exp7.npz");
    
    const double deg = M_PI / 180.0;
    double TH = 90.0 * deg;

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

    double NN_BOND_LENGTH = 2.078;
    
    std::vector<double> xyz;
    xyz = xyz_from_internal(1000.0, TH, NN_BOND_LENGTH);
    Eigen::RowVectorXd infq = qmodel.forward(xyz);
    std::cout << "INF q: " << infq << "\n"; 

    std::cout << "  (N-N 2.078 a0 [equilibrium])\n";
    std::cout << "  R \t\t\t NN \t\t\t\t\t QC\n";

    for (size_t k = 0; k < qc_cs.size(); ++k) {
        double R     = qc_cs[k].first;
        Eigen::RowVector3d qc    = qc_cs[k].second;

        xyz = xyz_from_internal(R, TH, NN_BOND_LENGTH);
        Eigen::RowVectorXd q = qmodel.forward(xyz);
        Eigen::RowVectorXd qcorr = q - infq;

        Eigen::Matrix3d coords(xyz.data()); 
        Eigen::RowVector3d nndip = qcorr * coords.transpose(); 
        
        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(3) << std::setprecision(8) << nndip
                                                       << "\t" << std::right << std::setw(3) << std::setprecision(8) << qc << "\n";
    }
}


int main()
{
    min_crossection_qc_table();
    min_crossection();

    return 0;
}


/*
Eigen::Vector3d QModel::forward2(std::vector<double> const& x) {

    std::vector<double> yij(ndist);  
    make_yij_2_1_4_purify(&x[0], &yij[0], natoms);
    
    print("yij", yij);

    Eigen::RowVectorXd p1 = Eigen::RowVectorXd::Zero(31);
    evpoly_1_1_1_4_purify(&yij[0], p1);

    Eigen::RowVectorXd p2 = Eigen::RowVectorXd::Zero(17);
    evpoly_2_1_4_purify(&yij[0], p2);

    std::cout << "p1:\n" << p1 << "\n"; 
    std::cout << "p2:\n" << p2 << "\n";

    Eigen::RowVectorXd p(48);
    p << p1, p2;

    Eigen::RowVectorXd ptr(48); 
    ptr = xscaler.transform(p);  
    std::cout << "ptr:\n" << ptr << "\n";

    Eigen::RowVectorXd qtr1 = modules[0].forward(ptr.segment(0, 31));
    Eigen::RowVectorXd qtr2 = modules[1].forward(ptr.segment(31, 17));

    Eigen::RowVectorXd qtr(3);
    qtr << qtr1, qtr2;
    std::cout << "qtr: " << qtr << "\n";

    Eigen::Matrix3d coords(x.data()); 
    std::cout << "coords:\n " << coords.transpose() << "\n";
    
    Eigen::Vector3d dip_tr = qtr * coords.transpose(); 
    std::cout << "dip_tr: " << dip_tr << "\n";

    return yscaler.inverse_transform(dip_tr);
} 
*/

