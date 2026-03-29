#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <random>
#include <math.h>
#include <mutex>

#include "scaler.hpp"

#define UNUSED(x) (void)(x)

// FULL DIMENSIONAL DIPOLEQ MODEL, 25/01/2023
#include "c_basis_2_2_1_3_purify.h"
#include "c_basis_2_1_1_1_3_purify.h"
#include "c_basis_1_1_2_1_3_purify.h"

const std::string CURR_FOLDER = "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/external/dipoleq-custom-mlp/";

const int NATOMS = 5;

static const int NDIST = NATOMS * (NATOMS - 1) / 2;
static double YIJ[NDIST];

static double a0 = 2.0;
static double x_i = 6.0;
static double x_f = 12.0;  

template <typename Container>
void print(std::string const& name, Container c) {
    std::cout << "[" << name << "]: ";
    for (auto el : c) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

const char* shift(int* argc, char*** argv)
{
    assert(*argc > 0);
    const char *result = *argv[0];
    *argc -= 1;
    *argv += 1;
    return result;
}

typedef struct {
    int (*run)(const char* program_path, int argc, char **argv);
    const char *id;
    const char *description;
} Subcmd;

int subcmd_run_dipole_from_arg(const char *program_path, int argc, char **argv);

#define DEFINE_SUBCMD(name, desc) \
    { \
        .run = subcmd_##name, \
        .id = #name, \
        .description = desc, \
    }

Subcmd subcmds[] = {
    DEFINE_SUBCMD(run_dipole_from_arg, "Run dipole for BF coordinates provided as the command line arguments"),
};
#define SUBCMDS_COUNT (sizeof(subcmds)/sizeof(subcmds[0]))

Subcmd *find_subcmd_by_id(const char *id) 
{
    for (size_t k = 0; k < SUBCMDS_COUNT; ++k) {
        if (strcmp(subcmds[k].id, id) == 0) {
            return &subcmds[k];
        }
    }

    return NULL;
}

void usage(const char* program_path)
{
    fprintf(stderr, "Usage: %s [subcommand]\n", program_path);
    fprintf(stderr, "Subcommands:\n");

    int width = 0;
    for (size_t i = 0; i < SUBCMDS_COUNT; ++i) {
        int len = strlen(subcmds[i].id);
        if (width < len) width = len;
    }

    for (size_t i = 0; i < SUBCMDS_COUNT; ++i) {
        fprintf(stderr, "    %-*s - %s\n", width, subcmds[i].id, subcmds[i].description);
    }
}

double sw(double x, double x_i, double x_f) 
{
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

void make_yij_4q_purify_exp4(const double *x, double *yij, int natoms, double x_i, double x_f)
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
            
            if (i == 0 && j == 1) { yij[k] = std::exp(-dst / a0); k = k + 1; continue; } // N1  N2
            if (i == 0 && j == 2) { yij[k] = std::exp(-dst / a0); k = k + 1; continue; } // N1  N1'
            if (i == 0 && j == 3) { yij[k] = std::exp(-dst / a0); k = k + 1; continue; } // N1  N2'
            if (i == 1 && j == 2) { yij[k] = std::exp(-dst / a0); k = k + 1; continue; } // N2  N1'
            if (i == 1 && j == 3) { yij[k] = std::exp(-dst / a0); k = k + 1; continue; } // N2  N2'
            if (i == 2 && j == 3) { yij[k] = std::exp(-dst / a0); k = k + 1; continue; } // N1' N2'

            double dst4 = dst * dst * dst * dst;
            double s = sw(dst, x_i, x_f);
            yij[k] = (1.0 - s) * std::exp(-dst / a0) + s * 1e2 / dst4;

            k++;
        }
    }
}

void make_yij_exp(const double *x, double *yij, double a0)
{
    size_t k = 0;
    for (size_t i = 0; i < NATOMS; ++i) {
        for (size_t j = i + 1; j < NATOMS; ++j) {
            double drx = x[3*i    ] - x[3*j    ]; 
            double dry = x[3*i + 1] - x[3*j + 1];
            double drz = x[3*i + 2] - x[3*j + 2];
            double dst = std::sqrt(drx*drx + dry*dry + drz*drz);

            yij[k] = std::exp(-dst / a0);  
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

void QMODEL_FILL_POLY(std::vector<size_t> const& npolys, double* x, Eigen::Ref<Eigen::RowVectorXd> p) 
{
    make_yij_exp(x, YIJ, a0);

    evpoly_1_1_2_1_3_purify(YIJ, p.segment(0, npolys[0]));
    evpoly_2_1_1_1_3_purify(YIJ, p.segment(npolys[0], npolys[1]));
    evpoly_2_2_1_3_purify(YIJ, p.segment(npolys[0] + npolys[1], npolys[2]));
}

void QMODEL_FILL_POLY_INF(std::vector<size_t> const& npolys, Eigen::Ref<Eigen::RowVectorXd> p) {
    p.segment(0, npolys[0])                     = Eigen::RowVectorXd::Zero(npolys[0]);
    p.segment(npolys[0], npolys[1])             = Eigen::RowVectorXd::Zero(npolys[1]);
    p.segment(npolys[0] + npolys[1], npolys[2]) = Eigen::RowVectorXd::Zero(npolys[2]);
}

#define QMODEL_IMPLEMENTATION
#include "qmodel.hpp"

typedef struct {
    double c[15];
} XYZ4q;

typedef struct {
    double R;
    double r;
    double Theta;
} BF;

XYZ4q xyz4q_from_bf(BF bf) 
{
    return XYZ4q {
        .c  = { /* N1  */  bf.r/2.0 * sin(bf.Theta), 0.0,  bf.r/2.0 * cos(bf.Theta),
                /* N2  */ -bf.r/2.0 * sin(bf.Theta), 0.0, -bf.r/2.0 * cos(bf.Theta),
                /* N1' */  bf.r/2.0 * cos(bf.Theta), 0.0, -bf.r/2.0 * sin(bf.Theta),
                /* N2' */ -bf.r/2.0 * cos(bf.Theta), 0.0,  bf.r/2.0 * sin(bf.Theta),
                /* Ar  */  0.0, 0.0, bf.R } 
    };
}

Eigen::Vector3d dipole_from_xyz4q(QModel & qmodel, XYZ4q xyz4q)
{
    Eigen::RowVectorXd q     = qmodel.forward(xyz4q.c);
    Eigen::RowVectorXd infq  = qmodel.forward_inf();
    Eigen::RowVectorXd qcorr = q - infq;
    Eigen::MatrixXd coords   = Eigen::Map<Eigen::Matrix<double, NATOMS, 3, Eigen::RowMajor>>(xyz4q.c);
    Eigen::RowVector3d dip   = qcorr * coords;

    return dip;
}

/*
void min_crossection()
{
    std::cout << std::fixed << std::setprecision(12);

    QModel qmodel; 
    qmodel.init("models/n2-ar-dipoleq-sym-ord4-exp6.npz");

    double NN_BOND_LENGTH = 2.078;
    
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

        Eigen::MatrixXd coords = Eigen::Map<Eigen::Matrix<double, natoms, 3, Eigen::RowMajor>>(xyz.data()); 
        Eigen::RowVector3d nndip = qcorr * coords.transpose(); 
    
        std::cout << R << "\t" << nndip << "\n";
    }
}
*/

/*
void min_crossection_qc_table()
{
    std::cout << std::fixed << std::setprecision(12); 
    
    QModel qmodel; 
    qmodel.init("models/n2-ar-dipoleq-effquad-1.npz");
    
    const double deg = M_PI / 180.0;
    double TH = 90.0 * deg;

    // minimal cross-section
    // CCSD(T), aug-cc-pVTZ, N-N 2.078 a0
    std::vector<std::pair<double, Eigen::Vector3d>> qc_cs = {
        std::make_pair(4.50, Eigen::Vector3d( 0.0000000100, 0.0000000000, -0.0028958900)),  
        std::make_pair(4.75, Eigen::Vector3d(-0.0000000100, 0.0000000000,  0.0006134200)),
        std::make_pair(5.00, Eigen::Vector3d( 0.0000000200, 0.0000000000,  0.0033565900)),
        std::make_pair(5.25, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0052885700)),
        std::make_pair(5.50, Eigen::Vector3d( 0.0000000000, 0.0000000200,  0.0064974300)),
        std::make_pair(5.75, Eigen::Vector3d( 0.0000000100, 0.0000000000,  0.0071221400)),
        std::make_pair(6.00, Eigen::Vector3d( 0.0000000100, 0.0000000100,  0.0073088100)),  
        std::make_pair(6.25, Eigen::Vector3d( 0.0000000000, 0.0000000100,  0.0071889800)),
        std::make_pair(6.50, Eigen::Vector3d( 0.0000000000, 0.0000000100,  0.0068703600)),
        std::make_pair(6.75, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0064352000)),
        std::make_pair(7.00, Eigen::Vector3d( 0.0000000000, 0.0000000100,  0.0059428800)),
        std::make_pair(7.25, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0054338300)),
        std::make_pair(7.50, Eigen::Vector3d( 0.0000000000, 0.0000000100,  0.0049342600)),
        std::make_pair(7.75, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0044598000)),
        std::make_pair(8.00, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0040189100)),
        std::make_pair(8.25, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0036152200)),
        std::make_pair(8.50, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0032494100)),
        std::make_pair(8.75, Eigen::Vector3d( 0.0000000100, 0.0000000100,  0.0029203200)),
        std::make_pair(9.00, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0026258600)),
        std::make_pair(9.25, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0023633800)),
        std::make_pair(9.50, Eigen::Vector3d( 0.0000000200, 0.0000000000,  0.0021300000)),
        std::make_pair(10.0, Eigen::Vector3d( 0.0000000000, 0.0000000100,  0.0017390700)),
        std::make_pair(10.5, Eigen::Vector3d( 0.0000000100, 0.0000000100,  0.0014312700)),
        std::make_pair(11.0, Eigen::Vector3d( 0.0000000600, 0.0000000000,  0.0011879700)),
        std::make_pair(12.0, Eigen::Vector3d( 0.0000000200, 0.0000000000,  0.0008387900)),
        std::make_pair(14.0, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0004550500)),
        std::make_pair(16.0, Eigen::Vector3d( 0.0000000000, 0.0000000200,  0.0002686600)),
        std::make_pair(18.0, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0001686400)),
        std::make_pair(20.0, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0001110900)),
        std::make_pair(25.0, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0000457900)),
        std::make_pair(30.0, Eigen::Vector3d( 0.0000000000, 0.0000000000,  0.0000221500)),
    };

    double NN_BOND_LENGTH = 2.078;
    
    std::cout << "  (N-N 2.078 a0 [equilibrium])\n";
    std::cout << "  R \t\t\t NN \t\t\t\t\t QC\n";

    for (size_t k = 0; k < qc_cs.size(); ++k) {
        double R     = qc_cs[k].first;
        Eigen::RowVector3d qc    = qc_cs[k].second;

        std::vector<double> xyz = xyz_from_internal(R, TH, NN_BOND_LENGTH);
        Eigen::RowVector3d nndip = dipole_from_xyz(qmodel, xyz);
        
        std::cout << "  " << std::setprecision(2) << R << "\t" << std::right << std::setw(3) << std::setprecision(8) << nndip
                                                       << "\t" << std::right << std::setw(3) << std::setprecision(8) << qc << "\n";
    }
}
*/

int subcmd_run_dipole_from_arg(const char *program_path, int argc, char **argv)
{
    UNUSED(program_path);

    if (argc != 3) {
        fprintf(stderr, "ERROR: expected 3 command line arguments [R, r, Theta] to evaluate dipole\n");
        exit(1);
    } 

    const char* Rs     = shift(&argc, &argv);
    const char* rs     = shift(&argc, &argv);
    const char* Thetas = shift(&argc, &argv);

    BF bf = {
        .R     = std::stof(Rs),
        .r     = std::stof(rs),
        .Theta = std::stof(Thetas)
    };

    XYZ4q xyz = xyz4q_from_bf(bf);
   
    QModel qmodel_short;
    QModel qmodel_long;

    bool log = false;
    qmodel_short.init(CURR_FOLDER + "models/n2-ar-dipoleq-effquad-2-short.npz", log);
    qmodel_long.init(CURR_FOLDER + "models/n2-ar-dipoleq-effquad-2-long.npz", log);

    a0 = 2.0;
    Eigen::Vector3d dipole_short = dipole_from_xyz4q(qmodel_short, xyz);

    a0 = 6.0;
    Eigen::Vector3d dipole_long = dipole_from_xyz4q(qmodel_long, xyz);

    double s = sw(bf.R, 9.5, 12.5);
    Eigen::Vector3d result = (1.0 - s) * dipole_short + s * dipole_long;

    printf("%.15f %.15f %.15f\n", result(0), result(1), result(2));

    return 0;
}

int main(int argc, char* argv[])
{
    const char* program_path = shift(&argc, &argv);

    if (argc <= 0) {
        usage(program_path);
        fprintf(stderr, "ERROR: no subcommand is provided\n");
        exit(1);
    }

    const char *subcmd_id = shift(&argc, &argv);
    Subcmd *subcmd = find_subcmd_by_id(subcmd_id);
    if (subcmd != NULL) {
        subcmd->run(program_path, argc, argv);
    } else {
        usage(program_path);
        fprintf(stderr, "ERROR: unknown subcommand `%s`\n", subcmd_id);
        exit(1);
    }

    //test_config();
    //min_crossection_qc_table();
    //min_crossection();

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

