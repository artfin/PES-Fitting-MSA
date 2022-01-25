#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

const double BOHRTOANG = 0.52917721067;
const double HTOCM     = 2.194746313702e5;

extern "C" {
    void c_evmono(double* x, double* m);
    void c_evpoly(double* m, double* p);
}

void print(std::string const& s, double const* const p, const size_t sz) {

    std::cout << "[" << s << "]:\n";
    for (size_t k = 0; k < sz; ++k) {
        std::cout << k << " " << p[k] << "\n";
    }
    std::cout << "-------------------\n";
} 

struct Atom {
    double x, y, z;
    friend std::ostream& operator<<(std::ostream& os, const Atom& atom);
};

std::ostream& operator<<(std::ostream& os, const Atom& atom) {
    os << atom.x << "\t" << atom.y << "\t" << atom.z;
    return os;
}

struct Config {
    std::vector<Atom> atoms;
    double energy;
}; 

std::vector<Config> load(std::string const& filename, const size_t nconfigs, const size_t natoms)
{
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        std::cerr << "Failed to open the file!\n";
        exit(1);
    }
    
    std::string line;
    std::vector<Atom> atoms;
    
    std::istringstream iss;
    char sym;
    double x, y, z;
    double energy;
    std::vector<Config> configs;

    for (size_t k = 0; k < nconfigs; ++k) {

        Config c;

        std::getline(ifs, line);
        
        std::getline(ifs, line);
        iss.str(line);
        iss >> c.energy;
        iss.clear();

        for (size_t j = 0; j < natoms; ++j) {
            std::getline(ifs, line);
            iss.str(line);
            
            iss >> sym >> x >> y >> z;
            iss.clear();

            Atom atom{x, y, z};
            c.atoms.push_back(atom);
        }

        configs.push_back(c);
    }

    return configs;
}

double** init2d(const size_t x, const size_t y) {
    double** a = new double* [x];
    for (size_t i = 0; i < x; ++i) {
        a[i] = new double [y];
    }

    return a;
}

void free2d(double ** a, const size_t x, const size_t y) {
    (void) y;

    for (size_t i = 0; i < x; ++i) {
        delete [] a[i];
    }

    delete a;
}

int main()
{
    std::cout << std::fixed << std::setprecision(16);

    const size_t NCONFIGS = 71610;

    const size_t NATOMS = 7;
    const size_t NDIS = NATOMS * (NATOMS - 1) / 2.0; 

    double** yij = init2d(NCONFIGS, NDIS); 

    auto configs = load("./ch4-n2-energies.xyz", NCONFIGS, NATOMS);

    const double a0 = 2.0;
    
    for (size_t n = 0; n < NCONFIGS; ++n) {
        auto& c = configs[n];

        size_t k = 0;
        for (size_t i = 0; i < NATOMS; ++i) {
            for (size_t j = i + 1; j < NATOMS; ++j) {

                /*
                if (i == 0 && j == 1) { yij[n][k] = 0.0; k++; continue; } // H1 H2
                if (i == 0 && j == 2) { yij[n][k] = 0.0; k++; continue; } // H1 H3
                if (i == 0 && j == 3) { yij[n][k] = 0.0; k++; continue; } // H1 H4
                if (i == 1 && j == 2) { yij[n][k] = 0.0; k++; continue; } // H2 H3
                if (i == 1 && j == 3) { yij[n][k] = 0.0; k++; continue; } // H2 H4
                if (i == 2 && j == 3) { yij[n][k] = 0.0; k++; continue; } // H3 H4
                if (i == 0 && j == 6) { yij[n][k] = 0.0; k++; continue; } // H1 C
                if (i == 1 && j == 6) { yij[n][k] = 0.0; k++; continue; } // H2 C
                if (i == 2 && j == 6) { yij[n][k] = 0.0; k++; continue; } // H3 C
                if (i == 3 && j == 6) { yij[n][k] = 0.0; k++; continue; } // H4 C
               
                if (i == 4 && j == 5) { yij[n][k] = 0.0; k++; continue; } // N1 N2 
                */

                double drx = c.atoms[i].x - c.atoms[j].x;
                double dry = c.atoms[i].y - c.atoms[j].y;
                double drz = c.atoms[i].z - c.atoms[j].z;

                yij[n][k] = std::sqrt(drx*drx + dry*dry + drz*drz);
                yij[n][k] /= BOHRTOANG;
                yij[n][k] = std::exp(-yij[n][k]/a0);

                k++; 
            }
        }
    }

    const size_t NCOORDS  = 21;
    const size_t NMON = 2892;
    const size_t NPOLY = 650;
   
    double * m = new double [NMON];
    double * p = new double [NPOLY];
    
    //double** c_A = init2d(NCONFIGS, NPOLY); 
    //double * v = new double [NCONFIGS];

    Eigen::Matrix<double, Eigen::Dynamic, 1> b; 
    b.resize(NCONFIGS);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
    A.resize(NCONFIGS, NPOLY);

    for (size_t n = 0; n < NCONFIGS; ++n) {

        print("yij", yij[n], NDIS);
        
        c_evmono(yij[n], m);
        print("m", m, NMON);

        c_evpoly(m, p);
        print("p", p, NPOLY);
        
        for (size_t k = 0; k < NPOLY; ++k) {
            A(n, k) = p[k];
        }

        b(n) = configs[n].energy;
        break;
    }
   
    /* 
    Eigen::VectorXd coeff = A.fullPivLu().solve(b.matrix());

    double rmse = 0.0;
    for (size_t n = 0; n < NCONFIGS; ++n) {
        c_evmono(yij[n], m);
        c_evpoly(m, p);

        double fit = 0.0;
        for (size_t k = 0; k < NPOLY; ++k) {
            fit += coeff(k) * p[k];
        }

        double energy = configs[n].energy;

        rmse += (fit - energy) * (fit - energy);

        std::cout << "n: " << n << "; energy (cm-1): " << energy * HTOCM << "; fit (cm-1): " << fit * HTOCM << "\n"; 
    }

    rmse = std::sqrt(rmse / NCONFIGS);
    std::cout << "RMSE (cm-1): " << rmse * HTOCM << "\n";
    */

    free2d(yij, NCONFIGS, NDIS);
    //free2d(A, NCONFIGS, NPOLY);
    free(m);
    free(p);
    
    return 0;
}
