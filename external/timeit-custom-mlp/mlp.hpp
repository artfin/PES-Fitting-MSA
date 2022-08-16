#ifndef MLP_H
#define MLP_H 

#include "scaler.hpp"

// https://github.com/rogersce/cnpy
#include "cnpy.h"

double SiLU(double x) {
    return x / (1.0 + std::exp(-x)); 
}

double dSiLU(double x) {
    double x_exp = std::exp(-x);
    double denominator = 1.0 + x_exp; 
    return 1.0 / denominator + x * x_exp / denominator / denominator;
}

template <typename Container>
void print(std::string const& name, Container c) {
    std::cout << "[" << name << "]: ";
    for (auto el : c) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

struct MLPModel 
{
    MLPModel(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases,
             StandardScaler xscaler, StandardScaler yscaler);

    double forward(std::vector<double> const& x);
    void   backward(std::vector<double> const& x, std::vector<double> & dx);

    std::vector<size_t> architecture;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::RowVectorXd> biases;
    std::vector<Eigen::RowVectorXd> neurons;
    std::vector<Eigen::RowVectorXd> cache_neurons;

    StandardScaler xscaler;
    StandardScaler yscaler;
    
    //void make_yij(const double* x, double* yij, int natoms);
    void make_drdx(Eigen::Ref<Eigen::MatrixXd> drdx, const double* x);
    void make_dydr(Eigen::Ref<Eigen::MatrixXd> dydr, const double* yij);

    double** alloc_2d(int nrows, int ncols);
    void print_2d(double** a, int nrows, int ncols);
    void delete_2d(double** a, int nrows, int ncols);
    
    Eigen::RowVectorXd p; 
    Eigen::RowVectorXd ptr; 

    Eigen::MatrixXd drdx;
    Eigen::MatrixXd dydr;
    Eigen::MatrixXd dpdx;

    Eigen::MatrixXd jac;
    
    Eigen::RowVectorXd dEdp; 
    Eigen::RowVectorXd dEdx; 
    
    Eigen::MatrixXd t1;
    Eigen::VectorXd t2; 
    Eigen::MatrixXd t3;

    std::vector<double> yij; 
};

MLPModel::MLPModel(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases,
                   StandardScaler xscaler, StandardScaler yscaler) :
   architecture(architecture), weights(weights), biases(biases), xscaler(xscaler), yscaler(yscaler)

{
    size_t sz = architecture.size();

    neurons.resize(sz);
    cache_neurons.resize(sz);

    for (size_t k = 0; k < sz; ++k) {
        neurons[k]       = Eigen::RowVectorXd::Zero(architecture[k]);
        cache_neurons[k] = Eigen::RowVectorXd::Zero(architecture[k]);
    }

    p   = Eigen::RowVectorXd::Zero(npoly);
    ptr = Eigen::RowVectorXd::Zero(npoly);
    
    yij.resize(ndist); 
    drdx = Eigen::MatrixXd::Zero(3 * natoms, ndist);
    dydr = Eigen::MatrixXd::Zero(ndist, ndist);
    
    jac  = Eigen::MatrixXd::Zero(ndist, npoly);
    dpdx = Eigen::MatrixXd::Zero(3 * natoms, npoly);
    
    dEdp = Eigen::RowVectorXd::Zero(npoly); 
    dEdx = Eigen::RowVectorXd::Zero(3 * natoms);

    t1.resize(npoly, architecture[1]);
    t2.resize(npoly);
    t3.resize(3 * natoms, ndist); 
}

double MLPModel::forward(std::vector<double> const& x)
/*
 * input  : atomic coordinates [a0]
 * output : energy [cm-1] 
 */
{
    MAKE_YIJ(&x[0], &yij[0], natoms);

    EVPOLY(&yij[0], p);

    /*
     * A stub to get working the potential for rigid case CH4-N2 
     */
    /*
    for (int k = 78; k > 0; k--) {
        p(k) = p(k - 1);
    }
    p(0) = 1.0; 
    */

    ptr = xscaler.transform(p);

    //std::cout << "p: " << p << std::endl;
    //std::cout << "ptr: " << ptr << std::endl;

    size_t sz = architecture.size();
    size_t hidden_dims = sz - 2;
   
    // in_features 
    neurons[0] = ptr;

    for (size_t i = 1; i <= hidden_dims; ++i) {
        cache_neurons[i] = neurons[i - 1] * weights[i - 1];
        cache_neurons[i].noalias() += biases[i - 1];
        
        for (size_t j = 0; j < neurons[i].size(); ++j) {
            neurons[i](j)       = SiLU(cache_neurons[i](j));
            cache_neurons[i](j) = dSiLU(cache_neurons[i](j));
        }
    }

    // last linear operation -> out_features [no activation]
    neurons[sz - 1] = neurons[sz - 2] * weights[sz - 2];
    neurons[sz - 1] += biases[sz - 2];

    return yscaler.inverse_transform(neurons[sz - 1])(0);
}

void MLPModel::backward(std::vector<double> const& x, std::vector<double> & dx)
{
    make_drdx(drdx, &x[0]);
    make_dydr(dydr, &yij[0]);
    EVPOLY_JAC(jac, &yij[0]);  // 30-40 mcs
    
    // 110-120 mcs
    // (36 x 27) x (27 x 27) x (27 x 1898)
    t3.noalias()   = drdx * dydr;
    dpdx.noalias() = t3 * jac;

    //int ncol = 10; 
    //for (int k = 0; k < 27; ++k) {
    //    std::cout << "dpdx(" << k << ", " << ncol << "): " << dpdx(k, ncol) << std::endl;
    //}

    // This general approach is PAINSTAKINGLY slow
    //   (more than 1,000 times slower!)
    /* 
    size_t sz = architecture.size();
    size_t hidden_dims = sz - 2;

    eye = Eigen::MatrixXd::Identity(weights[0].rows(), weights[0].rows());
    for (size_t i = 1; i <= hidden_dims; ++i) {
        eye *= weights[i - 1];
        eye *= cache_neurons[i].asDiagonal();
    }
    eye *= weights[sz - 2];
    */

    // Operation 
    //     eye = weights[0] * cache_neurons[1].asDiagonal() * weights[1];
    // is equivalent time-wise to manually unrolling the first multiplication into column-wise multiplication by number 
    //     eye = weights[0].array().rowwise() * cache_neurons[1].array();
    //     eye = eye * weights[1]; 
    //  
    
    t1 = weights[0].array().rowwise() * cache_neurons[1].array();
    t2.noalias() = t1 * weights[1]; 

    dEdp = t2.transpose().array() / xscaler.scale.array(); // component-wise division

    dEdx = dEdp * dpdx.transpose();
    dEdx *= yscaler.scale[0];

    Eigen::VectorXd::Map(&dx[0], 3 * natoms) = dEdx; 
    
    //std::cout << "dEdp: " << dEdp.rows() << "x" << dEdp.cols() << std::endl;
    //std::cout << "dpdx: " << dpdx.cols() << "x" << dpdx.rows() << std::endl;
    //std::cout << "dEdp: " << t2.transpose().array() << std::endl;
    //std::cout << "dpdx: " << dpdx << std::endl;
}

/*
void MLPModel::make_yij(const double * x, double* yij, int natoms)
{
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
            
            drx = ATOMX(x, i) - ATOMX(x, j);
            dry = ATOMY(x, i) - ATOMY(x, j);
            drz = ATOMZ(x, i) - ATOMZ(x, j);
            
            double dst = std::sqrt(drx*drx + dry*dry + drz*drz);
            //yij[k] = std::exp(-dst/2.0);
            
            //if (i == 0 && j == 1) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 H2
            //if (i == 0 && j == 2) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 H3
            //if (i == 0 && j == 3) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 H4
            //if (i == 1 && j == 2) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H2 H3
            //if (i == 1 && j == 3) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H2 H4
            //if (i == 2 && j == 3) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H3 H4
            //if (i == 0 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H1 C
            //if (i == 1 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H2 C
            //if (i == 2 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H3 C
            //if (i == 3 && j == 6) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // H4 C
            //if (i == 4 && j == 5) { yij[k] = exp(-dst/2.0); k = k + 1; continue; } // N1 N2

     
            //yij[k] = 1.0 / yij[k]; 
            //yij[k] = std::exp(-yij[k]/a0);
            //yij[k] = 1.0/(yij[k]*yij[k]*yij[k]*yij[k]);
            //yij[k] = 1000.0/(yij[k]*yij[k]*yij[k]*yij[k]*yij[k]*yij[k]); 
           
            //double yij6 = yij[k]*yij[k]*yij[k]*yij[k]*yij[k]*yij[k];
            
            double dst6 = dst * dst * dst * dst * dst * dst;
            double s = sw(dst);
            yij[k] = (1.0 - s) * std::exp(-dst / 2.0) + s * 1e4 / dst6;

            k++;
        }
    }
}
*/

void MLPModel::make_drdx(Eigen::Ref<Eigen::MatrixXd> drdx, const double* x) {
    double drx, dry, drz;
    double dr_norm;
    int k = 0;

    for (int i = 0; i < natoms; ++i) {
        for (int j = i + 1; j < natoms; ++j) {
            drx = x[3*i    ] - x[3*j    ];
            dry = x[3*i + 1] - x[3*j + 1];
            drz = x[3*i + 2] - x[3*j + 2];

            dr_norm = std::sqrt(drx*drx + dry*dry + drz*drz);

            drdx(3*i, k)     =  drx / dr_norm;
            drdx(3*i + 1, k) =  dry / dr_norm;
            drdx(3*i + 2, k) =  drz / dr_norm;
            drdx(3*j, k)     = -drx / dr_norm;
            drdx(3*j + 1, k) = -dry / dr_norm;
            drdx(3*j + 2, k) = -drz / dr_norm;

            k++;
        }
    }
}

void MLPModel::make_dydr(Eigen::Ref<Eigen::MatrixXd> dydr, const double* yij) {
    
    for (int i = 0; i < dydr.rows(); ++i) {
        dydr(i, i) = -1.0 / a0 * yij[i];
    }
}

StandardScaler build_scaler(cnpy::npz_t& npz, std::string const& scaler_type) {

    assert(scaler_type == "xscaler" || scaler_type == "yscaler");

    std::map<std::string, cnpy::NpyArray>::iterator it;
    std::string key;
   
    key = scaler_type + ".mean";
    it = npz.find(key);
    assert(it != npz.end() && "scaler.mean does not exist");
     
    const cnpy::NpyArray np_mean = it->second; 
    assert(np_mean.shape.size() == 1);
    
    const double* pm = np_mean.data<double>();
    auto mean = Eigen::RowVectorXd(np_mean.shape[0]);
    mean = Eigen::Map<const Eigen::RowVectorXd>(pm, np_mean.shape[0]);

    key = scaler_type + ".scale";
    it = npz.find(key);
    assert(it != npz.end() && "scaler.scale does not exist");

    const cnpy::NpyArray np_scale = it->second;
    assert(np_scale.shape.size() == 1);

    const double* ps = np_scale.data<double>();
    auto scale = Eigen::RowVectorXd(np_scale.shape[0]);
    scale = Eigen::Map<const Eigen::RowVectorXd>(ps, np_scale.shape[0]);

    return StandardScaler(mean, scale);
}

MLPModel build_model_from_npz(std::string const& npz_fname)
{
    cnpy::npz_t npz = cnpy::npz_load(npz_fname);

    std::map<std::string, cnpy::NpyArray>::iterator it; 
    it = npz.find("architecture");
    assert(it != npz.end() && "architecture does not exist");

    const cnpy::NpyArray np_architecture = it->second;
    assert(np_architecture.word_size == sizeof(size_t));
    const size_t* p = np_architecture.data<size_t>();
    const size_t sz = np_architecture.shape[0];
    auto architecture = std::vector<size_t>(p, p + sz);
    print("architecture", architecture);

    //it = npz.find("activation");
    //assert (it != npz.end() && "activation does not exist");
    //const cnpy::NpyArray np_activation = it->second;
    //assert(np_activation.word_size == sizeof(wchar_t));
    //const wchar_t* txt = np_activation.data<wchar_t>(); 
    //std::wstring wstr(txt); 
    //std::wcout << "ACTIVATION: " << wstr << std::endl;
        
    std::vector<Eigen::MatrixXd> weights(architecture.size() - 1); 
    int wc = 0;

    std::vector<Eigen::RowVectorXd> biases(architecture.size() - 1);
    int bc = 0;

    for (auto it = npz.begin(); it != npz.end(); ++it) {
        std::string name = it->first;
       
        size_t found_weight = name.find("weight");
        if (found_weight != std::string::npos) {
            const cnpy::NpyArray w = it->second; 
            assert(w.shape.size() == 2);
            
            const double* pw = w.data<double>();
            weights[wc] = Eigen::MatrixXd(w.shape[0], w.shape[1]);
            weights[wc] = Eigen::Map<const Eigen::MatrixXd>(pw, w.shape[0], w.shape[1]);
            
            //std::cout << "weights[" << wc << "] = " << weights[wc].rows() << "x" << weights[wc].cols() << std::endl;
            wc++;
        }

        size_t found_bias = name.find("bias");
        if (found_bias != std::string::npos) {
            const cnpy::NpyArray b = it->second;
            assert(b.shape.size() == 1);

            const double * pb = b.data<double>();
            biases[bc] = Eigen::RowVectorXd(b.shape[0]);
            biases[bc] = Eigen::Map<const Eigen::RowVectorXd>(pb, b.shape[0]);

            bc++;
        }
    } 

    auto xscaler = build_scaler(npz, "xscaler");
    auto yscaler = build_scaler(npz, "yscaler");

    return MLPModel(architecture, weights, biases, xscaler, yscaler);
}

#endif
