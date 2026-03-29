#ifndef MLP_H_
#define MLP_H_

#include <cassert>
#include <memory>
#include <mutex>

#include <Eigen/Dense>

// https://github.com/rogersce/cnpy
#include "cnpy.h"

void EVPOLY(double *y, Eigen::Ref<Eigen::RowVectorXd> p);
void MAKE_YIJ(const double *x, double *y);
void EVPOLY_JAC(Eigen::Ref<Eigen::MatrixXd> jac, double *y);
void MAKE_DYDR(Eigen::Ref<Eigen::MatrixXd> dydr, const double *x); 

struct StandardScaler {
    StandardScaler(Eigen::RowVectorXd const& mean, Eigen::RowVectorXd const& scale);
    
    Eigen::RowVectorXd transform(Eigen::RowVectorXd const& x);
    Eigen::RowVectorXd inverse_transform(Eigen::RowVectorXd const& xtr);

    Eigen::RowVectorXd mean;
    Eigen::RowVectorXd scale;
};

struct MLPES
{
    MLPES() = default;
    ~MLPES();

    void init(std::string const& npz_fname, size_t natoms, bool log);

    double forward(double* x);
    void backward(const double *x, double *dx);

    void make_drdx(Eigen::Ref<Eigen::MatrixXd> drdx, const double *x);
    
    double** alloc_2d(int nrows, int ncols);
    void print_2d(double** a, int nrows, int ncols);
    void delete_2d(double** a, int nrows, int ncols);
   
    size_t natoms;
    size_t ndist; 
    double *yij; 
    bool INITIALIZED = false;

    std::unique_ptr<StandardScaler> xscaler;
    std::unique_ptr<StandardScaler> yscaler;
    
    std::vector<size_t> arch;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::RowVectorXd> biases;
    std::vector<Eigen::RowVectorXd> neurons;
    std::vector<Eigen::RowVectorXd> cache_neurons;
    
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
};

#endif // MLP_H_

#ifdef MLP_IMPLEMENTATION

std::unique_ptr<StandardScaler> build_scaler(cnpy::npz_t& npz, std::string const& scaler_type);
std::vector<size_t> parse_architecture_entry(cnpy::npz_t const& npz, std::string const& entry_name);
Eigen::RowVectorXd parse_rowvectorxd_entry(cnpy::npz_t const& npz, std::string const& entry_name);
Eigen::MatrixXd parse_matrixxd_entry(cnpy::npz_t const& npz, std::string const& entry_name);

StandardScaler::StandardScaler(Eigen::RowVectorXd const& mean, Eigen::RowVectorXd const& scale) : mean(mean), scale(scale) { }

Eigen::RowVectorXd StandardScaler::transform(Eigen::RowVectorXd const& x) 
{
    // element-wise division
    return (x - mean).array() / scale.array();
}

Eigen::RowVectorXd StandardScaler::inverse_transform(Eigen::RowVectorXd const& xtr) {
    return scale.cwiseProduct(xtr) + mean;
}

double SiLU(double x) {
    if (x < -700.0) return 0;
    double x_exp = std::exp(-x);

    double result = x / (1.0 + x_exp);
    return result; 
}

double dSiLU(double x) {
    if (x < -700.0) return 0;
    double x_exp = std::exp(-x);
    
    double denominator = 1.0 + x_exp;

    double result = 1.0 / denominator + x * x_exp / denominator / denominator;
    //if (std::isinf(result)) {
    //    std::cout << "(dSilu) x: " << x << " => " << result << "\n";
    //    std::cout << "denominator = " << denominator << "\n";
    //    std::cout << "x_exp: " << x_exp << "\n";
    //    assert(false);
    //}
    return result; 
}

template <typename Container>
void print(std::string const& name, Container c) {
    std::cout << "[" << name << "]: ";
    for (auto el : c) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

void MLPES::init(std::string const& npz_fname, size_t natoms, bool log=false)
{
    this->natoms = natoms;
    ndist = natoms * (natoms - 1) / 2;
    yij = new double [ndist];

    /*
     * use mutex to make it thread-safe
     */
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    cnpy::npz_t npz = cnpy::npz_load(npz_fname);
    
    xscaler = build_scaler(npz, "xscaler");
    yscaler = build_scaler(npz, "yscaler");
    
    arch = parse_architecture_entry(npz, "architecture");

    for (size_t k = 0; k < arch.size() - 1; ++k) {
        auto w = parse_matrixxd_entry(npz, std::to_string(2 * k) + ".weight");
        auto b = parse_rowvectorxd_entry(npz, std::to_string(2 * k) + ".bias");

        weights.push_back(w);
        biases.push_back(b);

        if (log) std::cout << "Parsed layer (" << k << ") of size: " << w.rows() << " x " << w.cols() << "\n";
    } 
    
    size_t sz = arch.size();
    neurons.resize(sz);
    cache_neurons.resize(sz);

    for (size_t k = 0; k < sz; ++k) {
        neurons[k]       = Eigen::RowVectorXd::Zero(arch[k]);
        cache_neurons[k] = Eigen::RowVectorXd::Zero(arch[k]);
    }

    size_t npoly = arch[0];
    p   = Eigen::RowVectorXd::Zero(npoly);
    ptr = Eigen::RowVectorXd::Zero(npoly);
    
    drdx = Eigen::MatrixXd::Zero(3 * natoms, ndist);
    dydr = Eigen::MatrixXd::Zero(ndist, ndist);
    
    jac  = Eigen::MatrixXd::Zero(ndist, npoly);
    dpdx = Eigen::MatrixXd::Zero(3 * natoms, npoly);
    
    dEdp = Eigen::RowVectorXd::Zero(npoly); 
    dEdx = Eigen::RowVectorXd::Zero(3 * natoms);

    t1.resize(npoly, arch[1]);
    t2.resize(npoly);
    t3.resize(3 * natoms, ndist); 
    
    INITIALIZED = true;
}

MLPES::~MLPES()
{
    delete [] yij;
}

double MLPES::forward(double *x)
/*
 * input  : atomic coordinates [a0]
 * output : energy [cm-1] 
 */
{
    assert(INITIALIZED && "ERROR: mlpes model is not initialized");

    MAKE_YIJ(x, yij);
    EVPOLY(yij, p);

    ptr = xscaler->transform(p);

    size_t sz = arch.size();
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
    
    return yscaler->inverse_transform(neurons[sz - 1])(0);
}

void MLPES::backward(const double *x, double *dx)
{
    make_drdx(drdx, x);
    MAKE_DYDR(dydr, x);

    // we already rely on `backward` being called RIGHT after corresponding `forward`
    /* MAKE_YIJ(x, yij); */ 
    EVPOLY_JAC(jac, yij);  
    
    t3.noalias()   = drdx * dydr;
    dpdx.noalias() = t3 * jac;

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

    //for (size_t i = 0; i < cache_neurons[1].size(); ++i) {
    //    double c = cache_neurons[1](i); 
    //    std::cout << i << " " << c << " " << std::isinf(c) << "\n"; 
    //} 

    dEdp = t2.transpose().array() / xscaler->scale.array(); // component-wise division

    dEdx = dEdp * dpdx.transpose();
    dEdx *= yscaler->scale[0];

    Eigen::VectorXd::Map(&dx[0], 3 * natoms) = dEdx; 
}

void MLPES::make_drdx(Eigen::Ref<Eigen::MatrixXd> drdx, const double* x) {
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

cnpy::NpyArray get_entry(cnpy::npz_t const& npz, std::string const& entry_name) {
    for (auto it = npz.begin(); it != npz.end(); ++it) {
        std::string _name = it->first;
        if (_name == entry_name) return it->second; 
    }

    throw std::runtime_error("get entry: could not find entry=" + entry_name); 
}

std::vector<size_t> parse_architecture_entry(cnpy::npz_t const& npz, std::string const& entry_name) {
    cnpy::NpyArray np_subnetwork_arch = get_entry(npz, entry_name);
    assert(np_subnetwork_arch.word_size == sizeof(size_t));

    const size_t* p = np_subnetwork_arch.data<size_t>();
    const size_t sz = np_subnetwork_arch.shape[0];

    return std::vector<size_t>{p, p + sz};
}

Eigen::MatrixXd parse_matrixxd_entry(cnpy::npz_t const& npz, std::string const& entry_name) {
    cnpy::NpyArray np_w = get_entry(npz, entry_name);
    assert(np_w.shape.size() == 2);

    const double* pw = np_w.data<double>();
    Eigen::MatrixXd w = Eigen::MatrixXd(np_w.shape[0], np_w.shape[1]);
    w = Eigen::Map<const Eigen::MatrixXd>(pw, np_w.shape[0], np_w.shape[1]);

    return w;
}

Eigen::RowVectorXd parse_rowvectorxd_entry(cnpy::npz_t const& npz, std::string const& entry_name) {
    cnpy::NpyArray np_b = get_entry(npz, entry_name);
    assert(np_b.shape.size() == 1);

    const double* pb = np_b.data<double>();
    Eigen::RowVectorXd b = Eigen::RowVectorXd(np_b.shape[0]);
    b = Eigen::Map<const Eigen::RowVectorXd>(pb, np_b.shape[0]);

    return b;
}

std::unique_ptr<StandardScaler> build_scaler(cnpy::npz_t& npz, std::string const& scaler_type) {

    assert(scaler_type == "xscaler" || scaler_type == "yscaler");

    auto mean = parse_rowvectorxd_entry(npz, scaler_type + ".mean");
    auto scale = parse_rowvectorxd_entry(npz, scaler_type + ".scale");

    return std::make_unique<StandardScaler>(mean, scale);
}

#endif // MLP_IMPLEMENTATION
