#ifndef MLP_H
#define MLP_H 

#include <vector>
#include <Eigen/Dense>

// https://github.com/rogersce/cnpy
#include "cnpy.h"

void eig_flatten_rowvectorxds(std::vector<Eigen::RowVectorXd> const& qtrs, Eigen::Ref<Eigen::RowVectorXd> qtr) {

    size_t offset = 0;
    for (auto const& qq : qtrs) {
        qtr.middleCols(offset, qq.size()) = qq;
        offset += qq.size();
    }
}

double SiLU(double x) {
    return x / (1.0 + std::exp(-x)); 
}

struct MLP
{
    MLP(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases);
    Eigen::RowVectorXd forward(Eigen::RowVectorXd ptr);

    std::vector<size_t> architecture;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::RowVectorXd> biases;
    std::vector<Eigen::RowVectorXd> neurons;
    std::vector<Eigen::RowVectorXd> cache_neurons;
    
    Eigen::RowVectorXd p; 
    Eigen::RowVectorXd ptr; 

    std::vector<double> yij; 
};

MLP::MLP(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases) :
   architecture(architecture), weights(weights), biases(biases)
{
    size_t sz = architecture.size();
    neurons.resize(sz);
    cache_neurons.resize(sz);

    for (size_t k = 0; k < sz; ++k) {
        neurons[k] = Eigen::RowVectorXd::Zero(architecture[k]);
        cache_neurons[k] = Eigen::RowVectorXd::Zero(architecture[k]);
    }
    
    p   = Eigen::RowVectorXd::Zero(architecture[0]);
    ptr = Eigen::RowVectorXd::Zero(architecture[0]);

    yij.resize(ndist); 
}

Eigen::RowVectorXd MLP::forward(Eigen::RowVectorXd ptr)
/*
 * input  : atomic coordinates [a0]
 */
{
    size_t sz = architecture.size();
    size_t hidden_dims = sz - 2;
   
    // in_features 
    neurons[0] = ptr;

    for (size_t i = 1; i <= hidden_dims; ++i) {
        cache_neurons[i] = neurons[i - 1] * weights[i - 1];
        cache_neurons[i].noalias() += biases[i - 1];
        
        for (size_t j = 0; j < neurons[i].size(); ++j) {
            neurons[i](j) = SiLU(cache_neurons[i](j));
        }
    }

    // last linear operation -> out_features [no activation]
    neurons[sz - 1] = neurons[sz - 2] * weights[sz - 2];
    neurons[sz - 1] += biases[sz - 2];

    return neurons[sz - 1];
}

size_t count_subnets(cnpy::npz_t const& npz) {
    size_t counter = 0;
    for (auto it = npz.begin(); it != npz.end(); ++it) {
        std::string name = it->first;
        size_t found = name.find("architecture");
        if (found != std::string::npos) counter++; 
    }

    return counter;
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

struct QModel 
{
    QModel() = default;

    void init(std::string const& npz_fname);
    Eigen::RowVectorXd forward(std::vector<double> const& x);

    std::unique_ptr<StandardScaler> xscaler;
    std::unique_ptr<StandardScaler> yscaler;
    std::vector<MLP> modules;

    std::vector<size_t> npolys;
    size_t total_npoly;

    Eigen::RowVectorXd p;
    std::vector<Eigen::RowVectorXd> qtrs;
    Eigen::RowVectorXd qtr;
};

void QModel::init(std::string const& npz_fname) 
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    cnpy::npz_t npz = cnpy::npz_load(npz_fname);
    
    xscaler = build_scaler(npz, "xscaler");
    yscaler = build_scaler(npz, "yscaler");

    size_t n_subnetworks = count_subnets(npz);
    std::cout << "  Located n_subnetworks=" << n_subnetworks << "\n";

    for (size_t i = 0; i < n_subnetworks; ++i) {
        auto arch = parse_architecture_entry(npz, "architecture." + std::to_string(i));
        print("architecture." + std::to_string(i), arch);

        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::RowVectorXd> biases;

        for (size_t k = 0; k < arch.size() - 1; ++k) {
            auto w = parse_matrixxd_entry(npz, "blocks." + std::to_string(i) + "." + std::to_string(2 * k) + ".weight");
            auto b = parse_rowvectorxd_entry(npz, "blocks." + std::to_string(i) + "." + std::to_string(2 * k) + ".bias");

            std::cout << "Parsed layer (" << k << ") of size: " << w.rows() << " x " << w.cols() << std::endl;

            weights.push_back(w);
            biases.push_back(b);
        }

        auto m = MLP(arch, weights, biases);
        modules.push_back(m);
    }

    total_npoly = 0;

    size_t n_modules = modules.size();
    npolys.reserve(n_modules);
    size_t total_qs = 0;

    for (size_t k = 0; k < n_modules; ++k) {
        npolys[k] = modules[k].architecture[0];
        total_npoly += npolys[k];
        total_qs += modules[k].architecture.back();
    }
   
    p = Eigen::RowVectorXd::Zero(total_npoly);
    
    qtrs.resize(n_modules);
    qtr = Eigen::RowVectorXd::Zero(total_qs);
}

Eigen::RowVectorXd QModel::forward(std::vector<double> const& x) {

    FILL_POLY(npolys, x, p);
    Eigen::RowVectorXd ptr = xscaler->transform(p);

    size_t offset = 0;
    for (size_t k = 0; k < modules.size(); ++k) {
        qtrs[k] = modules[k].forward(ptr.segment(offset, npolys[k]));
        offset += npolys[k];
    }
    
    eig_flatten_rowvectorxds(qtrs, qtr); 

    return qtr;
}


//QModel build_qmodel_from_npz(std::string const& npz_fname)
//{
//    // use mutex to make it thread-safe
//    static std::mutex mutex;
//    std::lock_guard<std::mutex> lock(mutex);
//
//    cnpy::npz_t npz = cnpy::npz_load(npz_fname);
//    
//    auto xscaler = build_scaler(npz, "xscaler");
//    auto yscaler = build_scaler(npz, "yscaler");
//
//    size_t n_subnetworks = count_subnets(npz);
//    std::cout << "  Located n_subnetworks=" << n_subnetworks << "\n";
//
//    std::vector<MLP> modules;
//
//    for (size_t i = 0; i < n_subnetworks; ++i) {
//        auto arch = parse_architecture_entry(npz, "architecture." + std::to_string(i));
//        print("architecture." + std::to_string(i), arch);
//
//        std::vector<Eigen::MatrixXd> weights;
//        std::vector<Eigen::RowVectorXd> biases;
//
//        for (size_t k = 0; k < arch.size() - 1; ++k) {
//            auto w = parse_weights_entry(npz, "blocks." + std::to_string(i) + "." + std::to_string(2 * k) + ".weight");
//            auto b = parse_bias_entry(npz, "blocks." + std::to_string(i) + "." + std::to_string(2 * k) + ".bias");
//            weights.push_back(w);
//            biases.push_back(b);
//
//            std::cout << "Parsed layer (" << k << ") of size: " << w.rows() << " x " << w.cols() << std::endl;
//        }
//
//        auto m = MLP(arch, weights, biases);
//        modules.push_back(m);
//    }
//
//    return QModel(xscaler, yscaler, modules); 
//}

#endif
