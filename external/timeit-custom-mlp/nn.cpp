#include <iostream>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>

#include <Eigen/Dense>

// https://github.com/rogersce/cnpy
#include "cnpy.h"

const double a0 = 2.0;
// my C implementation [order=4; NPOLY=79]
extern "C" {
    void evpoly(double x[], double p[]);
}

template <typename Container>
void print(std::string const& name, Container c) {
    std::cout << "[" << name << "]: ";
    for (auto el : c) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
}

// ================================================================================
struct StandardScaler
{
public:
    StandardScaler(Eigen::RowVectorXd const& mean, Eigen::RowVectorXd const& scale);
    
    Eigen::RowVectorXd transform(Eigen::RowVectorXd const& x);
    Eigen::RowVectorXd inverse_transform(Eigen::RowVectorXd const& xtr);

private:
    Eigen::RowVectorXd m_mean;
    Eigen::RowVectorXd m_scale;
};

StandardScaler::StandardScaler(Eigen::RowVectorXd const& mean, Eigen::RowVectorXd const& scale) :
    m_mean(mean), m_scale(scale) 
{
}

Eigen::RowVectorXd StandardScaler::transform(Eigen::RowVectorXd const& x) 
{
    // element-wise division
    return (x - m_mean).array() / m_scale.array();
}

Eigen::RowVectorXd StandardScaler::inverse_transform(Eigen::RowVectorXd const& xtr) {
    return m_scale.cwiseProduct(xtr) + m_mean;
}
// ================================================================================

// ================================================================================
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorXd;

class MultiLayerPerceptron
{
public:
    MultiLayerPerceptron(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases);

    Eigen::RowVectorXd propagateForward(Eigen::RowVectorXd input);

private:
    std::vector<size_t> m_architecture;
    std::vector<MatrixXd> m_weights;
    std::vector<RowVectorXd> m_biases;
    std::vector<RowVectorXd> m_neurons;
};

MultiLayerPerceptron::MultiLayerPerceptron(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases) :
   m_architecture(architecture), m_weights(weights), m_biases(biases)

{
    m_neurons.resize(architecture.size());
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double SiLU(double x) {
    return x * sigmoid(x);
}
 

Eigen::RowVectorXd MultiLayerPerceptron::propagateForward(Eigen::RowVectorXd input)
{
    //std::cout << std::fixed << std::setprecision(10);
    //std::cout << "input:\n" << input << "\n";
    //std::cout << "input.size: " << input.rows() << "x" << input.cols() << std::endl;

    size_t sz = m_architecture.size();
    size_t hidden_dims = sz - 2;
   
    // in_features 
    m_neurons[0] = input;

    for (size_t i = 1; i <= hidden_dims; ++i) {
        m_neurons[i] = m_neurons[i - 1] * m_weights[i - 1];
        m_neurons[i].noalias() += m_biases[i - 1];
        
        for (size_t j = 0; j < m_neurons[i].size(); ++j) {
            m_neurons[i](j) = SiLU(m_neurons[i](j));
        }

        //std::cout << "i=" << i << ": weights: " << m_weights[i - 1].rows() << "x" << m_weights[i - 1].cols()
        //          << "; neurons: " << m_neurons[i].cols() << "\n";
        //std::cout << m_neurons[i] << std::endl;
    }

    // last linear operation -> out_features [no activation]
    m_neurons[sz - 1] = m_neurons[sz - 2] * m_weights[sz - 2];
    m_neurons[sz - 1] += m_biases[sz - 2];

    return m_neurons[sz - 1];
}
// ================================================================================

// ================================================================================
struct Evaluator
{
public:
    Evaluator(MultiLayerPerceptron model, StandardScaler xscaler, StandardScaler yscaler);
    ~Evaluator();

    double forward(std::vector<double> const& x); 

private:
    MultiLayerPerceptron m_model;
    StandardScaler m_xscaler;
    StandardScaler m_yscaler;

    double* yij;
    double* poly;
};

Evaluator::Evaluator(MultiLayerPerceptron model, StandardScaler xscaler, StandardScaler yscaler) :
    m_model(model), m_xscaler(xscaler), m_yscaler(yscaler)
{

    yij = new double [21];
    poly = new double [79];
}

Evaluator::~Evaluator()
{
    delete yij;
    delete poly;
}

#define ATOMX(x, i) x[3*i]
#define ATOMY(x, i) x[3*i + 1]
#define ATOMZ(x, i) x[3*i + 2]

double Evaluator::forward(std::vector<double> const& x) 
/*
 * input: atomic coordinates [a0] 
 */
{
    const size_t NATOMS = 7;

    double drx, dry, drz;
   
    size_t k = 0;
    for (size_t i = 0; i < NATOMS; ++i) {
        for (size_t j = i + 1; j < NATOMS; ++j) {
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

            yij[k] = std::sqrt(drx*drx + dry*dry + drz*drz);
            yij[k] = std::exp(-yij[k]/a0);
            k++;
        }
    }
     
    evpoly(yij, poly);
    
    auto xp = Eigen::RowVectorXd(79);
    xp = Eigen::Map<const Eigen::RowVectorXd>(poly, 79);

    //auto xp = Eigen::RowVectorXd::Zero(1, 79);

    Eigen::RowVectorXd m_xtr = m_xscaler.transform(xp);
    Eigen::RowVectorXd m_ytr = m_model.propagateForward(m_xtr);
    Eigen::RowVectorXd res   = m_yscaler.inverse_transform(m_ytr);

    return res(0);
}
// ================================================================================

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

Evaluator build_evaluator_from_npz(std::string const& npz_fname)
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

    auto mlp = MultiLayerPerceptron(architecture, weights, biases);

    auto xscaler = build_scaler(npz, "xscaler");
    auto yscaler = build_scaler(npz, "yscaler");

    return Evaluator(mlp, xscaler, yscaler);
}

void timeit()
{
    auto evaluator = build_evaluator_from_npz("silu.npz");  
    
    std::vector<double> cc = {
 	  1.1935874160,	  1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	 -1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	  1.1935874160, 	 1.1935874160,
 	  1.1935874160,	 -1.1935874160, 	 1.1935874160,
 	  2.5980762114,   2.5980762114, 	 1.5590762114,
 	  2.5980762114,	  2.5980762114, 	 3.6370762114,
 	  0.0000000000,	  0.0000000000, 	 0.0000000000,
    };
  
    std::random_device rd;
    std::mt19937 mt(rd()); 
    std::uniform_real_distribution<> dist(0, 1); 
    
    size_t ncycles = 10000000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;

    for (size_t k = 0; k < ncycles; ++k) {
        cc[12] = 3.0 + dist(mt); cc[13] = 3.0 + dist(mt); cc[14] = 3.0 + dist(mt);
        cc[15] = 3.0 + dist(mt); cc[16] = 3.0 + dist(mt); cc[17] = 3.0 + dist(mt);

        start = std::chrono::high_resolution_clock::now();
        double out = evaluator.forward(cc);
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    
    std::cout << "Elapsed nanoseconds:   " << elapsed << std::endl;
    std::cout << "Nanoseconds per cycle: " << elapsed / ncycles << std::endl;
}

int main2()
{
    timeit();

    return 0;
}

int main()
{
    auto evaluator = build_evaluator_from_npz("silu.npz");  

    std::vector<double> cc = {
 	  1.1935874160,	  1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	 -1.1935874160, 	 -1.1935874160,
 	 -1.1935874160,	  1.1935874160, 	 1.1935874160,
 	  1.1935874160,	 -1.1935874160, 	 1.1935874160,
 	  2.5980762114,   2.5980762114, 	 1.5590762114,
 	  2.5980762114,	  2.5980762114, 	 3.6370762114,
 	  0.0000000000,	  0.0000000000, 	 0.0000000000,
    };
    
    double out = evaluator.forward(cc);

    std::cout << std::fixed << std::setprecision(16);
    std::cout << "out: " << out << std::endl;

    return 0;
}
