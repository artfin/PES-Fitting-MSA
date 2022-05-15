#include <iostream>
#include <iomanip>

#include "load_model.hpp"
#include "ai_pes_ch4_n2_opt1.hpp"

//const double NN_BOND_LENGTH = 2.078; // a0
const double NN_BOND_LENGTH = 2.15; // a0

template <typename T>
std::vector<T> linspace(const T start, const T end, const size_t size) {

    const T step = (end - start) / (size - 1);

    std::vector<T> v(size);
    for (size_t k = 0; k < size; ++k) {
        v[k] = start + step * k;
    }

    return v;
}

double internal_pes(NNPIP & pes, double R, double PH1, double TH1, double PH2, double TH2)
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

    std::cout << "N " << cart[12] << " " << cart[13] << " " << cart[14] << "\n";
    std::cout << "N " << cart[15] << " " << cart[16] << " " << cart[17] << "\n";

    return pes.pes(cart);
}

int main()
{
    std::cout << std::fixed << std::setprecision(16);

    AI_PES_ch4_n2 symm_pes;
    symm_pes.init();
    
    const size_t natoms = 7;
    const std::string model_path = "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/models/nonrigid/L1/L1-nonrigid-only/torchscript-model.pt";
    NNPIP nn_pes(natoms, model_path);

    //std::vector<double> c = {
    //     1.4007532260, -1.3132036250,  0.7770669080, 
    //     0.6956926500,  1.3539814810, -1.3716276090,
    //    -1.4136072120, -0.9910356850, -1.1154785290,
    //    -0.9236609640,  1.1264660110,  1.4631193370,
    //     7.0393772810,  4.2537622620,  1.9110195630,
    //     7.9707203340,  3.1646132690,  0.4224439140,
    //     0.0000000000,  0.0000000000,  0.0000000000,
    //};
    //std::cout << "Expected: -69.9754756053000051\n";
    
    std::vector<double> c = {
         1.1935874160000000,  1.1935874160000000, -1.1935874160000000, 
        -1.1935874160000000, -1.1935874160000000, -1.1935874160000000,
        -1.1935874160000000,  1.1935874160000000,  1.1935874160000000,
         1.1935874160000000, -1.1935874160000000,  1.1935874160000000,
         2.6988162320077000,  0.7573006066205795, -5.5380999470061214,
         4.3468022908240336,  2.0191594443482823, -5.6377905864638782,
         0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
    };

    std::cout << nn_pes.pes(c) << "\n";
    std::cout << "Expected: 201.3350805254999898\n";

    const double deg = M_PI / 180.0;
    double PH1 = 47.912 * deg;
    double TH1 = 56.167 * deg;
    double PH2 = 0.0    * deg;
    double TH2 = 135.0  * deg;

    //std::vector<double> Rv = linspace(4.5, 40.0, 1000);
    std::vector<double> Rv = {1000.0};

    // R = 1000a0
    const double INFVAL = -0.2314792876363541;

    for (double R : Rv) {
        double nnval   = internal_pes(nn_pes, R, PH1, TH1, PH2, TH2) - INFVAL;
        double symmval = symm_pes.pes(R, PH1, TH1, PH2, TH2);
        std::cout << R << " " << nnval << " " << symmval << "\n";
    }

    return 0;
}
