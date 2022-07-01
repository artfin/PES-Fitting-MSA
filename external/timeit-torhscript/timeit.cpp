#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#include "load_model.hpp"
#include "ai_pes_ch4_n2_opt1.hpp"

void timeit_symmpes()
{
    AI_PES_ch4_n2 symmpes;
    symmpes.init();

    std::random_device rd;
    std::mt19937 mt(rd()); 
    std::uniform_real_distribution<> dist(0, 1); 

    size_t ncycles = 1000000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;
    
    for (size_t k = 0; k < ncycles; ++k) {
        double R = 5.0 + dist(mt) * 25.0;
        double phi1 = 2.0 * M_PI * dist(mt);
        double theta1 = M_PI * dist(mt);
        double phi2 = 2.0 * M_PI * dist(mt);
        double theta2 = M_PI * dist(mt);

        start = std::chrono::high_resolution_clock::now();
        double out = symmpes.pes(R, phi1, theta1, phi2, theta2); 
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
   
    std::cout << " --- Symmetry-adapted PES --- " << std::endl; 
    std::cout << "Elapsed nanoseconds:   " << elapsed << std::endl;
    std::cout << "Nanoseconds per cycle: " << elapsed / ncycles << std::endl;
}

void timeit_torchscript()
{
    const size_t NATOMS = 7;
    const std::string torchscript_filename = "../../models/rigid/best-model/silu-torchscript.pt"; 
    NNPIP nn_pes(NATOMS, torchscript_filename);
    
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

    size_t ncycles = 1000000;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end; 
    double elapsed = 0.0;
    
    for (size_t k = 0; k < ncycles; ++k) {
        cc[12] = 3.0 + dist(mt); cc[13] = 3.0 + dist(mt); cc[14] = 3.0 + dist(mt);
        cc[15] = 3.0 + dist(mt); cc[16] = 3.0 + dist(mt); cc[17] = 3.0 + dist(mt);

        start = std::chrono::high_resolution_clock::now();
        double out = nn_pes.pes(cc);
        end = std::chrono::high_resolution_clock::now();

        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
   
    std::cout << " --- (TorchScript) PES --- " << std::endl; 
    std::cout << "Elapsed nanoseconds:   " << elapsed << std::endl;
    std::cout << "Nanoseconds per cycle: " << elapsed / ncycles << std::endl;
}

int main()
{
    timeit_symmpes();
    timeit_torchscript();

    return 0;
}
