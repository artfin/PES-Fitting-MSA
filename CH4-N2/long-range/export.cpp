#include <iostream>
#include <iomanip>
#include <fstream>

#include "lr_pes_ch4_n2.hpp"
	
const double HTOCM = 2.194746313702e5;

int main()
{
    std::cout << std::fixed << std::setprecision(16);
    
    LR_PES_ch4_n2 pes;
    pes.init();

    std::ifstream ifs("cache_coords.dat");
    std::ofstream ofs("cache_lr.dat");
    ofs << std::fixed << std::setprecision(16);

    double R, ph1, th1, ph2, th2;

    for (std::string line; std::getline(ifs, line); ) {
        std::stringstream ss;
        ss << line;
        ss >> R >> ph1 >> th1 >> ph2 >> th2;

        double C = 15.0; // sigmoid center of symmetry
        double S = 10.0; // the speed of `turning on` the long-range model 
        double W = 1.0 / (1.0 + std::exp(-S * (R - C)));
        double en = pes.pes(R, ph1, th1, ph2, th2) * W;

        ofs << R << " " << ph1 << " " << th1 << " " << ph2 << " " << th2 << " " << en << "\n"; 
    }

    return 0;
}
