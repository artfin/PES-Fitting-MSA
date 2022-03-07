#include <iostream>
#include <iomanip>

#include "lr_pes_ch4_n2.hpp"
	
const double HTOCM = 2.194746313702e5;

int main()
{
    LR_PES_ch4_n2 pes;
    pes.init();

    double en = pes.pes(30.0, 0.7853981633974483, 0.9553166181240510, 0.0000000000000000, 0.0000000000000000) * HTOCM;
    std::cout << std::fixed << std::setprecision(16);

    std::cout << "expected:   " << -0.0295539482050078 << "\n";
    std::cout << "calculated: " << en << "\n";

    return 0;
}
