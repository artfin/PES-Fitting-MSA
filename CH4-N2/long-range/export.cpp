#include <iostream>
#include <iomanip>

#include "lr_pes_ch4_n2.hpp"
	
const double HTOCM = 2.194746313702e5;

int main()
{
    std::cout << std::fixed << std::setprecision(16);
    
    LR_PES_ch4_n2 pes;
    pes.init();

    double R, ph1, th1, ph2, th2;
    std::cin >> R >> ph1 >> th1 >> ph2 >> th2;

    double en = pes.pes(R, ph1, th1, ph2, th2);
    std::cout << en << "\n";

    return 0;
}
