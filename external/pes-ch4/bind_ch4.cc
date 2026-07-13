#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
    
#include "bind_ch4.h"

const double BohrToAng = 0.529177;
const double HTOCM = 219474.6;

int main()
{
    potinit();

    /*
    double x[15] = {
        0.0000000000,  0.0000000000,  0.0000000000, 
        1.0860100000,  0.0000000000,  0.0000000000,
        -0.3620033333,  0.0000000000, -1.1239000473,
        -0.3620033333, -0.8867234520,  0.5119500235,
        -0.3620033333,  0.8867234517,  0.5119500240,
    };
    double E_eq = -40.4408920148;
    double E = -40.4365415983;
    */

    /*
    double x[15] = {
         0.172300,  0.137248,  0.260482,
         0.712469,  0.071162, -0.659133,
        -0.457260, -0.611138, -0.178166,
        -0.636172,  0.872560,  0.096898,
         0.495166, -0.295482,  1.203970,
    };
    double E_eq = -40.4408920148;
    double E = -40.4052519243;
    */

    double x[15] = {
        -0.114595000,  -0.227713000,   0.358060000,
         0.818453000,   0.097828000,  -0.135071000,
        -0.345644000,  -0.648784000,  -0.625514000,
        -0.515377000,   0.731756000,   0.022303000,
        -0.050244000,  -0.644653000,   1.394970000,
    };

    double q[10];
    xyz_to_internal_CH4(x, q);

    printf("[XYZ_TO_INTERNAL] q:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%.10lf\n", q[i]);
    }
 
   /* 
    xyz_to_internal2(x, q);
    printf("[XYZ_TO_INTERNAL2] q:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%.10lf\n", q[i]);
    }
    */

    double V;
    poten_xy4(q, &V);

    printf("V: %.10lf\n", V);
    
    char symbol[5] = {'C', 'H', 'H', 'H', 'H'};

    for (int i = 0; i < 15; i += 3) {
        printf("%c,, %.10lf, %.10lf, %.10lf\n", symbol[i / 3], x[i] / BohrToAng, x[i + 1] / BohrToAng, x[i + 2] / BohrToAng);
    }
    
    double E_eq = -40.4408920148;
    double E = -40.3937783629;

    double V_CCSD = (E - E_eq) * HTOCM;
    printf("V_CCSD: %.10lf\n", V_CCSD);

    return 0;
}
