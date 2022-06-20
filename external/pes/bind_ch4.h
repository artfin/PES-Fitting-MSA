#ifndef BIND_CH4_H
#define BIND_CH4_H

extern "C" {
    void potinit();
    
    // returns CH4 PES [cm-1]
    // `poten_xy4` accepts a vector of internal coordinates {r1, r2, r3, r4, alpha12, alpha13, alpha14, alpha23, alpha24, alpha34}
    // where r is the C-H distance (Angstrom) and alpha is planar angle (radian). 
    void poten_xy4(double *q, double* res);
}

void xyz_to_internal_CH4(double x[15], double q[10])
/* 
 * x: 
 *    xC  = x[0]  yC  = x[1]  zC  = x[2]
 *    xH1 = x[3]  yH1 = x[4]  zH1 = x[5]
 *    xH2 = x[6]  yH2 = x[7]  zH2 = x[8]
 *    xH3 = x[9]  yH3 = x[10] zH3 = x[11]
 *    xH4 = x[12] yH4 = x[13] zH4 = x[14]
 * q: [0] r1 -- d(C-H1) 
 *    [1] r2 -- d(C-H2)
 *    [2] r3 -- d(C-H3)
 *    [3] r4 -- d(C-H4)
 *    [4] alpha12 -- ang(H1-C-H2)
 *    [5] alpha13 -- ang(H1-C-H3)
 *    [6] alpha14 -- ang(H1-C-H4)
 *    [7] alpha23 -- ang(H2-C-H3)
 *    [8] alpha24 -- ang(H2-C-H4)
 *    [9] alpha34 -- ang(H3-C-H4)
 */
{
    double r1 = sqrt((x[3]-x[0])*(x[3]-x[0]) + (x[4]-x[1])*(x[4]-x[1]) + (x[5]-x[2])*(x[5]-x[2]));
    double r2 = sqrt((x[6]-x[0])*(x[6]-x[0]) + (x[7]-x[1])*(x[7]-x[1]) + (x[8]-x[2])*(x[8]-x[2]));
    double r3 = sqrt((x[9]-x[0])*(x[9]-x[0]) + (x[10]-x[1])*(x[10]-x[1]) + (x[11]-x[2])*(x[11]-x[2]));
    double r4 = sqrt((x[12]-x[0])*(x[12]-x[0]) + (x[13]-x[1])*(x[13]-x[1]) + (x[14]-x[2])*(x[14]-x[2]));

    double cosq12 = ((x[3] - x[0]) * (x[6] - x[0]) + (x[4] - x[1]) * (x[7] - x[1]) + (x[5] - x[2]) * (x[8] - x[2])) / (r1 * r2);
    double cosq13 = ((x[3] - x[0]) * (x[9] - x[0]) + (x[4] - x[1]) * (x[10] - x[1]) + (x[5] - x[2]) * (x[11] - x[2])) / (r1 * r3);
    double cosq14 = ((x[3] - x[0]) * (x[12] - x[0]) + (x[4] - x[1]) * (x[13] - x[1]) + (x[5] - x[2]) * (x[14] - x[2])) / (r1 * r4);
    double cosq23 = ((x[6] - x[0]) * (x[9] - x[0]) + (x[7] - x[1]) * (x[10] - x[1]) + (x[8] - x[2]) * (x[11] - x[2])) / (r2 * r3);
    double cosq24 = ((x[6] - x[0]) * (x[12] - x[0]) + (x[7] - x[1]) * (x[13] - x[1]) + (x[8] - x[2]) * (x[14] - x[2])) / (r2 * r4);
    double cosq34 = ((x[9] - x[0]) * (x[12] - x[0]) + (x[10] - x[1]) * (x[13] - x[1]) + (x[11] - x[2]) * (x[14] - x[2])) / (r3 * r4);

    q[0] = r1;
    q[1] = r2;
    q[2] = r3;
    q[3] = r4;
    q[4] = acos(cosq12);
    q[5] = acos(cosq13);
    q[6] = acos(cosq14);
    q[7] = acos(cosq23);
    q[8] = acos(cosq24);
    q[9] = acos(cosq34);
}

double pot_CH4(double x[15]) 
/*
 * x: Angstrom -> energy: cm-1
 */
{
    static double q[10];
    xyz_to_internal_CH4(x, q);

    double V;
    poten_xy4(q, &V);

    return V;
}

#endif
