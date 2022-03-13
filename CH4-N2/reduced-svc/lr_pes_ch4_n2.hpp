#pragma once

#include <math.h>
#include <Eigen/Dense>

class LR_PES_ch4_n2
{
public:
    void init();
    double pes(const double R, const double th1, const double ph1, const double th2, const double ph2);
    
private:
    Eigen::Matrix3d S;
    void transform_angles(double ph, double th, double & phrot, double & throt);

    double pes_rotated(const double R, const double ph1, const double th1, const double ph2, const double th2);
    Eigen::Matrix3d fillS(const double alpha, const double beta, const double gamma);
    Eigen::Matrix3d fillSz(const double angle);
    Eigen::Matrix3d fillSy(const double angle);
};
