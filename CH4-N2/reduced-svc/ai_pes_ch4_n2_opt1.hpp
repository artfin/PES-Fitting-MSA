#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#include <gsl/gsl_sf_legendre.h>

class AI_PES_ch4_n2
{
public:
    void init();
	static constexpr size_t index(const int l, const int m) {
        return l * (l + 1) / 2 + m;
    }

    static void fill_legP1(const double theta1);
    static void fill_legP2(const double theta2);

    static double pes(double R, double phi1, double theta1, double phi2, double theta2);
    static double dpesdR(double R, double phi1, double theta1, double phi2, double theta2);
    static double dpesdphi1(double R, double phi1, double theta1, double phi2, double theta2);
    static double dpesdtheta1(double R, double phi1, double theta1, double phi2, double theta2);
    static double dpesdphi2(double R, double phi1, double theta1, double phi2, double theta2);
    static double dpesdtheta2(double R, double phi1, double theta1, double phi2, double theta2);

private:
    static std::vector<double> LegP1, LegP2;

    static const int lmax_bond = 13;
    static const int lmax_n2 = 12;

    static gsl_sf_legendre_t norm;
    static constexpr double csphase = -1; // to include Condon-Shortley phase
};
