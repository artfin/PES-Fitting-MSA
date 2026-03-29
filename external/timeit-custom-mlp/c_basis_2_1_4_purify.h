#ifndef C_BASIS_2_1_4_PURIFY_H_
#define C_BASIS_2_1_4_PURIFY_H_

#include <Eigen/Dense>

extern "C" void evpoly_2_1_4_purify(double* y, Eigen::Ref<Eigen::RowVectorXd> p);

#endif // C_BASIS_2_1_4_PURIFY_H_
