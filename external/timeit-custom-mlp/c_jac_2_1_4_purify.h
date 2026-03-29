#ifndef C_JAC_2_1_4_PURIFY_H_
#define C_JAC_2_1_4_PURIFY_H_

#include <Eigen/Dense>

extern "C" void evpoly_jac_2_1_4_purify(Eigen::Ref<Eigen::MatrixXd> jac, double* y);

#endif // C_JAC_2_1_4_PURIFY_H_
