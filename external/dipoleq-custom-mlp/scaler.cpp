#include "scaler.hpp"

StandardScaler::StandardScaler(Eigen::RowVectorXd const& mean, Eigen::RowVectorXd const& scale) :
    mean(mean), scale(scale) 
{
}

Eigen::RowVectorXd StandardScaler::transform(Eigen::RowVectorXd const& x) {
    // element-wise division
    return (x - mean).array() / scale.array();
}

Eigen::RowVectorXd StandardScaler::inverse_transform(Eigen::RowVectorXd const& xtr) {
    return scale.cwiseProduct(xtr) + mean;
}
