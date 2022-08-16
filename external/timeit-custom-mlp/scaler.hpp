#ifndef SCALER_H
#define SCALER_H 

#include <Eigen/Dense>

struct StandardScaler {
    StandardScaler(Eigen::RowVectorXd const& mean, Eigen::RowVectorXd const& scale);
    
    Eigen::RowVectorXd transform(Eigen::RowVectorXd const& x);
    Eigen::RowVectorXd inverse_transform(Eigen::RowVectorXd const& xtr);

    Eigen::RowVectorXd mean;
    Eigen::RowVectorXd scale;
};

#endif
