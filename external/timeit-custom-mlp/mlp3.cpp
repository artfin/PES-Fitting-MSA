double** Evaluator::alloc_2d(int nrows, int ncols) {
    double** a = new double* [nrows];
    for(int i = 0; i < nrows; ++i) {
        a[i] = new double [ncols];
    }

    return a;
}

void Evaluator::delete_2d(double** a, int nrows, int ncols) {

    for (int i = 0; i < nrows; ++i) {
        delete[] a[i];
    }
    
    delete a;
}

void Evaluator::print_2d(double** a, int nrows, int ncols) {
    
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            std::cout << a[i][j] << " ";
        }
        std::cout << "\n";
    }
}

struct MultiLayerPerceptron3 {
    MultiLayerPerceptron3(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases);
    Eigen::RowVectorXd propagateForward(Eigen::RowVectorXd input);
    Eigen::RowVectorXd backward();

    Eigen::MatrixXd W1, W2;
    Eigen::RowVectorXd bias1, bias2;
    Eigen::RowVectorXd l1, l2; // hidden and output layers
    Eigen::RowVectorXd cache_l1; // hidden layer before activation
};

MultiLayerPerceptron3::MultiLayerPerceptron3(std::vector<size_t> const& architecture, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::RowVectorXd> const& biases) {
    size_t sz = architecture.size();
    assert(sz == 3);

    assert(weights.size() == 2);
    W1 = weights[0];
    W2 = weights[1];

    assert(biases.size() == 2);
    bias1 = biases[0];
    bias2 = biases[1];
   
    l1 = Eigen::RowVectorXd::Zero(architecture[1]);
    l2 = Eigen::RowVectorXd::Zero(architecture[2]);

    cache_l1 = Eigen::RowVectorXd::Zero(architecture[1]);
}

Eigen::RowVectorXd MultiLayerPerceptron3::propagateForward(Eigen::RowVectorXd input) {
    cache_l1.noalias() = input * W1;
    cache_l1 += bias1;
    l1 = cache_l1.unaryExpr(&SiLU);

    l2.noalias() = l1 * W2;
    l2 += bias2;

    return l2;
}

Eigen::RowVectorXd MultiLayerPerceptron3::backward() {
    cache_l1 = cache_l1.unaryExpr(&dSiLU);
    return (W1 * cache_l1.asDiagonal() * W2).transpose(); 
}

