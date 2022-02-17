import torch
import torch.nn as nn

import numpy as np
import sklearn
from scipy import stats

def chi_split(X, y, test_size=0.2, random_state=None, nbins=10):
    full_dist, binedges = np.histogram(y, bins=nbins, density=True)

    pvalues, chi = [], []
    for seed in range(100):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
        train_dist, train_bin = np.histogram(y_train, bins=binedges, density=True)
        chisq, p = stats.chisquare(train_dist, f_exp=full_dist)
        chi.append(chisq)
        pvalues.append(p)

    best_seed = np.argmin(chi)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=best_seed)
    return X_train, X_test, y_train, y_test

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, ypred, y):
        return torch.sqrt(self.mse(ypred, y))


class StandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

        # detect the constant features (zero standard deviation)
        self.zero_idx = (self.std == 0).nonzero()

    def transform(self, x, mean=None, std=None):
        c = torch.clone(x)

        if mean is not None:
            c -= mean
        else:
            c -= self.mean

        if std is not None:
            c /= std
        else:
            c /= self.std

        # -> transform those features to zero 
        c[:, self.zero_idx] = 0.0
        return c

    @classmethod
    def from_precomputed(cls, mean, std, zero_idx):
        scaler = cls()
        scaler.mean     = mean
        scaler.std      = std
        scaler.zero_idx = zero_idx
        return scaler

class IdentityScaler:
    def fit(self, x):
        pass

    def transform(self, x):
        return x


