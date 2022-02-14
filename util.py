import torch
import torch.nn as nn

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

class IdentityScaler:
    def fit(self, x):
        pass

    def transform(self, x):
        return x


