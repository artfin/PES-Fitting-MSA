import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class L1Regularization(torch.nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = torch.tensor(lambda_).to(DEVICE)

    def __repr__(self):
        return "L1Regularization(lambda={})".format(self.lambda_.item())

    def forward(self, model):
        l1_norm = torch.tensor(0.).to(dtype=torch.float64, device=DEVICE)
        for p in model.parameters():
            l1_norm += p.abs().sum()

        return self.lambda_ * l1_norm

class L2Regularization(torch.nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = torch.tensor(lambda_).to(DEVICE)

    def __repr__(self):
        return "L2Regularization(lambda={})".format(self.lambda_.item())

    def forward(self, model):
        l2_norm = torch.tensor(0.).to(DEVICE)
        for p in model.parameters():
            l2_norm += (p**2).sum()
        return self.lambda_ * l2_norm
