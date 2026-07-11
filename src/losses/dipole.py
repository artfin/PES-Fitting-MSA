import torch

from config import DEVICE
from distributed import reduce_min


class WRMSELoss_Ratio_dipole(torch.nn.Module):
    def __init__(self, dwt=1.0):
        super().__init__()
        self.dwt    = torch.tensor(dwt).to(DEVICE)

        self.y_mean = None
        self.y_std  = None

    def set_scale(self, y_mean, y_std):
        self.y_mean = torch.FloatTensor(y_mean.tolist()).to(DEVICE)
        self.y_std  = torch.FloatTensor(y_std.tolist()).to(DEVICE)

    def __repr__(self):
        return "WRMSELoss_Ratio_dipole(dwt={})".format(self.dwt)

    def forward(self, y, y_pred):
        """
        y:      (E, dipx,      dipy,      dipz     )
        y_pred: (   dipx_pred, dipy_pred, dipz_pred)
        """
        assert self.y_mean is not None
        assert self.y_std is not None

        # descale
        dip_pred = y_pred * self.y_std[1:] + self.y_mean[1:]
        yd       = y      * self.y_std     + self.y_mean
        dip      = yd[:, 1:]
        en       = yd[:, 0]

        # Sync minimum across ranks for consistent weighting in distributed mode
        en_min = reduce_min(en.min())
        w  = self.dwt / (self.dwt + en - en_min)

        dd   = dip - dip_pred
        wdd  = torch.einsum('ij,i->ij', dd, w)
        wmse = torch.mean(torch.einsum('ij,ij->i', wdd, dd))

        #for k in range(10):
        #    print("dip: {}; dip_pred: {}".format(dip[k].detach().numpy(), dip_pred[k].detach().numpy()))

        return 1000.0 * torch.sqrt(wmse)
        #return 1000.0 * torch.mean(torch.abs(wdd))
