import torch

from config import TORCH_FLOAT
from losses import WMSELoss_TrustRegion_wgradients

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainingGradientsMixin:
    def compute_gradients(self, dataset):
        Xtr = dataset.X

        Xtr.requires_grad = True

        y_pred = self.model(Xtr)
        dEdp   = torch.autograd.grad(outputs=y_pred, inputs=Xtr, grad_outputs=torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        # now we have derivatives of energy w.r.t. to polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        # `torch.einsum` throws a Runtime error without an explicit conversion to Double
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dataset.dX.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return y_pred, dEdx
    def compute_gradients_from_energy(self, X_subset, dX_subset, y_pred_subset):
        """
        Compute gradients for a subset given pre-computed energy predictions.
        This avoids a second forward pass through the model.

        Args:
            X_subset: Input polynomials for subset (must have requires_grad=True)
            dX_subset: Polynomial gradients for subset
            y_pred_subset: Energy predictions for subset (from same forward pass)
        """
        logging.debug("compute_gradients_from_energy: X_subset shape={}, dX_subset shape={}".format(
            X_subset.shape, dX_subset.shape))

        dEdp = torch.autograd.grad(
            outputs=y_pred_subset,
            inputs=X_subset,
            grad_outputs=torch.ones_like(y_pred_subset),
            retain_graph=True,
            create_graph=True
        )[0]

        # take into account normalization of polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx = \sigma(E) * dE/d(poly) * d(poly)/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dX_subset.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return dEdx
    def compute_trust_mask(self, dataset):
        """
        Compute trust region active set based on energy prediction errors
        and optionally gradient errors from the previous epoch.

        Uses soft boundaries with sigmoid weighting:
            phi(e_i) = sigmoid((threshold - error) / soft_scale) in [0, 1]
        Active set = {i : phi(e_i) > soft_cutoff} (memory optimization).

        When GRADIENT_TRUST_THRESHOLD is set, configs with large gradient
        errors (from previous epoch) are down-weighted using a soft sigmoid.
        This is combined multiplicatively with energy-based weights.

        Returns:
          trust_indices : 1-D LongTensor of active-set config indices
          trust_mask    : 1-D BoolTensor of shape (N,) indicating membership
          energy_errors : 1-D float tensor of |E_pred - E_true| (cm^-1)
          gradient_weights : 1-D float tensor of combined weights for the active set
        """
        trust_threshold = getattr(self, 'current_trust_threshold', None)
        if trust_threshold is None:
            trust_threshold = self.cfg_loss.get('TRUST_THRESHOLD', 50.0)

        soft_scale = self.cfg_loss.get('TRUST_SOFT_SCALE', None)
        soft_cutoff = self.cfg_loss.get('TRUST_SOFT_CUTOFF', 0.01)

        # Gradient trust: filter by previous epoch's gradient errors
        grad_trust_threshold = self.cfg_loss.get('GRADIENT_TRUST_THRESHOLD', None)
        grad_trust_soft_scale = self.cfg_loss.get('GRADIENT_TRUST_SOFT_SCALE', None)

        with torch.no_grad():
            y_pred = self.model(dataset.X)

            # Descale energies
            en_mean = torch.from_numpy(self.yscaler.mean_).to(self.device)
            en_std = torch.from_numpy(self.yscaler.scale_).to(self.device)

            en_pred_descaled = y_pred * en_std + en_mean
            en_true_descaled = dataset.y * en_std + en_mean

            energy_errors = torch.abs(en_pred_descaled - en_true_descaled).view(-1)

            # Always use soft boundary with sigmoid weighting
            phi_energy = WMSELoss_TrustRegion_wgradients.soft_phi(
                energy_errors, trust_threshold, soft_scale=soft_scale
            )
            trust_mask = phi_energy > soft_cutoff
            trust_indices = torch.nonzero(trust_mask, as_tuple=False).view(-1)
            gradient_weights = phi_energy[trust_indices]

            # Apply gradient trust filtering (uses previous epoch's gradient errors)
            if (grad_trust_threshold is not None
                    and self._prev_train_gradient_errors is not None
                    and self._prev_train_gradient_errors.numel() == energy_errors.numel()):
                # Compute soft phi for gradient errors (same sigmoid as energy)
                phi_grad = WMSELoss_TrustRegion_wgradients.soft_phi(
                    self._prev_train_gradient_errors,
                    grad_trust_threshold,
                    soft_scale=grad_trust_soft_scale
                )
                # Multiply energy weights by gradient weights
                gradient_weights = gradient_weights * phi_grad[trust_indices]

        return trust_indices, trust_mask, energy_errors, gradient_weights
    def compute_gradients_eval(self, dataset):
        """
        Compute gradients for evaluation (no create_graph needed).
        Much more memory efficient than compute_gradients() since we don't need
        to backpropagate through the gradient computation.
        """
        Xtr = dataset.X.clone().detach()
        Xtr.requires_grad = True

        with torch.enable_grad():
            y_pred = self.model(Xtr)
            dEdp = torch.autograd.grad(
                outputs=y_pred,
                inputs=Xtr,
                grad_outputs=torch.ones_like(y_pred),
                retain_graph=False,
                create_graph=False
            )[0]

        Xtr.requires_grad = False

        # take into account normalization of polynomials
        x_scale = torch.from_numpy(self.xscaler.scale_).to(self.device)
        dEdp = torch.div(dEdp, x_scale)

        # gradient = dE/dx
        dEdx = torch.einsum('ij,ijk -> ik', dEdp.to(TORCH_FLOAT), dataset.dX.to(TORCH_FLOAT))

        # take into account normalization of model energy
        y_scale = torch.from_numpy(self.yscaler.scale_).to(self.device)
        dEdx = torch.mul(dEdx, y_scale)

        return y_pred.detach(), dEdx.detach()
