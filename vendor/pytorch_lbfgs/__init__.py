"""
Vendored copy of hjmshi/PyTorch-LBFGS (https://github.com/hjmshi/PyTorch-LBFGS).
See ./LICENSE for upstream license (3-Clause BSD).

Exposes LBFGS / FullBatchLBFGS without pulling in the utils helpers, which
depend on numpy/keras for the CIFAR examples and are not needed here.
"""

from .LBFGS import LBFGS, FullBatchLBFGS

__all__ = ["LBFGS", "FullBatchLBFGS"]
