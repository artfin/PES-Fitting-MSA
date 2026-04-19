"""
Samplers for multi-batch L-BFGS training.

Two modes are supported, mirroring hjmshi/PyTorch-LBFGS examples:

- ``full_overlap``: one random batch per step; the same batch is used for
  the gradient that drives the search direction and for the Wolfe (or Armijo)
  line search. Each call to ``next_step`` returns ``(Sk,)`` where ``Sk`` is
  the batch index array.

- ``multi_batch``: consecutive random batches with a fixed overlap. The
  step uses an aggregated gradient over overlap_prev, non-overlap, and
  overlap_next with weights (alpha, 1-2*alpha, alpha). The curvature pair
  is formed only on the current overlap set Ok, so ``next_step`` returns
  ``(Ok, Nk)``; the "previous overlap" is carried inside the sampler.

Both samplers use an independently seeded numpy ``RandomState`` so the
stream is reproducible and does not perturb ``np.random``.
"""

import numpy as np


class FullOverlapSampler:
    def __init__(self, n_samples, batch_size, seed=42):
        assert batch_size > 0
        self.n_samples = int(n_samples)
        self.batch_size = min(int(batch_size), self.n_samples)
        self._rng = np.random.RandomState(int(seed))

    def steps_per_epoch(self):
        return max(1, self.n_samples // self.batch_size)

    def next_step(self):
        perm = self._rng.permutation(self.n_samples)
        return (perm[: self.batch_size],)


class MultiBatchSampler:
    """Independent sampling per step (per hjmshi multi_batch example).

    Each call to ``next_step`` draws a fresh permutation and returns
    ``(Ok, Nk)``. The previous overlap Ok_prev from the last step is
    tracked internally and is accessible through ``current_prev_overlap``.
    """

    def __init__(self, n_samples, batch_size, overlap_fraction=0.25, seed=42):
        assert 0.0 < overlap_fraction < 0.5
        self.n_samples = int(n_samples)
        self.batch_size = min(int(batch_size), self.n_samples)
        self.overlap_fraction = float(overlap_fraction)
        self.Ok_size = max(1, int(self.overlap_fraction * self.batch_size))
        self.Nk_size = max(1, self.batch_size - 2 * self.Ok_size)
        self._rng = np.random.RandomState(int(seed))
        self._Ok_prev = None

    def steps_per_epoch(self):
        return max(1, self.n_samples // self.batch_size)

    def reset_prev_overlap(self):
        """Seed the first Ok_prev for a fresh epoch / state reset."""
        perm = self._rng.permutation(self.n_samples)
        self._Ok_prev = perm[: self.Ok_size]
        return self._Ok_prev

    def current_prev_overlap(self):
        if self._Ok_prev is None:
            self.reset_prev_overlap()
        return self._Ok_prev

    def next_step(self):
        perm = self._rng.permutation(self.n_samples)
        Ok = perm[: self.Ok_size]
        Nk = perm[self.Ok_size : self.Ok_size + self.Nk_size]
        return Ok, Nk

    def advance(self, Ok):
        """Call after the optimizer step to shift Ok_prev <- Ok."""
        self._Ok_prev = Ok
