# Potential energy surfaces for monomers

The potential energy surface for methane is taken from the following paper:
A. Owens, S. N. Yurchenko, A. Yachmenev, et al. "A highly accurate ab initio potential energy surface for methane", JCP 145, 104305 (2016).
htpps://doi.org/10.1063/1.4962261

The code is taken from supplementary materials and slightly rewritten for better use; the initialization is now separated into the `potinit` procedure.
The use of potentials in C is demonstrated in the `sampler.cc` file and in Python in `pybind_example.py` file.


# Sampling
File `sampler.cc` contains the Markov Chain Monte Carlo sampling of the CH4 PES. The sampled configurations have been used in the quantum chemical calculations of the intermolecular energies of the CH4-N2 dimer.

Sampler utilizes the Mersenne Twister pseudorandom generator provided by the `mtwist` package (taken from http://www.cs.hmc.edu/~geoff/mtwist.html).
