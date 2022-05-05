import numpy as np
import matplotlib.pyplot as plt

LightSpeed_cm = 2.99792458e10
EVTOCM        = 8065.73
DALTON        = 1.66054e-27
EVTOJ         = 1.602176565e-19

Boltzmann     = 1.380649e-23
Hartree       = 4.3597447222071e-18
HkT           = Hartree / Boltzmann
HTOCM         = 2.1947463136320e5
VkT           = HkT / HTOCM

def Poten_N2(r):
    """
    returns N2 potential [cm-1] approximated as a Morse curve
    the parameters are derived from experiment
    accepts the distance in A
    """
    # https://doi.org/10.1098/rspa.1956.0135 
    De = 9.91 # eV 

    omega = 2358.57 # cm-1
    nu = 2.0 * np.pi * LightSpeed_cm * omega # 1/s
    mu = 14.003074004460 / 2.0 * DALTON

    a = np.sqrt(mu / (2.0 * De * EVTOJ)) * nu * 1e-10 # A 
    re = 1.09768 # A

    return (De * EVTOCM) * (1 - np.exp(-a * (r - re)))**2
