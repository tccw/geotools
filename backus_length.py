import numpy as np
from bruges.rockphysics import backus
from welly import Well
from sympy import *
from collections import namedtuple

def backus_length(vp, vs, rho, dz, f, lb = np.arange(1,300,50)):
    """
    Liner & Fei Backus thickness determination via the 'Backus Number.'

    Parameters
    ----------
    vp, vs, rho: arrays
    dz: depth step
    f: the maximum frequency we are interesting in maintaining accurate
        wavefield informamtion after averaging
    lb: range of backus lengths to test. This range does not need to be
        exhaustive for the method utilized here to work

    This function uses the work of Liner & Fei to calculate the Backus average
    of the input curves at varying layer thicknesses. It outputs two terms:

    1. The scattering limit (B <1/3) for which the simulated wavefield is
    indistinguishable for the original model and the averaged one

    2. The transmission limit (1/3 < B < 2) for which the scattered field is
    progressively degraded to an unusable level, but the direct-arrival
    wavefront remains accurate in timing, waveform, and amplitude. This is the
    limit that would apply to depth migration, normal moveout, and other seismic
    processes that are based, ultimately, on wavefront accuracy.

    B = (L'*f)/Vs min

    Variables:
        B = Backus number
        L' = Backus layer thickness
        f = frequency (Hz)
        Vs min = The minimum shear velocity after backus averaging

    References:

    [https://library.seg.org/doi/abs/10.1190/1.2723204]

    The Backus number
    Liner,Chris et al.
    The Leading Edge(2007),26(4):420
    http://dx.doi.org/10.1190/1.2723204

    """
    b = {f'{l} m':backus(vp,vs,rho,l,dz) for l in lb}
    vs_min = [np.nanmin(getattr(b[k],'Vs')) for k in b.keys()]

    sctr = []
    transm = []
    L = Symbol('L')

    for v in vs_min:
        B = L * f / v
        sctr.append(solve(B - 0.3333))
        transm.append(solve(B - 2))

    Backus_Length = namedtuple('Backus_Length', 'scattering_limit transmission_limit')
    limits = Backus_Length(scattering_limit = np.mean(sctr).round(4),
             transmission_limit = np.mean(transm).round(4))

    return limits
