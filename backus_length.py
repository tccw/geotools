import numpy as np
from bruges.rockphysics import backus
from welly import Well
from collections import namedtuple

def backus_length(vp, vs, rho, lolb, dz, f):
    """
    Liner & Fei Backus thickness determination via the 'Backus Number.'

    Args:
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        rho (ndarray): Bulk density.
        lolb (ndarray): List of test Backus averaging lengths in m.
        dz (float): The depth sample interval in m.
        f (ndarray): Max dominant or actual frequency for which we wish to
                     preserve the integrity of the scattering field
                     
    Returns: 
        namedtuple: the max scattering limit and transmission limit Backus averaging lengths
            in meters. Useful for deciding what the optimal Backus averaging window is for your data.

    Notes:
          B = (L'*f)/Vs min
        Variables:
            B = Backus number
            L' = Backus layer thickness
            f = frequency of interest (Hz)
            Vs min = The minimum shear velocity AFTER backus averaging
        
        1. The scattering limit (B <=1/3) for which the simulated wavefield is
        indistinguishable for the original model and the averaged one
        
        2. The transmission limit (1/3 < B <= 2) for which the scattered field is
        progressively degraded to an unusable level, but the direct-arrival
        wavefront remains accurate in timing, waveform, and amplitude. This is the
        limit that would apply to depth migration, normal moveout, and other seismic
        processes that are based, ultimately, on wavefront accuracy.
      
        References:
        [https://library.seg.org/doi/abs/10.1190/1.2723204]
        The Backus number
        Liner,Chris et al.
        The Leading Edge(2007),26(4):420
        http://dx.doi.org/10.1190/1.2723204
    """
    b = {f'{l} m':backus(vp,vs,rho,l,dz) for l in lolb}
    vs_min = np.array([np.nanmin(getattr(b[k],'Vs').round(0)) for k in b.keys()])

    L_sctr = ((1/3) * vs_min) / f
    L_transm = (2 * vs_min) / f

    Backus_Length = namedtuple('Backus_Length', 'scattering_limit transmission_limit')
    limits = Backus_Length(scattering_limit=np.mean(L_sctr).round(2),
                           transmission_limit=np.mean(L_transm).round(2))

    return limits
    """
    Liner & Fei Backus thickness determination via the 'Backus Number.'

    Args:
        vp (ndarray): P-wave interval velocity.
        vs (ndarray): S-wave interval velocity.
        rho (ndarray): Bulk density.
        lolb (ndarray): List of test Backus averaging lengths in m.
        dz (float): The depth sample interval in m.
        f (ndarray): Max dominant or actual frequency for which we wish to
                     preserve the integrity of the scattering field

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
    b = {f'{l} m':backus(vp,vs,rho,l,dz) for l in lolb}
    vs_min = np.array([np.nanmin(getattr(b[k],'Vs').round(0)) for k in b.keys()])

    L_sctr = ((1/3) * vs_min) / f
    L_transm = (2 * vs_min) / f

    Backus_Length = namedtuple('Backus_Length', 'scattering_limit transmission_limit')
    limits = Backus_Length(scattering_limit=np.mean(L_sctr).round(2),
                           transmission_limit=np.mean(L_transm).round(2))

    return limits
