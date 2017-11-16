"""
Utility functions (typically for comparing power spectra)
"""
from __future__ import print_function, division, unicode_literals, absolute_import
from numpy import arange, float64, inner, cumprod, flatnonzero, empty, pi, zeros
from scipy.special import lpn


def make_cl(coeffs, lmax):
    """ Calculate the C_l (spherical harmonic coeffs) up to l_max """
    M = len(coeffs) # Markov order of process
    C_l = empty((lmax+1, M), dtype=float64)
    llp1 = arange(lmax+1)  # Array of l(l+1)
    llp1 *= llp1 + 1 

    C_l[:,0] = 1.0
    for i in range(M-1):
        C_l[:,i+1] = llp1
        
    C_l = 1.0 / inner(cumprod(C_l, axis=1), coeffs)

    if C_l.min() <  0.0:
        print('C_l in', C_l.min(), C_l.max())
        l = flatnonzero(C_l<0)[0]
        raise Exception('At l=%d, C_l=%f. C_l must all be positive!'%(l, C_l[l]))
    return C_l

def analytic_cov(coeffs,cos_mu, lmax=1000):
    """
    Evaluate the covariance function 

                  inf
                  ---
    C(cos(mu)) := \   (2l+1) C_l P_l (cos(mu)) / 4 pi
                  /
                  ---
                  l=0

    """

    # Make the C_l from the Markov coefficients
    C_l = make_cl(coeffs,lmax)
    # Coeffs excluding legendre poly, i.e. C_l * (2l+1) / 4 pi
    l_coeffs  = C_l * (2*arange(lmax+1)+1) * (0.25/pi)

    correl = zeros(len(cos_mu))
    for i,z in enumerate(cos_mu):
        Plz, dPlz_dz = lpn(lmax, z) # Legendre polys and their derivs
        correl[i] = inner(Plz, l_coeffs)

    return correl

