"""
Optimised covariance and cross-covariance
"""
from __future__ import print_function, division, unicode_literals, absolute_import
from . import chyp_c
from .lib import update_cov
from numpy import sqrt, complex128, float64, pi, sin, roots, unique, real, zeros, shape, empty, empty_like, ones, power
import numpy as np

def lam_from_llp1(llp1):
    """
    l given l(l+1)
    """
    if real(llp1)<-0.25:
        lam = -0.5 + 1.0j * sqrt(-0.25 - llp1)
    else:
        lam = -0.5 - sqrt(0.25 + llp1)

    return lam

def partial_decomposition(coeffs):
    """
    The fraction 
    
         1
    -----------
    Sum c_n k^n

    where c_n is from the tuple of coeffs. We use the partial fraction
    decomposition to write this as

            /     a_i      \
    Sum    |  ------------  |
            \  ( k - b_i)  /

    i.e. the {b_i} are the roots of the polynomial. Note that even if the 
    c_n are all real and positive, the {a_i} and {b_i} are generally complex.

    If any of the roots are repeated then we have an exception, we cannot handle this case
    
    returns a, b [tuples]

    """

    b = roots(list(reversed(coeffs)))
    
    if len(unique(b))!=len(b):
        print('Polynomial has', len(b), 'roots but only', len(unique(b)), 'unique roots')
        raise Exception('Power spectrum has repeated roots.')

    a = [1.0 / sum((n+1) * c_i * b_i**n for n,c_i in enumerate(coeffs[1:])) for b_i in b]

    return a, b

def cov_covar(zpts, m_max, coeffs, log=None):
    """ Find the covariance and cross covariance matrices for the zpts """
    if all(zpts[:-1] > zpts[1:]):
        print('In reverse order to normal (increasing z)', file=log)
        print('Call the function and reverse', file=log)
        cov, cross_cov = all_z_covar_hyp(zpts[::-1], m, coeffs)
        cov = transpose(cov[:,:,::-1], (1,0,2))
        cross_cov = transpose(cross_cov[:,:,::-1], (1,0,2))
        return cov, cross_cov
    
    if any(zpts[:-1] > zpts[1:]):
        raise Exception('z values should be monotonic...')

    a,kvals = partial_decomposition(coeffs)
    M = len(coeffs)-1
    cov = zeros((M,M)+shape(zpts), dtype=complex128)
    cross_cov = zeros((M,M)+shape(zpts[1:]), dtype=complex128)

    # twiddle factors
    tau_p = empty((2*M-1,len(zpts)), dtype=float64)
    tau_p[M-1] = 1.0;
    tau_p[M] = sqrt((1.0-zpts)/(1.0+zpts)) # tau^1
    tau_p[M-2] = 1.0/tau_p[M] # tau^-1
    for i in range(2,M):
        tau_p[M+i-1] = tau_p[M+i-2]*tau_p[M] # tau^i
        tau_p[M-1-i] = tau_p[M-i]*tau_p[M-2] # tau^-i

    eta_ratio = tau_p[M,1:] * tau_p[M-2,:-1]
#    tau_p = np.require(tau_p.T, requirements=['C'])

    F = empty((m_max+M+1, len(zpts)), dtype=complex128)
    H = empty_like(F)
    
    cov = zeros((m_max+1,len(zpts), M, M), dtype=float64)
    cross_cov = zeros((m_max+1,len(zpts)-1, M, M), dtype=float64)

    x = 0.5-zpts*0.5
    y = 0.5+zpts*0.5

    # Loop over partial fraction decomposition
    for i, (ai, llp1) in enumerate(zip(a, kvals)):


        if len(kvals)==2 and i==1 and llp1==kvals[i-1].conj():
            print('Using conjugation for coefficient pair', file=log)
            cov += cov
            cross_cov += cross_cov
            return cov, cross_cov


#        print('Hypergeometric funcn calc', file=log)
        for m in range(m_max+M+1):
            F[m] = chyp_c(llp1, m, x)
            H[m] = chyp_c(llp1, m, y)


        norm = -(0.25/pi) * ai * pi / sin(lam_from_llp1(llp1)*pi)

#        print('Putting into matrix in C', file=log)
        update_cov(tau_p, eta_ratio, norm, llp1, F, H, cov, cross_cov)

    return cov, cross_cov

    
