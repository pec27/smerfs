"""
Optimised covariance and cross-covariance
"""

from lib import chyp_c
from numpy import sqrt, complex128, pi, sin, roots, unique, real, zeros, shape, empty, empty_like, ones, power
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
        print 'Polynomial has', len(b), 'roots but only', len(unique(b)), 'unique roots'
        raise Exception('Power spectrum has repeated roots.')

    a = [1.0 / sum((n+1) * c_i * b_i**n for n,c_i in enumerate(coeffs[1:])) for b_i in b]

    return a, b

def cov_covar(zpts, m_max, coeffs):
    """ Find the covariance and cross covariance matrices for the zpts """
    if all(zpts[:-1] > zpts[1:]):
        print 'In reverse order to normal (increasing z)'
        print 'Call the function and reverse'
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
    tau = sqrt((1.0-zpts)/(1.0+zpts))


    zeta = empty(M, dtype=complex128)
    zetaA = empty(M, dtype=complex128) # alternating (-1)^p factor
    F = empty((m_max+M+1, len(zpts)), dtype=complex128)
    H = empty_like(F)
    
    cov = zeros((m_max+1,M,M,len(zpts)), dtype=complex128)
    cross_cov = zeros((m_max+1,M,M,len(zpts)-1), dtype=complex128)

    # Loop over partial fraction decomposition
    for ai, llp1 in zip(a, kvals):

        for m in range(m_max+M+1):
            F[m] = chyp_c(llp1, m, 0.5-zpts*0.5)
            H[m] = chyp_c(llp1, m, 0.5+zpts*0.5)
            
        eta = ones(len(zpts)-1)
        zeta[0] = -(0.25/pi) * ai * pi / sin(lam_from_llp1(llp1)*pi)
        zetaA[0] = 1.0        

        for m in range(m_max+1):

            # zeta symbols
            v = (m+1)+np.arange(M-1)
            zeta[1:] = (v*(v-1) - llp1)/v
            zetaA[1:] = -zeta[1:]
            zeta = np.cumprod(zeta)
            zetaA = np.cumprod(zetaA)

            # Todo: numpy broadcast with reshape
            for p in range(M):
                for q in range(M):
                    cov[m,p,q] += power(tau, p-q) * zeta[p] * zetaA[q] * F[m+p] * H[m+q]
                    cross_cov[m,p,q] += eta * power(tau[1:], q) / power(tau[:-1], p) * zeta[q] * zetaA[p] * H[m+p,:-1] * F[m+q,1:]

            zeta[0] = zeta[1]/(m+1)
            eta *= tau[1:] / tau[:-1]

    return cov, cross_cov

    
