"""
Utility functions for testing
"""
from numpy import *
from smerfs import chyp_c, lam_from_llp1, cov_covar, partial_decomposition
import numpy as np
from numpy.linalg import inv
from scipy.special import gamma, lpmn


def Jpq(m, coeffs, x, y):
    """
    Analytic expressions for J_pq
    Not optimised in any way, nor has care been taken to avoid high-m problems
    """

    M = len(coeffs)
    bs, llp1s = partial_decomposition(coeffs)

    res = zeros((M-1,M-1), dtype=complex128)

    for b, llp1 in zip(bs, llp1s):
        res += (0.5/pi)*b*G_m_llp1_pq(x,y, m, llp1, M)

    return res

def pochhammer(z, m):
    """
    Rising factorial (z)_m (scipys poch doesnt work with complex z)
    """
    return gamma(z+m)/gamma(z)



def _prefac(llp1, m):
    if m==0:
        return 1.0

    n = arange(m)
    f = (n*(n+1)-llp1) / (1.0 + n)
    return cumprod(f)[-1]

def P_m_llp1(m, llp1, x):
    """
    Evaluate the generalised Legendre function with 

    m    - non-negative integer order
    llp1 - lam*(lam+1) where lam the complex degree (excluding positive integers)
    x    - real in (-1,1]

    """

#    lam = lam_from_llp1(llp1)
#    return pochhammer(-lam, m)*pochhammer(1+lam,m)*chyp_c(llp1, m, 0.5-x*0.5) * sqrt(power((1.0-x)/(1.0+x), m)) * (1.0/ gamma(1+m))
    return _prefac(llp1, m) * chyp_c(llp1, m, 0.5-x*0.5) * sqrt(power((1.0-x)/(1.0+x), m))

def G_m_llp1(x,y, m, llp1):
    """
    Greens function for single x,y,m, lam(lam+1)
    """
    low, high = sorted((x,y)) # ordered values
    lam = lam_from_llp1(llp1)
    return 0.5 * gamma(-m-lam) * gamma(-m+1+lam) * P_m_llp1(m,llp1, -low) * P_m_llp1(m,llp1, high)

def G_m_llp1_pq(x,y, m, llp1, M):
    """
    Like G_m_llp1 but using the ladder operator up to M-1 w.r.t. x(p) and y(q)
    """

    left = x<=y

    lam = lam_from_llp1(llp1)
    res = empty((M-1,M-1), dtype=complex128)
    
    for p in range(M-1):
        for q in range(M-1):
            if left:
                r = power(-1,p) * P_m_llp1(m+p, llp1,-x) *P_m_llp1(m+q, llp1,y)
            else:
                r = power(-1,q) * P_m_llp1(m+p, llp1,x) *P_m_llp1(m+q, llp1,-y)

            res[p,q] = r[0] * 0.5 * gamma(-lam-m)*gamma(1+lam-m)
    return res

def num_deriv(f, x, dx=1e-5):
    """ utility function for numerical derivatives """
    return (0.5/dx) * (f(x+dx) - f(x-dx))

