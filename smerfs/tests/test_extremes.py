"""
Test extreme values in m, M and z
"""
from __future__ import print_function, division, unicode_literals, absolute_import
from .utils import *

def test_highz():
    """
    High z values
    """
    z = -cos([1e-3, 2e-3])
    m = 5
    coeffs = (1.0, 0.0, 0.1)

    cov, covar = cov_covar(z, m+1, coeffs)
    cov = cov[m,:,:,0]

    max_imag = abs(cov.ravel().imag).max()
    asym = abs(cov - cov.T).ravel().max()

    print(max_imag, asym)

    assert(max_imag<1e-12)
    assert(asym<1e-10)


def test_high_m():
    """
    High m values 
    (quite slow since it goes 0...m)
    """
    z = array([0.5,0.6])
    m = 500
    coeffs = (1.0, 0.0, 0.1)

    cov, covar = cov_covar(z, m+1, coeffs)
    cov = cov[m,:,:,0]

    max_imag = abs(cov.ravel().imag).max()
    asym = abs(cov - cov.T).ravel().max()

    print(max_imag, asym)
    
    assert(max_imag<1e-12)
    assert(asym<1e-10)

def test_high_M():
    """
    Lots of coeffs (M) (cant actually go high with this)
    """

    z = array([0.5,0.6])
    m = 5
    coeffs = (1.0,)+(0.0,)*4 + (0.1,)

    cov, covar = cov_covar(z, m+1, coeffs)
    cov = cov[m,:,:,0]

    max_imag = abs(cov.ravel().imag).max()
    asym = abs(cov - cov.T).ravel().max()

    print(max_imag, asym)
    
    assert(max_imag<1e-8)
    assert(asym<1e-8)
    
