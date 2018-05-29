"""
Load the compiled library (libsmerfs.so) from ../build
"""
from __future__ import print_function, division, absolute_import
from numpy.ctypeslib import ndpointer
import ctypes
from numpy import float64, frombuffer, empty, complex128, array, require
from os import path
import sys
import sysconfig

_libsmerfs = None
c_contig = 'C_CONTIGUOUS' 

def initlib():
    """ Init the library (if not already loaded) """
    global _libsmerfs

    if _libsmerfs is not None:
        return _libsmerfs

    suffix = sysconfig.get_config_var('SO')

    name = path.join(path.dirname(path.abspath(__file__)), '../build/libsmerfs'+suffix)
    if not path.exists(name):
        raise Exception('Library '+str(name)+' does not exist. Maybe you forgot to make it?')

    print('Loading libsmerfs - Stochastic Markov Evaluation of Random Fields on the Sphere')
    _libsmerfs = ctypes.cdll.LoadLibrary(name)

    # Hypergeometric function evaluation
    # C declaration is below
    # int hyp_llp1(const double complex llp1, const int m, const int nz, const double *zvals, double complex *out)
    func = _libsmerfs.hyp_llp1
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags=c_contig), ndpointer(complex128, flags=c_contig)]

    return _libsmerfs
    
def chyp_c(llp1, m, z):
    """
    Calculates the Gauss hypergeometric function

     F (-lam, lam+1; 1+m; z) 
    2 1

    with arguments

    llp1 - lam(lam+1) complex 
    m    - nonnegative integer
    z    - array of reals in (-1, 1]

    and returns array of results (complex128) of the same size as z.

    Notes - This is used in the evaluation of Legendre functions
    """
    lib = initlib()
    z = require(z, dtype=float64, requirements=[c_contig]) # not allowed array slices etc
    nz = z.size 

    out = empty(z.size, dtype=complex128)
    res = lib.hyp_llp1(llp1.real, llp1.imag, m, z.size, z, out)
    if res != 0:
        raise Exception('libsmerfs could not calculate hypergeometric function!')

    return out



