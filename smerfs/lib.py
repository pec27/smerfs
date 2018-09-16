"""
Load the compiled library (libsmerfs.so) 
"""
from __future__ import print_function, division, absolute_import
from numpy.ctypeslib import ndpointer
import ctypes
from numpy import float64, frombuffer, empty, complex128, array, require
from numpy.linalg import inv, LinAlgError
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

    name = path.join(path.dirname(path.abspath(__file__)), '../libsmerfs'+suffix)
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

    # Invert many small symmetric matrices
    # int inverse(const int N, const int M, const double *restrict matrices, double *restrict out)
    func = _libsmerfs.inverse
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags=c_contig), ndpointer(ctypes.c_double, flags=c_contig)]
    
    return _libsmerfs
    
def inv_sym(matrices):
    """
    Inverse of many small symmetric matrices
    """
    assert(len(matrices.shape)==3)
    assert(matrices.shape[1]==matrices.shape[2])
    N = matrices.shape[0]
    M = matrices.shape[1]
    if M>3:
        return np.inv(matrices)

    matrices = require(matrices, dtype=float64, requirements=[c_contig])
    out = empty((N,M,M), dtype=float64)
    lib = initlib()
    res = lib.inverse(N,M,matrices, out)
    if res==-1:
        raise Exception('Matrix size %d not supported'%M)
    if res>0:
        raise LinAlgError('Singular matrix (%d)'%(res-1))
        
    return out

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



