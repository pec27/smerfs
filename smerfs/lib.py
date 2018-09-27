"""
Load the compiled library (libsmerfs.so) 
"""
from __future__ import print_function, division, absolute_import
from numpy.ctypeslib import ndpointer
import ctypes
from numpy import float64, frombuffer, empty, complex128, array, require
from numpy.linalg import inv, LinAlgError, cholesky
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

    
    # Cholesky many small symmetric matrices
    func = _libsmerfs.cholesky
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags=c_contig), ndpointer(ctypes.c_double, flags=c_contig)]

    # Build covariances in one pass in a C-func
    # int update_cov(const int m_max, const int nz, const int M, 
    #	       double complex norm, const double complex llp1,
    #	       const double complex *restrict F, const double complex *restrict H,
    #	       const double *restrict tau_power, const double *restrict eta_ratio,
    #	       double *restrict cov, double *restrict cross_cov)

    func = _libsmerfs.update_cov
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                     ndpointer(complex128, flags=c_contig), ndpointer(complex128, flags=c_contig),
                     ndpointer(float64, flags=c_contig), ndpointer(float64, flags=c_contig), 
                     ndpointer(float64, flags=c_contig), ndpointer(float64, flags=c_contig)]
                     

    return _libsmerfs
    
def update_cov(tau_power, eta_ratio, norm, llp1, F, H, cov, cross_cov):
    nz,M = cov.shape[1:3]
    m_max = cov.shape[0]-1
    assert(cov.shape[3]==M)
    assert(cross_cov.shape==(m_max+1,nz-1, M,M))
    assert(eta_ratio.shape==(nz-1,))

    lib = initlib()

    res = lib.update_cov(m_max, nz, M, norm.real, norm.imag, llp1.real, llp1.imag,
                         F, H, tau_power, eta_ratio, cov, cross_cov)
    assert(res==0)
    return cov, cross_cov

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

def cho(matrices):
    """
    Cholesky of many small symmetric matrices
    """
    assert(len(matrices.shape)==3)
    assert(matrices.shape[1]==matrices.shape[2])
    N = matrices.shape[0]
    M = matrices.shape[1]
    if M>2:
        return np.cholesky(matrices)

    matrices = require(matrices, dtype=float64, requirements=[c_contig])
    out = empty((N,M,M), dtype=float64)
    lib = initlib()
    res = lib.cholesky(N,M,matrices, out)
    if res==-1:
        raise Exception('Matrix size %d not supported'%M)
    if res>0:
        raise LinAlgError('Non +ve definite matrix (%d)'%(res-1))
        
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



