"""
Load the compiled library (libsmerfs.so) 
"""
from __future__ import print_function, division, absolute_import
from numpy.ctypeslib import ndpointer
import ctypes
from numpy import float64, frombuffer, empty, complex128, array, require, empty_like, uint32, float32
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

    suffix = sysconfig.get_config_var('EXT_SUFFIX')

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

    
    # State space for 2x2 and 3x3 matrices
    # int state_space(const int N, const int M, 
    #		const double *restrict cross_cov, const double *restrict cov, 
    #		double *restrict innov, double *restrict trans)
    func = _libsmerfs.state_space
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ctypes.c_int, 
                     ndpointer(float64, flags=c_contig), ndpointer(float64, flags=c_contig), 
                     ndpointer(float64, flags=c_contig), ndpointer(float64, flags=c_contig)]



    # Standard normal  float32s using the Ziggurat method (Marsaglia & Tsang)
    # int zigg(const int num_needed, int num_ints, const uint32_t *restrict rand_ints, 
    #	 float *restrict out)
    func = _libsmerfs.zigg
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_int, ctypes.c_int, 
                     ndpointer(uint32, flags=c_contig), ndpointer(float32, flags=c_contig)]

    return _libsmerfs

def zigg(n_wanted, r_int):
    assert(r_int.dtype==uint32)

    out = empty(n_wanted, float32)
    lib = initlib()

    # number of normals remaining:
    todo = lib.zigg(n_wanted, r_int.size, r_int, out)
    # return the random normals (or as many as I could make with these integers)
    return out[:(n_wanted-todo)]

def state_space(cross_cov, cov):
    N,M = cov.shape[:2]
    assert(M in (2,3))
    assert(cov.shape==(N,M,M))
    assert(cross_cov.shape==(N-1,M,M))

    cc = require(cross_cov, requirements=['C'], dtype=float64)
    cv = require(cov, requirements=['C'], dtype=float64)
    
    trans = empty_like(cross_cov)
    innov = empty_like(cov)
    lib = initlib()
    res = lib.state_space(N,M,cc, cv, innov, trans)

    if res==-1:
        raise Exception('Matrix size %d not supported'%M)
    if res>0:
        raise LinAlgError('Inverse or cholesky proplem at matrix (%d)'%(res-1))

    assert(res==0)
    return innov, trans

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
        return cholesky(matrices)

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



