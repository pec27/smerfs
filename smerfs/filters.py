"""
Module for the objects for holding the filters and loading/saving.

EquatorialSphereFilter - Version that starts at equator and uses north-south symmetry, save as .dat files

load_filter(name) - method to load filters
build_filter(nz, nphi, coeffs, lmax, eta, dtype) - method to make new filters

"""
from __future__ import print_function, division, unicode_literals, absolute_import
from numpy.fft import irfft
from numpy import float64, empty, cumsum, arange, uint32, sqrt, load, pi, sin, empty_like, transpose, require, complex128
from .condition import gm_walkz
from .random import z_complex_normal # ziggurat normals
from numpy.random import RandomState

try:
    import cPickle as pickle
except:
    import pickle

class EquatorialSphereFilter:
    """ 
    Starts at the equator and works up
    (saves calculation and memory on the SphereFilter)

    The SphereFilter is an object that holds filter coefficients for filtering
    white noise on S^2 to produce a Gaussian Random Field with a given power
    spectrum, using the method
    
    s.create_realisation()

    The power spectrum of the walk is given by the tuple s.coeffs, which gives
    the coefficients for the pseudo-Markov process.
    """

    def __init__(self, coeffs, trans, innovs, nz, nphi, randomstate=None): 


        self.coeffs = tuple(coeffs)
        self._trans  = trans
        self._jumps  = innovs
        self._nphi = int(nphi)
        self._nz = nz
        if randomstate is None:
            randomstate = RandomState(seed=123)
        self._randomstate = randomstate
    
    def create_realisation(self, log=None):
        """ create a realisation of the given GMRF """
        
        p = self._nphi # number of phi points
        n = self._nz # number of z points
        M = len(self.coeffs)-1 # Markov order of the process

        rs = self._randomstate

        n_m = p//2 + 1 # number of m-modes for these phi-points
        # generate the noise
        if log is not None:
            print('Creating {:,} random numbers'.format(n_m*M*n*2), file=log)

        noise = z_complex_normal(n*M*n_m, rs).reshape((n,M,n_m))
        res = empty((n,n_m), dtype=noise.dtype)

        # Work from the (or just below the) equator up
        uhalf = n//2 + 1
        assert(self._jumps.shape[0]==uhalf)
        if log is not None:
            print('Filtering', file=log)
        fp_fstart = (self._jumps[0] * noise[0]).sum(1)
        fp_f = fp_fstart
        res[uhalf-1] = fp_f[0] # just interested in storing the value, not the derivatives
        for i in range(uhalf-1):
            fp_f = (self._trans[i] * fp_f+ self._jumps[i+1] * noise[i+1]).sum(1)
            res[uhalf-i-2] = fp_f[0]

        # And down...
        # (first need to reverse derivatives)
        for i in range(M):
            if i%2==0:
                fp_f[i] = fp_fstart[i]
            else:
                fp_f[i] = -fp_fstart[i]
        
        # If we had an even number of iso-lattitude lines then the first jump was from below
        # the equator to above, and we don't repeat this step.
        # If there is an odd number the steps are exactly the same.

        skip = 1 - (n%2) # 1 if even, 0 if odd.
        for i in range(n - uhalf):
            fp_f = (self._trans[i+skip] * fp_f+ self._jumps[i+skip+1] * noise[uhalf+i]).sum(1)
            res[uhalf+i] = fp_f[0]

        res[:,0] = res[:,0].real * sqrt(2.0) # Makes up for us simulating the real component g_0 with complex noise

        if log is not None:
            print('FFT', file=log)
        res = irfft(res, p)
        res *= p # FFT convention

        return res


    def shape(self):
        """ return (ntheta, nphi), the shape of the realisations """
        return self._nz, self._trans.shape[-1]

    def __str__(self):
        return 'coeffs: '+str(self.coeffs)+' shape '+str(self.shape())

    def save(self, name):
        """ save the filter to the named file or file-like object (write method) """

        trans_packed, innovs_packed = packup_coeffs(self._trans, self._jumps, dtype=self._trans.dtype)
        data = (self.coeffs, trans_packed, innovs_packed, self._nz, self._nphi)

        if hasattr(name, 'write'):
            pickle.dump(data, name, protocol=2) # need 2 protocol for numpy data in binary
            return 

        assert(name[-4:]=='.dat') # check is a .dat file 

        with open(name, 'wb') as f:
            pickle.dump(data, f, protocol=2) # need 2 protocol for numpy data in binary


     
def load_filter(name, randomstate=None, log=None):
    """ 
    load an EquatorialSphereFilter (.dat) 
    
    name          - path to file
    [randomstate] - optional numpy.RandomState object for re-sampling.

    returns sf - SphereFilter or EquatorialSphereFilter
    """
    if hasattr(name, 'read') and hasattr(name, 'readline'):
        coeffs, transp, innovsp, nz, nphi = pickle.load(name)
    else:
        f = open(name, 'rb')
        coeffs, transp, innovsp, nz, nphi = pickle.load(f)
        f.close()

    trans, innovs = unpack_coeffs(transp, innovsp, nz, nphi, log=log)

    rs = randomstate
    if rs is None:
        rs = RandomState()
    esf = EquatorialSphereFilter(coeffs, trans, innovs, nz, nphi, rs)
    return esf

def packup_coeffs(trans, innovs, dtype, log=None):
    print('Packing up coefficients',file=log)
    M = innovs.shape[1]

    trans_packed = trans.astype(dtype)
    nz, n_m = innovs.shape[0], innovs.shape[3]

    # Only lower triangle for Cholesky decomposition
    lower_tri = cumsum(arange(M))
    innov_packed = empty((nz, (M*M+M)//2,n_m), dtype=dtype)

    for i, p0 in enumerate(lower_tri):
        p1 = p0 + i + 1
        innov_packed[:,p0:p1,:] = innovs[:,i,:i+1,:]
    assert((innov_packed!=0).all())

    return trans_packed, innov_packed

def unpack_coeffs(trans_packed, innov_packed, nz, nphi, log=None):
    print('Unpacking coefficients', file=log)
    # Split out the trans matrices for the different m values.
    M  = trans_packed.shape[1]
    trans = trans_packed.copy()
        
    # Similar for innov matrices, these are further complicated as we only
    # store lower triangle of the Cholesky decomposition.
    m_size = trans.shape[-1]
    nz = trans.shape[0]+1
    innovs = empty((nz, M, M, m_size), dtype=innov_packed.dtype)

    lower_tri = cumsum(arange(M))
    

    for i, p0 in enumerate(lower_tri):
        p1 = p0 + i + 1
        innovs[:,i,:i+1] = innov_packed[:,p0:p1,:] # Lower triangle
        innovs[:,i,i+1:] = 0.0 # Upper triangle

    return trans, innovs
    uhalf = nz//2+1 # Upper half of sphere
    n_m = nphi//2+1
    trans_matrix = empty((uhalf-1, M, M, n_m), dtype=trans_packed.dtype)
    innov_matrix = empty((uhalf, M, M, n_m), dtype=innov_packed.dtype)
    
    for m in range(n_m):
        trans_matrix[:,:,:,m] = trans[m]
        innov_matrix[:,:,:,m]   = innovs[m]
    
    return trans_matrix, innov_matrix


def build_filter(nz, nphi, coeffs, dtype=float64, log=None):
    """ 
    Construct an EquatorialSphereFilter for the pseudo-Markov process on the iso-latitude
    grid with points equidistant in theta and phi.

    nz     - the number of theta points, theta = (2i+1)/(2pi nz)
    nphi   - the number phi points, {2 pi i / nphi : for i in [0,1, ... nphi-1]}
    coeffs - coefficients of the pseudo-Markov process

    returns 
    sf - an instance of SphereFilter
    """

    trans, innovs = gm_walkz(nz, nphi//2+1, coeffs, dtype, log)
    if log is not None:
        print('Reshaping', file=log)
    # making new shape:
    trans = require(transpose(trans, (1,2,3,0)), requirements=['C'])  #  (nz-1, M,M, nphi//2+1)
    innovs = require(transpose(innovs, (1,2,3,0)), requirements=['C']) #  (nz-1, M,M, nphi//2+1)
    sf = EquatorialSphereFilter(coeffs, trans, innovs, nz, nphi)
    return sf


if __name__=='__main__':
    from numpy import float32
    from os import path
    nz, nphi = 256*16, 512*16
    dtype = float32
    coeffs = (1.0,0.0,0.0003)

    # cmb like
#    nz, nphi = 256*2,512*2
#    coeffs = (1.0,0.0, 1e-6)
    name = 'filter_%d_%d.dat'%(nz,nphi)
    if path.exists(name):
        sf = load_filter(name)
    else:
        sf = build_filter(nz,nphi, coeffs, dtype=dtype)
        sf.save(name)
    from time import time
    t0 = time()
    res = sf.create_realisation()
    dt = time() - t0
    print('Time taken', dt, 's')

    import pylab as pl

    print(res.shape)
    pl.imshow(res)
    pl.show()
