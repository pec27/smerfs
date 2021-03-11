"""
Use a Ziggurat method to make random normal variates faster than numpy
"""
from __future__ import absolute_import
from numpy import empty, float32, frombuffer, uint32, complex64
from numpy.random import RandomState
from .lib import zigg

def z_standard_normal(num, random_state=None):
    """
    Standard normal float32s by the Ziggurat method
    """

    # Extra randoms samples becuase the Ziggurat rejects some
    zig_fac = 0.025

    if random_state is None:
        random_state = RandomState(seed=123)

    # space to store the randoms
    out = empty(num, dtype=float32)
    i0 = 0

    while i0<num:
        n_needed = num-i0

        # How many random integers do we need for n_needed random normals?
        pad = n_needed*zig_fac + (n_needed*zig_fac * (1-zig_fac))**0.5 # mean + binomial estimate for 1 std. dev. (c.f. ignore (1-zf) and get Poisson est.)
        guess_n_rand = int(n_needed + pad +1) 
        if i0>0:
            print('Not enough random integers for normals, increasing')
            print('Need {:,} random numbers'.format(n_needed), 'guess {:,}'.format(guess_n_rand), 'i0=',i0)
        
        ri = frombuffer(random_state.bytes(4*guess_n_rand), dtype=uint32)

        # consume ri as needed
        res = zigg(n_needed, ri)

        out[i0:i0+res.size] = res
        i0 += res.size
        
    return out

def z_complex_normal(size, random_state=None):
    """ make size complex normals """
    v = z_standard_normal(size*2, random_state)
    v = v.view(complex64)
    v *= 0.5**0.5 # <re(z)**2 + im(z)**2> = 1
    return v

def speed_test():
    from scipy.special import ndtri # inverse cumulative normal
    from time import time

    import numpy as np
    n_bin = 1000
    bins = ndtri((np.arange(n_bin-1)+1)/float(n_bin))
    
    random_state = RandomState(seed=123)
    z = z_standard_normal(10)
    t0 = time()
    z = z_standard_normal(10000000, random_state)
    t1 = time()
    print('Time to generate {:,} random normals'.format(len(z)), '%.3fs'%(t1-t0))
    z_bin = np.bincount(np.digitize(z, bins), minlength=n_bin)


    print('Mean', z.mean(), 'variance', z.var())
    print('Bin counts in', z_bin.min(), z_bin.max())
    bin_low, bin_high = np.argmin(z_bin), np.argmax(z_bin)
    print('Lowest bin %d in i=%d, max %d in %d'%(z_bin[bin_low], bin_low, z_bin[bin_high], bin_high))

#    import pylab as pl
#    pl.plot(z_bin)
#    pl.show()
    t0 = time()
    z = random_state.standard_normal(z.size)
    t1 = time()
    print('Time to generate {:,} random normals'.format(len(z)), '%.3fs'%(t1-t0))
    print('Mean', z.mean(), 'variance', z.var())    
    z_bin = np.bincount(np.digitize(z, bins), minlength=n_bin)
    print('Bin counts in', z_bin.min(), z_bin.max())
    bin_low, bin_high = np.argmin(z_bin), np.argmax(z_bin)
    print('Lowest bin %d in i=%d, max %d in %d'%(z_bin[bin_low], bin_low, z_bin[bin_high], bin_high))

if __name__=='__main__':
    speed_test()
