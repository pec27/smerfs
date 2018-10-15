"""
Test the random numbers
"""
from __future__ import print_function, division, unicode_literals, absolute_import
import numpy as np
from numpy.random import RandomState
from smerfs.random import z_standard_normal

def test_zig():
    """ Test the Ziggurat generator has approximately normal distribn """

    from scipy.special import ndtri # inverse cumulative normal

    rand_size =1000000
    n_bin = 1000
    bins = ndtri((np.arange(n_bin-1)+1)/float(n_bin))
    
    random_state = RandomState(seed=123)
    z = z_standard_normal(rand_size, random_state)
    z_bin = np.bincount(np.digitize(z, bins), minlength=n_bin)


    print('Mean', z.mean(), 'variance', z.var())
    print('Bin counts in', z_bin.min(), z_bin.max())
    bin_low, bin_high = np.argmin(z_bin), np.argmax(z_bin)
    print('Lowest bin %d in i=%d, max %d in %d'%(z_bin[bin_low], bin_low, z_bin[bin_high], bin_high))

    mean_bin = rand_size//n_bin

    over = z_bin[bin_high]-mean_bin
    under = mean_bin - z_bin[bin_low]

    assert(over<200)
    assert(under<200)

