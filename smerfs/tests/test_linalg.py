""" testing the linear algebra """
from __future__ import print_function, division, unicode_literals, absolute_import
import numpy as np
from numpy.linalg import inv, cholesky
from smerfs.lib import inv_sym, cho

def test_inv2x2():
    """ Test inverses of symmetric 2x2 matrices """
    a = np.array((((2,0.1),(0.1,0.8)), ((-2,0.1),(0.1,0.81))))

    np_inv = inv(a)
    lib_inv = inv_sym(a)

    err = np_inv-lib_inv
    max_err = max(np.abs(err).ravel())
    
    print('inv', np_inv)
    
    print(lib_inv)
    print('Max err', max_err)
    assert(max_err<1e-16)

def test_inv3x3():
    """ inverses of symmetric 3x3 """
    a = np.array((((2,0.1,-0.2),(0.1,1.2,0.05),(-0.2,0.05,0.8)),))

    np_inv = inv(a)
    lib_inv = inv_sym(a)

    err = np_inv-lib_inv
    max_err = max(np.abs(err).ravel())
    
    print('inv', inv(a))
    
    print(lib_inv)
    print('Max err', max_err)
    assert(max_err<1e-15)

def test_cholesky():
    """ Test cholesky of symmetric 2x2 """

    a = np.array((((2,0.1),(0.1,0.8)), ((2,-0.1),(-0.1,0.81))))

    np_cho = cholesky(a)
    lib_cho = cho(a)

    err = np_cho-lib_cho
    max_err = max(np.abs(err).ravel())
    
    print('cho', np_cho)
    
    print(lib_cho)
    print('Max err', max_err)
    assert(max_err<1e-16)
