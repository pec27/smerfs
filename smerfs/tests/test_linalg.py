""" testing the linear algebra """
from __future__ import print_function, division, unicode_literals, absolute_import
import numpy as np
from numpy.linalg import inv, cholesky
from smerfs.lib import inv_sym, cho
from smerfs.condition import _state_space_innovations_all

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

def test_state_space22():
    """ C-implementation of state space """

    cov = np.array((((1.5,1e-3),(1e-3,0.4)), ((0.3,2e-3), (2e-3,1.1))), dtype=np.float64)
    cross_cov = np.array((((0.1,0.01),(-0.02,0.1)),), dtype=np.float64)


    trans, innov = _state_space_innovations_all(cov, cross_cov, default_c=False)

    print('trans', trans)
    print('innov', innov)

    trans_c, innov_c = _state_space_innovations_all(cov, cross_cov)

    diff_trans = trans - trans_c
    diff_innov = innov - innov_c

    max_trans_err = max(np.abs(diff_trans.ravel()))
    max_innov_err = max(np.abs(diff_innov.ravel()))

    print('trans_c', trans_c)
    print('innov_c', innov_c)

    print('trans', max_trans_err)
    print('innov', max_innov_err)

    assert(max_trans_err<1e-13)
    assert(max_innov_err<1e-13)

def test_state_space33():
    """ C-implementation of state space for 3x3 """

    cov = np.array((((1.5,1e-3,0.0003),(1e-3,0.4,0.03), (0.0003,0.03, 1.0)), 
                    ((0.3,2e-3,0.005), (2e-3,1.1,-0.01), (0.005,-0.01,1.0))), dtype=np.float64)

    cross_cov = np.array((((0.1,0.01,0.0),
                           (-0.02,0.1,0.0), 
                           (0.0,0.0,0.5)),), dtype=np.float64)


    trans, innov = _state_space_innovations_all(cov, cross_cov, default_c=False)

    print('trans', trans)
    print('innov', innov)

    trans_c, innov_c = _state_space_innovations_all(cov, cross_cov)

    diff_trans = trans - trans_c
    diff_innov = innov - innov_c

    max_trans_err = max(np.abs(diff_trans.ravel()))
    max_innov_err = max(np.abs(diff_innov.ravel()))

    print('trans_c', trans_c)
    print('innov_c', innov_c)

    print('trans', max_trans_err)
    print('innov', max_innov_err)

    print('diff trans', diff_trans)
    print('diff innov', diff_innov)

    assert(max_trans_err<1e-13)
    assert(max_innov_err<1e-13)


