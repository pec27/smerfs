"""
Build the coefficients for conditional sampling, e.g. g_m(z1) | g_m(z_1)
"""
from numpy import float64, arange, pi, cos, sin, empty, transpose, isfinite, dot
from time import time
from numpy.linalg import cholesky, inv, LinAlgError, det, eigvalsh
from cov import cov_covar

def _state_space_innovations(cov00, cov11, cov10):
    """
    Find the innovation matrix (J) and transition matrix (T)
    for a state space model

    X1 = dot(T,X0) + dot(J,W)

    where the W is an NxN matrix of complex standard normal variates and

    cov00_pq = E[X0_p* X0_q]
    cov10_pq = E[X1_p* X0_q]
    cov11_pq = E[X1_p* X1_q]

    where * denotes the complex conjugate.

    returns J, T
    """

    trans = dot(cov10, inv(cov00))
    
    A = cov11 - dot(cov10, trans.T)
    try:
        innov = cholesky(A)
    except LinAlgError:
        print 'Could not make cholesky decomposition of'
        print A
        raise
    return innov, trans

def _state_space_innovations_all(cov, cross_cov):
    """
    Like state-space innovations but for a whole array (n,m,m) of 
    covariances and cross-covariances
    """

    M = cov.shape[-1]
    n = cross_cov.shape[0]
    innov = cov.copy()

    

    # Build the transition matrix
    trans = inv(cov[:-1])
    trans.shape = (n,1,M,M)
    cross_cov.shape = (n,M,1,M)
    trans = (trans*cross_cov).sum(3)

    # Now the innovation matrix
    # The first step of the walk has no transition
    trans.shape = (n,1,M,M)
    innov[1:] -= (cross_cov * trans).sum(3)

    # restore the arrays to good shapes
    trans.shape = (n,M,M)
    cross_cov.shape = (n,M,M)

    # cholesky decomposition of innovation matrix
    innov = cholesky(innov)

    return trans, innov
    
def gm_walkz(nz, msize, coeffs, dtype=float64):
    """ 
    construct a filter for mode m as it walks down z in zpts, i.e.
    X[0] =             Q[0] W[0]
    X[1] = F[0] X[0] + Q[1] W[1]
    X[2] = F[1] X[1] + Q[2] W[2]
    etc.
    """

    M = len(coeffs)-1
    
    # Equidistant points in theta (start at first point below equator and work up)
    theta = (arange(nz//2+1)[::-1]+0.5) * (pi/nz)
    z_pts = cos(theta)
    sin_t = sin(theta)


    # Calculate the coefficients of the walk
    t_start = time()
    t0 = t_start

    # space to store all these matrices
    innovs = empty((msize,len(z_pts),M,M), dtype=dtype) 
    trans  = empty((msize,len(z_pts)-1,M,M), dtype=dtype) 

    print 'Building covariances',
    t0 = time()
    cov_all, cross_all = cov_covar(z_pts, msize-1, coeffs)
    t1 = time()
    print '[in %.3fs]'%(t1-t0)
    t0 = t1
    print 'Matrix decomposition',
    
    for m in range(msize):


        # Arrange the arrays so we can use numpy to simultaneously do the state-space transitions
        cov = transpose(cov_all[m].real, (2,0,1))
        cross_cov = transpose(cross_all[m].real, (2,1,0)) # having to transpose p and q, did I switch somewhere?

        try:
            tran, innov = _state_space_innovations_all(cov, cross_cov)
        except LinAlgError:
            # Must have failed the Cholesky (or inversion) of one of the matrices
            for i in range(len(cross_cov)):
                
                try:
                    _state_space_innovations(cov[i], cov[i+1], cross_cov[i])
                except LinAlgError:
                    print 'At m=%d, z[%d]=%f -> z[%d]=%f (of %d), failed to perform cholesky decomposition of matrix'%(m,i,z_pts[i], i+1,z_pts[i+1],len(cross_cov))
                    print 'Covariance 0\n',cov[i]
                    print 'Covariance 1\n',cov[i+1]
                    print 'Cross\n', cross_cov[i]
                    print 'Number of phi points (%d)'%(2*msize-2)
                    raise
            raise Exception('Failure in array of matrices but not individually. This should never happen.')
            raise
        innovs[m] = innov
        trans[m] = tran

        check = isfinite(innovs[m]).all() and isfinite(trans[m]).all()
        if not check:
            raise Exception('At m=%d infinite values in innovs/trans matrices (perhaps use dtype=float64?)'%m)

    t1 = time()
    dt, t0 = t1-t0, t1
    
    print '[in %.3fs]'%dt

    return trans, innovs

if __name__=='__main__':
    
    trans, innovs = gm_walkz(nz=512*1, msize=256*1, coeffs=(1.0,0.0,0.1))
