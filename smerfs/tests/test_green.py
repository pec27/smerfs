""" testing the Green's function parts """
from __future__ import print_function, division, unicode_literals, absolute_import
from .utils import *

def test_wronskian():
    """ 
    Wronskian formula for W{P^m_lam(x), P^m_lam(-x)}
    """

    for m,lam, x in [(3,0.75+0.1j,0.4), (0,0.75+0.1j,0.4), (25,0.75+0.1j,0.1)]:
        llp1 = lam*(lam+1)
        p = lambda x : P_m_llp1(m, llp1, x)  # P^m_lam(x)
        pm = lambda x : p(-x) # P^m_lam(x)

        # numerical Wronskian
        w_num = p(x) * num_deriv(pm,x) - num_deriv(p,x) * pm(x)

        # analytic formula
        w = 2.0 / ((1-x*x) * gamma(-lam-m) * gamma(lam+1-m))

        err = abs(w_num-w) / abs(w)
        

        print(w, w_num, err)
        assert(err<1e-7)
    

def test_Pml():
    """ 
    Generalised Legendre functions matches scipy at integer degree and order 
    """

    for m,lam, x in [(0,0,0.1), (0,4,0.4), (3,4,0.2),
                     (3,5,0.4), (2,5,0.4), (3,22,0.4)]:
        llp1 = lam*(lam+1)
        p = P_m_llp1(m, llp1, x)  # P^m_lam(x)
        p_scipy = lpmn(m, lam, x)[0][-1,-1] # Scipy version

        err = abs(p-p_scipy) / abs(p_scipy)
        
        if err>1e-9:
            print(p, p_scipy, err)
        print('m=%d l=%d x=%.5f'%(m,lam,x), 'error', err)
        assert(err<1e-6)

def test_green_jump():
    """
    Jump conditions for Greens function
    """
    dx = 1e-10
    for m,lam,x in [(3,0.75+0.1j,0.4), (0,0.75+0.1j,0.9), (25,0.75+0.1j,0.1)]:
        # Check
        # (-lam(lam+1) + L_m)G_m(x,y) = delta(x-y)
        # Where L_m[f] = ((1-x^2) f')' - m*m/(1-x*x)
        llp1 = lam*(lam+1)
        # Difference eqn of derivatives
        gp = lambda z: G_m_llp1(z+dx,x, m, llp1)
        gm = lambda z: G_m_llp1(z-dx,x, m, llp1)
        Ap = -num_deriv(gp, x, dx) * (1-x*x)
        Am = -num_deriv(gm, x, dx) * (1-x*x)
    
        jump = Ap-Am
        err = abs(jump-1.0)
        print('Jump condition', jump, 'err', err, 'expected 1.0')
        assert(err<1e-6)
        
        
def test_prefac():
    """
    Cancellation of Legendre function prefactors
    """

    # Note the gamma function expression fails beyond around m=100 due to too high exponents
    for m,lam in [(3,0.75+0.1j), (0,0.75+0.1j), (25,0.75+0.1j), (80,0.75+0.1j)]:
        # Check
        
        # 0.5 gamma(-lam-m)gamma(1+lam-m) * ((-lam)_m (lam+1)_m / m!)^2

        # = -pi/ 2sin(lam pi)  Prod_v=0^m-1 (v(v+1) - lam(lam+1)) / (1+v)^2
        llp1 = lam*(lam+1)

        w0 = 0.5*gamma(-lam-m)*gamma(1+lam-m) * square(pochhammer(0.0-lam,m)*pochhammer(lam+1,m) / gamma(m+1))


        w1 = -0.5*pi/sin(lam*pi)
        if m>0:
            v = np.arange(m)
            w1 *= cumprod((v*(v+1) - llp1)/square(1+v))[-1]
        err = abs(w1-w0)

        print('m=',m,'lam=',lam, 'w0', w0, 'w1', w1,'err', err)
        assert(err<1e-13)
        
def test_cov_symmetric_real():
    """
    Covariance matrices symmetric and real
    """

    for coeffs in [(1.0, 0.0, 0.1), (1.0, 0.0, 0.0, -0.1), (1.0, 0.0, 0.0, 0.0, 0.1)]:
        print('coeffs=',coeffs)
        for m in [0,3,6]:
            print('m=',m)
            for x in [-0.3, 0.5, 0.8]:

                cov = Jpq(m, coeffs, x, x)
                max_imag = abs(cov.ravel().imag).max()
                asym = abs(cov - cov.T).ravel().max()
                print('x=', x,'Asym', asym, 'Max imaginary part', max_imag)
                assert(asym<1e-10)
                assert(max_imag<1e-12)

def test_opt_cov():
    """
    Optimized covariance
    """

    for coeffs in [(1.0, 0.0, 0.1), (1.0, 0.0, 0.0, -0.1), (1.0, 0.0, 0.0, 0.0, 0.1)]:
        print('coeffs=',coeffs)
        for m in [0,3,6]:
            print('m=',m)
            for x in [-0.3, 0.5, 0.8]:

                covA = Jpq(m, coeffs, x, x)

                covO, covarO = cov_covar(array([x,x]), m+1, coeffs)
                cov = covO[m,:,:,0]
                
                d = abs(cov-covA).max()
                max_imag = abs(cov.ravel().imag).max()
                asym = abs(cov - cov.T).ravel().max()
                print('x=', x,covA.shape, cov.shape,'Asym', asym, 'Max imaginary part', max_imag, 'matrix difference', d, covA, 'new', cov)
                assert(d<1e-10)
                assert(asym<1e-10)
                assert(max_imag<1e-12)
    


def test_opt_cross_cov():
    """
    Optimized cross-covariance
    """

    for coeffs in [(1.0, 0.0, 0.1), (1.0, 0.0, 0.0, -0.1), (1.0, 0.0, 0.0, 0.0, 0.1)]:
        print('coeffs=',coeffs)
        for m in [0,3,6]:
            print('m=',m)
            xpts = [-0.3, 0.5, 0.8]
            for x,y in zip(xpts[:-1], xpts[1:]):

                covA = Jpq(m, coeffs, x, y)

                covO, covarO = cov_covar(array([x,y]), m+1, coeffs)
                cov = covarO[m,:,:,0]
                
                d = abs(cov-covA).max()
                max_imag = abs(cov.ravel().imag).max()

                print('x=', x,'y', y, 'Max imaginary part', max_imag, 'matrix difference', d,end='')
                assert(d<1e-12)
                assert(max_imag<1e-12)
    


