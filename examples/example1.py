
from smerfs.lib import chyp_c_single_series, chyp_c
from scipy.special import gamma
from numpy import sqrt, array, pi, sin

def reduce_m(llp1, m_top, m_req, z):
    m_begin = m_top - 2
    Fm2 = chyp_c_single_series(llp1, m_begin+2, z)
    Fm1 = chyp_c_single_series(llp1, m_begin+1, z)
    print('F[m=%d] ='%(m_begin+2), Fm2)
    print('F[m=%d] ='%(m_begin+1), Fm1)    
    
    last = [Fm1, Fm2]
    for m in range(m_begin, m_req-1, -1):
        Fm1, Fm2 = last
        Fm_rec = ((1-2*z)/(1-z))*Fm1 + (1 - llp1/((m+2)*(m+1))) * (z/(1-z)) * Fm2
        last = [Fm_rec, Fm1]
    Fm_rec = last[0]
    print('F[m=%d] ='%(m_req), Fm_rec)        
    return Fm_rec

def eval_Fm_goursat(llp1, m, z):
    """ Goursat quadratic transform """
    c = m+1    

    a = 0.5 -1.0j * sqrt(-0.25 - llp1) # - lam

    s = 1-2*z

    prefac = sqrt(pi)
    t0 = 1
#    t1 = -2*s/(gamma(0.5 * (m+1-a))*gamma(0.5*(m+a)))    
    t1 = -2*s

    if m % 2 == 1:
        # Odd
        prefac *= (1-z)
        t0 /= (gamma(1.5 - 0.5 *a)*gamma(1+0.5*a))
        t1 /= gamma(1 -0.5*a) * gamma(0.5 + 0.5*a)        
        for n in range(m//2):
            rho = 4*(1-z)*(1-z)*(2*n+2)*(2*n+3)
            t0 *= rho / ((2*n+3)*(2*n+2) - llp1)
            t1 *= rho / ((2*n+1)*(2*n+2) - llp1)

    else:
        # Even
        t0 /=gamma(1-0.5*a)*gamma(0.5*(1+a))
        t1 /= gamma(0.5 * (1-a))*gamma(0.5*a)
        for n in range(m//2):
            rho = 4*(1-z)*(1-z)*(2*n+1)*(2*n+2)
            t0 *= rho / ((2*n+2)*(2*n +1)-llp1)
            t1 *= rho / (2*n*(2*n+1) + -llp1)
    t0 *= prefac
    t1 *= prefac    


    largest_t = [0,t0]
    if abs(t1) > abs(largest_t[1]):
        largest_t = [0,t1]
        
    res = t0 + t1

    for n in range(1,100000):
        t0 = t0 * ((m+2*n-1)*(m+2*n-2)-llp1)*s*s/(2*(2*n-1)*n)                
        t1 = t1 * ((m+2*n)*(m+2*n-1)-llp1)*s*s/(2*(2*n+1)*n)                

        if abs(t0) > abs(largest_t[1]):
            largest_t = [n,t0]
        
        if abs(t1) > abs(largest_t[1]):
            largest_t = [n,t1]
        
        res += t0 + t1
    print('Goursat largest term', largest_t[1], 'at n=%d'%largest_t[0], '(absolute value %e)'%abs(largest_t[1]),
          'guessed', 0.25*m*s*s/(z*(1-z)), '+', 0.25*abs(s)*sqrt(abs(llp1)/(z*(1-z))), 'or', 0.25*abs(s)/sqrt(z*(1-z)) * (m + sqrt(abs(llp1))))
    return res


m = 23
z = 0.91
llp1 = 0.0 + 3000.0j # around 3000j for NASA case
# 3 term recurrence

Fm_rec = reduce_m(llp1, m+4, m, z)
Fm_direct = chyp_c_single_series(llp1, m, z)
Fm_goursat = eval_Fm_goursat(llp1, m, z)
Fm_poles = chyp_c(llp1, m, z) # only this for z>0.8
print("Guess direct max term at n=", sqrt(z*abs(llp1)/(1-z)))
print("Direct evaluation", Fm_direct)
print("Goursat          ", array([Fm_goursat]))
print("Recurrence       ", Fm_rec)
print("Pole summation   ", Fm_poles)
    
exit(0)

from smerfs import build_filter

import numpy as np
import matplotlib.pyplot as pl
nz = 128*8 # 128 points equally spaced in theta (z=cos(theta))
nphi = 256*8 # I usually use nphi=2*nz so regular pixels at equator
# GRF with C_lambda = 1/(1.0 + 10^-4 (lambda(lambda+1))^2)
coeffs = (1.0, 0.0, 1e-7) # this has length scale around l=10 
sf = build_filter(nz=nz, nphi=nphi, coeffs=coeffs, dtype=np.float64) # Build the filter coefficients
# Make an example realisation
res = sf.create_realisation() # has shape (nz, nphi)
#pl.imshow(res) # Plot
#pl.savefig('res.png', dpi=100)

from smerfs.utils import analytic_cov
ntheta = 100
theta = np.linspace(0, 4*np.pi/180, ntheta)
z = np.cos(theta)
correl = analytic_cov(coeffs, z) # Contributions from Legendre polynomials
pl.plot(theta*180/np.pi, correl)
pl.xlabel(r'$\theta\; ({\rm deg})$')
pl.ylabel(r'$C(\cos \theta)$')
#pl.xlim(1,-1)
#pl.savefig('correl1.png',dpi=100)
pl.show()



