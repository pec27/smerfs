
from smerfs.lib import chyp_c_single_series, chyp_c
    

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

m = 100
z = 0.801
llp1 = 0.0 + 300.0j
# 3 term recurrence

Fm_rec = reduce_m(llp1, m+4, m, z)
Fm_direct = chyp_c_single_series(llp1, m, z)
print("Direct evaluation", Fm_direct)
print("Recurrence       ", Fm_rec)
print(chyp_c(llp1, m, z))
    
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



