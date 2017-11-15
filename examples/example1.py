from smerfs import build_filter

import numpy as np
import matplotlib.pyplot as pl
nz = 128 # 128 points equally spaced in theta (z=cos(theta))
nphi = 256 # I usually use nphi=2*nz so regular pixels at equator
# GRF with C_lambda = 1/(1.0 + 10^-4 (lambda(lambda+1))^2)
coeffs = (1.0, 0.0, 1e-4) # this has length scale around l=10 
sf = build_filter(nz=nz, nphi=nphi, coeffs=coeffs, dtype=np.float64) # Build the filter coefficients
# Make an example realisation
res = sf.create_realisation() # has shape (nz, nphi)
#pl.imshow(res) # Plot
#pl.savefig('res.png', dpi=100)

from smerfs.utils import analytic_cov
z = np.linspace(-1,1, 1000)
correl = analytic_cov(coeffs, z) # Contributions from Legendre polynomials
pl.plot(z, correl)
pl.xlabel(r'$\cos \theta$')
pl.ylabel(r'$C(\cos \theta)$')
pl.xlim(1,-1)
pl.savefig('correl1.png',dpi=100)




