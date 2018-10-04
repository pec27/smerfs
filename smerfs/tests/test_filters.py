from __future__ import print_function, division, unicode_literals, absolute_import
from smerfs.filters import build_filter, load_filter
import numpy as np
import io


def test_build():
    """ Test build of a small filter """
    nz = 8 # 128 points equally spaced in theta (z=cos(theta))
    nphi = 16 # I usually use nphi=2*nz so regular pixels at equator
    # GRF with C_lambda = 1/(1.0 + 10^-4 (lambda(lambda+1))^2)
    coeffs = (1.0, 0.0, 1e-4) # this has length scale around l=10 
    sf = build_filter(nz=nz, nphi=nphi, coeffs=coeffs, dtype=np.float64) # Build the filter coefficients
    
    sf.create_realisation()
    trans, innovs = sf._trans, sf._jumps

    file_obj = io.BytesIO() # buffer to save in
    sf.save(file_obj)
    file_obj.seek(0) # go back to start...

    sf = load_filter(file_obj)
    print(trans.shape)
    assert(sf._nz==nz)
    assert(sf._nphi==nphi)
    assert(sf.coeffs==coeffs)
    assert(np.all(np.equal(trans, sf._trans)))
    assert(np.all(np.equal(innovs, sf._jumps)))
    
    print(sf)


    

