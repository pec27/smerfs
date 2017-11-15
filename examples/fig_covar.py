from numpy import *
from numpy.random import RandomState
from smerfs import cov_covar
from smerfs.condition import _state_space_innovations_all


def build_path(nz,m,coeffs, rs):


    if nz%2==1:
        raise Exception('nz must be even!')

    theta = (arange(nz/2+1)[::-1]+0.5) * (pi/nz)
    z_pts = cos(theta)
    max_z = len(z_pts)

    cov, cross_cov = cov_covar(z_pts[:max_z], m, coeffs) # hypergeometric function decomp

    print 'Covariance shape',cov.shape
    # Arrange the arrays so we can use numpy to simultaneously do the state-space transitions
    cov = transpose(cov[m].real, (2,0,1))
    cross_cov = transpose(cross_cov[m].real, (2,1,0))
    tran, innov = _state_space_innovations_all(cov, cross_cov)
        
    
    # Make the walk for a single m
    print innov.shape
    n_innov = innov.shape[0]
    nmat = innov.shape[1]
    

    # white noise (mean square magnitue=1.0, real and imaginary 0.5 each)
    if m>0:
        white = (rs.standard_normal(nmat*n_innov) + 1.0j*rs.standard_normal(nmat*n_innov)) * sqrt(0.5)
    else:
        white = rs.standard_normal(nmat*n_innov)
    white.shape  = (n_innov, nmat)

    # Make the random walk
    pt = inner(innov[0], white[0])
    pts = [pt]
    for i in range(tran.shape[0]):
        pt = inner(tran[i],pt) + inner(innov[i+1],white[i+1])
        pts.append(pt)

    # and then walk in the other direction (one less step since we start just below the equator, and we dont need the step to initialise also)
    n_back = n_innov-2
    if m>0:
        white = (rs.standard_normal(nmat*n_back) + 1.0j*rs.standard_normal(nmat*n_back)) * sqrt(0.5)
    else:
        white = rs.standard_normal(nmat*n_back)

    white.shape  = (n_back, nmat)

    # Make the random walk
    back = array([(-1)**i for i in range(nmat)])  # odd derivatives change sign when going backwards
    pt = pts[0]* back
    pts = list(reversed(pts))

    for i in range(n_back):
        pt = inner(tran[i+1],pt) + inner(innov[i+2],white[i])
        pts.append(pt*back)

    pts = list(reversed(pts))
    # make cos(theta) over -pi to -pi
    cos_t = list(reversed(-z_pts[1:])) + list(z_pts[1:])
    theta = (arange(nz)+0.5) * (180.0/nz)
    print len(cos_t),'cos_t'

    g = array([pt[0] for pt in pts])
    # Expecten mean square magnitude of g
    var_g = array(list(reversed(cov[1:,0,0]))+list(cov[1:,0,0]))
    return g, var_g, cos_t, theta

def test_var(nz,m,coeffs,samples):
    """ test that the variance is what is predicted """
    
    rs = RandomState(seed=1)


    g, var_g, cos_t, theta = build_path(nz,m,coeffs, rs)
    # hold the variance of |g|
    var_g_est = (g*g.conj()).real
    
    for i in range(samples-1): #already did 1st sample
        g, var_g, cos_t, theta = build_path(nz,m,coeffs, rs)
    
        var_g_est += (g*g.conj()).real
        
    var_g_est *= 1.0/samples
    
    
        

    import pylab as pl
    pl.plot(theta,var_g_est/var_g,label='sampled var/var')
    pl.axhline(1.0,color='k', ls='-')
    # error on the ratio
    rat_error = sqrt(1.0/samples) # I THINK THIS IS WRONG, check!!
    pl.axhline(1.0+rat_error,color='0.5',ls=':')
    pl.axhline(1.0-rat_error,color='0.5',ls=':')
    print 'Testing my var calc', sqrt(square(var_g_est/var_g - 1).mean()), rat_error
    pl.legend(frameon=False)

    pl.xlim(-90,90)
    pl.show()
    
def fig(nz,coeffs,name=None):
    rs = RandomState(seed=3)

    ticks = [0,30,60,90,120,150,180]
    import pylab as pl
    from iccpy.figures import latexParams
    pl.rcParams.update(latexParams)
    
    pl.figure(figsize=(6.64, 2.8))
    for i,m in enumerate([0,5]):
        g, var_g, cos_t, theta = build_path(nz,m,coeffs, rs)

        
        # ignore derivatives and imaginary part
        re_g = g.real
        print len(g), 'g'

        if m!=0:
            std_re_g = sqrt(var_g*0.5) # standard deviation of real part of g
        else:
            std_re_g = sqrt(var_g) # g_0 real

        ax =pl.subplot(211+i)
        ax.plot(theta,re_g, 'k')
        ax.fill_between(theta, -std_re_g, std_re_g, color='0.8')

        pl.xlim(0,180)

        ax.xaxis.set_ticks(ticks)
        ax.xaxis.set_ticklabels([r'$%d^\circ$'%val for val in ticks])
        if m>0:
            pl.ylabel(r'$\Re \left[ g_%d (\cos \theta )\right]$'%m)
        else:
            pl.ylabel(r'$g_0 (\cos \theta )$')
    pl.xlabel(r'$\theta$')
    if name is None:
        pl.show()
    else:
        pl.savefig(name)
    
if __name__=='__main__':
#    fig(nz=30000,coeffs=(10.0, 0.0, 1.0), name='sample_walk_10.pdf')
    # test if rougher coeffs still have E[|g_m(z)|^2]=0 at z=+/- 
#    fig(nz=300,coeffs=(1.0, 1e-3))
    test_var(nz=50,m=0,coeffs=(10.0, 0.0, 1.0), samples=300)
