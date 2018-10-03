#include <complex.h>
#include <math.h>
#include <stdlib.h>

int update_cov(const int m_max, const int N, const int M, 
	       const double norm_re, const double norm_im, 
	       const double llp1_re, const double llp1_im,
	       const double complex *restrict F, const double complex *restrict H,
	       const double *restrict tau_power, const double *restrict eta_ratio2,
	       double *restrict cov, double *restrict cross_cov)
{
  /*
    Update the covariance (cov) and cross-covariance matrices with a single term 
    from the partial fraction decomposition (i.e. single norm and l(l+1)).

    m_max     - final m term (i.e. 0...m_max)
    N         - Number of z points
    M         - number of terms in partial fraction decomposition
    norm_re   - real(norm)
    norm_im   - imag(norm)
    llp1_re   - real(l(l+1))
    llp1_im   - imag(l(l+1))
    F         - (m_max+M+1, N) complex array of 2_F_1 (l,l+1;1+m;(1-z)/2)
    H         - (m_max+M+1, N) complex array of 2_F_1 (l,l+1;1+m;(1+z)/2)
    tau_power - (2M-1, N) double array of tau^p, with p 
                from 1-M...M-1 where tau_i = ((1-z_i)/(1+z_i))^(1/2)
    eta_ratio - (N-1,) double array of tau_{i+1} / tau_i
    updating:

    cov       - (m_max+1, N, M, M) array of covariances
    cross_cov - (m_max+1, N-1, M, M) array of cross-covariances

    returns 0 on success, -1 on out-of-memory
    
   */
  const int n_cross = N-1;
  double complex zeta[M], Rc[M], zetaA[M], zAH[M];
  double complex norm = norm_re + I*norm_im;
  const double complex llp1 = llp1_re + I*llp1_im;

  double *eta = (double *)malloc(n_cross * sizeof(double));

  if (!eta)
    return -1;

  for (int i=0;i<n_cross;++i)
    eta[i] = 1.0;

  for (int m=0,mN=0;m<=m_max;++m, mN+=N)
    {
	  
      // zeta symbols
      zeta[0] = 1.0;
      zeta[1] = m -llp1/(m+1); // First zeta

      for (int i=2,v=m+2;i<M;++i,++v)
	{
	  const double complex mult = (v*(v-1) - llp1)/v;
          zeta[i] = zeta[i-1]*mult;
	}



      for (int p=0;p<M;++p)
	zetaA[p] = (p&1) ? -zeta[p] : zeta[p];

      for (int p=0, t_idx = (M-1)*N;p<M;++p, t_idx-=N)
	zAH[p] = zetaA[p] * H[(m+p)*N] * tau_power[t_idx];

      
      // Pre-multiply LHS by the norm
      for (int p=0;p<M;++p)
	zeta[p] *= norm;

      // First covariance
      for (int p=0,F_idx=mN,t_idx=(M-1)*N;p<M;++p,F_idx+=N, t_idx+=N)
	{
	  const double complex Lp = tau_power[t_idx] * (zeta[p] * F[F_idx]); 
      
	  for (int q=0;q<M;++q)
	    {
	      const double complex Rq = zAH[q];
	      // Update real part
	      *cov++ += creal(Lp) * creal(Rq) - cimag(Lp)*cimag(Rq);
	    }
	}

      // Covariance and cross
      for (int n=0,tau0=(M-1)*N+1;n<n_cross;++n, tau0++)
	{

	  for (int p=0, t_idx = tau0;p<M;++p, t_idx-=N)
	    {
	      Rc[p] = eta[n] * zAH[p];
	      zAH[p] = zetaA[p] * H[(m+p)*N+n+1] * tau_power[t_idx];
	    }	    


	  for (int p=0,F_idx=mN+n+1,t_idx=tau0;p<M;++p,F_idx+=N, t_idx+=N)
	    {
	      const double complex Lp = tau_power[t_idx] * (zeta[p] * F[F_idx]); 
	      for (int q=0;q<M;++q)
		{
		  // Update with real parts
		  *cov++ += creal(Lp) * creal(zAH[q]) - cimag(Lp)*cimag(zAH[q]);
		  *cross_cov++ += creal(Lp)*creal(Rc[q]) - cimag(Lp)*cimag(Rc[q]);
		}
	    }
	  
	  eta[n] *= eta_ratio2[n]; //      eta *= tau[1:] / tau[:-1]
	}

      norm *= ((m+1)*m - llp1)/((m+1)*(m+1)); // norm for next m
      
    }

  free(eta);
  return 0;
}
