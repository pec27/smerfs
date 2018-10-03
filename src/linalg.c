/*
  Linear algebra (for when numpy.linalg is too slow)
 */
#include <math.h>

int inverse(const int N, const int M, const double *restrict matrices, double *restrict out)
{
  /*
    Inverse of N symmetric MxM matrices  

    where M=2 or 3
   */
  if (M==2)
    for (int i=0;i<N;++i, matrices+=4, out+=4)
      {
	const double a=matrices[0], b=matrices[1], d=matrices[3];
	
	const double det = a*d - b*b;
	if (det==0)
	  return i+1; // Couldnt invert matrix i
	const double inv_det = 1.0/det;
	
	out[0] = d*inv_det;
	out[2] = out[1] = -b*inv_det;
	out[3] = a*inv_det;
      }
  else if (M==3)
    for (int i=0;i<N;++i, matrices+=9, out+=9)
      {
	const double a11=matrices[0], a12=matrices[1], a13=matrices[2],
	  a22=matrices[4], a23=matrices[5], a33=matrices[8];
	  
	const double u1 = a33*a22 - a23*a23,
	  u2 = a23*a13 - a33*a12,
	  u3 = a23*a12 - a22*a13;
	const double det = a11*u1 + a12*u2 + a13*u3;
	if (det==0)
	  return i+1; // Couldnt invert matrix i
	const double inv_det = 1.0/det;
	
	out[0] = inv_det * u1;
	out[3] = out[1] = inv_det * u2;
	out[4] = inv_det * (a33*a11-a13*a13);
	out[6] = out[2] = inv_det * u3;
	out[7] = out[5] = inv_det * (a12*a13 - a23*a11);
	out[8] = inv_det * (a22*a11-a12*a12);
      }
  else
    return -1; // Matrix size not supported

  return 0;
}

static inline int cholesky22(const double a11, const double a12, const double a22, double *restrict out)
{
  // Cholesky of symmetric 2x2 matrix
  const double b2 = a12*a12/a11, c2= a22-b2;
  
  if ((a11<0) | (c2<0))
    return 1; // Not positive def.
  
  out[0] = sqrt(a11);
  out[1] = 0.0;
  out[2] = a12/out[0];
  out[3] = sqrt(c2);

  return 0;
}

static inline int cholesky33(const double a11, const double a12, const double a13,
			     const double a22, const double a23, const double a33,
			     double *restrict out)
{
  // Cholesky of symmetric 3x3 matrix

  const double inv11 = 1.0/a11,
    r = a22 - a12*a12*inv11,
    inv_r = 1.0/r,
    tau = a23-a13*a12*inv11,
    mu = a33 - a13*a13 *inv11 - tau*tau*inv_r;

  if ((a11<=0) | (r<=0) | (mu<=0))
    return 1;

  out[0] = sqrt(a11);
  out[4] = sqrt(r);
  out[8] = sqrt(mu);

  out[5] = out[2] = out[1] = 0.0;

  out[3] = a12 * out[0] * inv11;
  out[6] = a13 * out[0] * inv11;
  out[7] = tau * out[4] * inv_r;

  return 0;
}

int cholesky(const int N, const int M, const double *restrict matrices, double *restrict out)
{
  /*
    Cholesky (lower) of N symmetric MxM matrices  
    where M=2

    N        - number of matrices
    M        - size of matrix (MxM)
    matrices - (N*M*M) array of doubles for matrices
    out      - (N*M*M) array of double for outputs (lower triangles)

    returns 0 on success, i+1 if i-th matrix failed, -1 if M unsupported


   */
  if (M==2)
    for (int i=0;i<N;++i, matrices+=4, out+=4)
      {
	if (cholesky22(matrices[0], matrices[1], matrices[3], out))
	  return i+1; // Not positive def.
      }
  else 
    return -1;

  return 0;
}

int state_space(const int N, const int M, 
		const double *restrict cross_cov, const double *restrict cov, 
		double *restrict innov, double *restrict trans)
{
  /*
    Construct the innovation and transition matrices in the 2x2 case
   */
  if (M==2)
    {
      // First innovation matrix is just copy of cov
      if (cholesky22(cov[0], cov[1], cov[3], innov))
	return 1;

      innov += 4; 

      for (int i=1;i<N;++i, cross_cov+=4, innov+=4, trans+=4)
	{

	  
	  const double cov11=cov[0], cov12=cov[1], cov22=cov[3];
	
	  const double det = cov11*cov22 - cov12*cov12;
	  if (det==0)
	    return i+1; // Couldnt invert matrix i
	  const double inv_det = 1.0/det;
	  // Inverse of covariance i
	  const double icov11 = cov22*inv_det,
	    icov12 = -cov12*inv_det,
	    icov22 = cov11*inv_det;
	  
	  const double cross11 = cross_cov[0], cross12 = cross_cov[1],
	    cross21 = cross_cov[2], cross22 = cross_cov[3];

	  trans[0] = cross11 * icov11 + cross12 * icov12;
	  trans[1] = cross11 * icov12 + cross12 * icov22;// recall icov21=icov12
	  trans[2] = cross21 * icov11 + cross22 * icov12; 
	  trans[3] = cross21 * icov12 + cross22 * icov22; // recall icov21=icov12
	  
	  cov += 4;

	  // BB^T (symmetric)
	  const double bb11 = cov[0] - trans[0] * cross11 - trans[1]*cross12,
	    bb12 = cov[1] - trans[0]*cross21 - trans[1]*cross22,
	    bb22 = cov[3] - trans[2]*cross21 - trans[3]*cross22;

	  // innov = cholesky(BB)
	  if (cholesky22(bb11, bb12, bb22, innov))
	    return i+1; // Not positive def.

	}
      }
  else  if (M==3)
    {
      // First innovation matrix is just copy of cov
      if (cholesky33(cov[0], cov[1], cov[2], cov[4], cov[5], cov[8], innov))
	return 1;

      innov += 9; 

      for (int i=1;i<N;++i, cross_cov+=9, innov+=9, trans+=9)
	{

	  
	  const double cov11=cov[0], cov12=cov[1], cov13=cov[2],
	    cov22=cov[4], cov23=cov[5], cov33=cov[8];


	
	const double u1 = cov33*cov22 - cov23*cov23,
	  u2 = cov23*cov13 - cov33*cov12,
	  u3 = cov23*cov12 - cov22*cov13;
	const double det = cov11*u1 + cov12*u2 + cov13*u3;
	if (det==0)
	  return i+1; // Couldnt invert matrix i
	const double inv_det = 1.0/det;
	
	const double icov11 = inv_det * u1,
	  icov12 = inv_det * u2,
	  icov22 = inv_det * (cov33*cov11 - cov13*cov13),
	  icov13 = inv_det * u3,
	  icov23 = inv_det * (cov12*cov13 - cov23*cov11),
	  icov33 = inv_det * (cov22*cov11-cov12*cov12);

	const double cross11 = cross_cov[0], cross12 = cross_cov[1], cross13 = cross_cov[2],
	  cross21 = cross_cov[3], cross22 = cross_cov[4], cross23 = cross_cov[5],
	  cross31 = cross_cov[6], cross32 = cross_cov[7], cross33 = cross_cov[8];

	  trans[0] = cross11 * icov11 + cross12 * icov12 + cross13 * icov13;
	  trans[1] = cross11 * icov12 + cross12 * icov22 + cross13 * icov23;
	  trans[2] = cross11 * icov13 + cross12 * icov23 + cross13 * icov33;

	  trans[3] = cross21 * icov11 + cross22 * icov12 + cross23 * icov13;
	  trans[4] = cross21 * icov12 + cross22 * icov22 + cross23 * icov23;
	  trans[5] = cross21 * icov13 + cross22 * icov23 + cross23 * icov33;

	  trans[6] = cross31 * icov11 + cross32 * icov12 + cross33 * icov13;
	  trans[7] = cross31 * icov12 + cross32 * icov22 + cross33 * icov23;
	  trans[8] = cross31 * icov13 + cross32 * icov23 + cross33 * icov33;
	  
	  cov += 9;

	  // BB^T (symmetric)
	  const double bb11 = cov[0] - trans[0] * cross11 - trans[1]*cross12 - trans[2]*cross13,
	    bb12 = cov[1] - trans[0]*cross21 - trans[1]*cross22 - trans[2]*cross23,
	    bb13 = cov[2] - trans[0]*cross31 - trans[1]*cross32 - trans[2]*cross33,
	    bb22 = cov[4] - trans[3]*cross21 - trans[4]*cross22 - trans[5]*cross23,
	    bb23 = cov[5] - trans[3]*cross31 - trans[4]*cross32 - trans[5]*cross33,
	    bb33 = cov[8] - trans[6]*cross31 - trans[7]*cross32 - trans[8]*cross33;


	  // innov = cholesky(BB)
	  if (cholesky33(bb11, bb12, bb13, bb22, bb23, bb33, innov))
	    return i+1; // Not positive def.

	}
      }
  else 
    return -1;

  return 0;



}
