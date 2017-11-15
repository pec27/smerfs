/*
  SMERFS - Stochastic Markov Evaluation of Random Fields on the Sphere

  Peter Creasey 
*/
#include <stdlib.h>
#include <math.h>
#include <complex.h>
//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

#define MAX_ITERATIONS 10000 /* same as scipy */
#define EPS 1e-13
const double PI = 3.141592653589793;

static double complex cpsi(const double complex z)
{
  /*
    Digamma function for complex z.
    psi(z) := d/dz  log(Gamma(z))

    [Static, so only visible here]

   */
  // Coefficients for power series expansion
  static double a[8] = {-0.8333333333333e-1,0.83333333333333333e-2, -0.39682539682539683e-2,.41666666666666667e-2, -0.75757575757575758e-02,0.21092796092796093e-01, -0.83333333333333333e-01,0.4432598039215686};

  int n, i, neg_real;
  double complex z2, zn, psi, zs;

  neg_real = creal(z) < 0.0;

  if (neg_real)
    zs = -z;
  else
    zs = z;
 
  // Shift the zs to make real part > 8
  if (creal(zs) < 8.0)
    {
      // For small numbers use recurrence relation to make them larger than 8
      // e.g. Beal, M (2003)
      n = 8 - (int)creal(zs);
      zs += n;
    }
  else n = 0;

  psi = clog(zs) - 0.5/zs;

  // Expansion in z2
  z2 = zs*zs;
  zn = 1.0;
  for (i=0;i<8;i++)
    {
      zn /= z2;
      psi += a[i] * zn;
    }

  // Recurrence relation (undo the shift)
  for (i=1; i<n+1;i++)
    psi += 1/(i-zs);

  // Transform negative back
  if (neg_real)
    psi -= 1/z + PI / ctan(PI * z);

  return psi;

}
static double complex gamma_ratio(const double complex l, const int m)
{
  /*
    Calculate gamma(m+1) * gamma(m) / gamma(m-l) / gamma(m+l+1)
    
    l is complex
    m>0 an integer
    
    The straightforward method of doing this fails (out of range) for even
    reasonably large m,
    so we use the Pockhammer expansion and Eulers reflection formula
  */
  double complex C = csin(-PI * l)/PI; // Eulers reflection formula 
  const double complex llp1 = l * (l+1);
  int n;
  for (n=1;n<=m;n++)
    C *= (n*n)/(n*(n-1) - llp1); 

  return C/m;
}


double complex hyp_lmz(const double complex l, const int m, const double z, const double complex gamma_ratio0, const double complex psi0)
{
  /*
    Hypergeometric function  F(-l, l+1, 1+m,z) for large z (0.5<z<1)
    Using  AMS55 15.3.11

    llp1 := l(l+1) complex
    m integer
    z double
           Only needed if m>0
    gamma_ratio0 := gamma(m+1) * gamma(m) / gamma(m-l) / gamma(m+l+1) 

    psi0 := psi(1) + psi(1+m) - psi(m-l) -  psi(m+l+1) 
   */


  double complex poch, psi_term, res, term;
  const double complex llp1 = l*(l+1);
  const double s = 1-z;
  int n;


  res = 0.0;
  if (m>0)
    {
      // Now the constant term at the start of AMS55 15.3.11
      // Zeroth term gamma(m) * gamma(1+m) / (gamma(a+m) * gamma(b+m));
      term = gamma_ratio0;
      for (n=0; n<m;n++)
	{
	  res += term;
	  if ((fabs(creal(term)) < fabs(creal(res)) * EPS) && ((fabs(cimag(term)) < fabs(cimag(res)) * EPS)))
	    break;
	  term *= s * ((n+1)*n - llp1) / ((n+1) * (n+1-m));
	}
    }

  // Now the infinite series

  n=0; // Zero-th term
  psi_term = psi0 - log(s);
  poch = -csin(PI * l) / PI; // Euler's reflection formula
  if (m>0)
    poch *= pow(s, m);
  if (m&1)
    poch = -poch;

  
  term = poch * psi_term;

  res += term;
  
  // Higher terms...
  for (n=1; n<MAX_ITERATIONS;n++)
    {
      
      poch *= ((m+n-1) * (m+n) - llp1) * s / (n*(n+m));     // Use recursive definition for speed
      //psi_term = cpsi(n+1) + cpsi(n+m+1) - cpsi(n+m-l) -  cpsi(m+n+1+l) - log(1-z);
      psi_term += (2.0*n+m)/((n+m)*n) - (2*(n+m) - 1) / ((n+m)*(n+m-1) - llp1);
      term = poch * psi_term;
      /*
#ifdef DEBUG     
      printf("n=%d coeff ratio %e p+ %ej\n", n, creal(poch), cimag(poch));
      printf("n=%d coeff %e + %ej\n", n, creal(poch), cimag(poch));
      printf("n=%d psi_term %e + %ej\n", n, creal(psi_term), cimag(psi_term));
      printf("n=%d term %e + %ej\n", n, creal(term), cimag(term));
#endif
      */
      res += term;
      if ((fabs(creal(term)) < fabs(creal(res)) * EPS) && (fabs(cimag(term)) < fabs(cimag(res)) * EPS))
	return res;
    }
#ifdef DEBUG
  printf("Too many iterations (%d) at z=%f in psi-function expansion\n",MAX_ITERATIONS, z);
#endif
  return res;
}
static double complex hyp_llpz(const double complex llp1, const int m, const double z)
{
 /*

    Return the value of the Gauss hypergeometric function (2F1) with complex argument
    
    hyp2f1(-l, l+1, 1+m, z)
    where
    llp1 := l(l+1)

    Using the series definition (i.e. converges fast for 0 <= z <= 0.5)
    llp1 - a complex scalar*
    m - an integer >= 0
    z - real value 

 */

  double complex term, res;
  int q;

  term = 1.0;
  res = term;
  q = 1;
  
  do {
    term *= z * ((q*(q-1) - llp1) / (q*(m+q)));
    res += term;
    q++; 
  } while (((fabs(creal(term)) > fabs(creal(res) * EPS)) || (fabs(cimag(term)) > fabs(cimag(res) * EPS))) && (q<MAX_ITERATIONS));
  if (q==MAX_ITERATIONS)
    {
#ifdef DEBUG
      printf("Too many iterations (%d) at z[%d]=%f for m=%d\n", MAX_ITERATIONS, q, (float)z, m);
#endif
      exit(1);
    }
  return res;

}
int hyp_llp1(const double llp1_real, const double llp1_imag, const int m, const int nz, const double *zvals, double complex *out)
{
  /*
    Return the value of the Gauss hypergeometric function (2F1) with complex argument
    
    hyp2f1(-l, l+1, 1+m, z)
    where
    llp1 := l(l+1)


    llp1 - a complex scalar (note l is guaranteed not a positive integer)
    m - an integer >= 0
    z - array of values to calculate for (real)
    out - array of complex outputs (real, imag pairs)
   
    return 0 on success, 1 on failure (too many iterations)
   */
  int i;
  double complex llp1, l, gamma_ratio0, psi0;
#ifdef DEBUG
  int used_psi=0;
#endif

  llp1 = llp1_real + llp1_imag * I;
  const double alpha = (m+1)*(m+1)/cabs(llp1); // Determines ratio of coefficients
  double zcrit = 1.0 - 0.125*alpha; // When to turnover to psi-expansion
  if (alpha>10) zcrit=0.99;
  if (zcrit>0.99) zcrit=0.99;
  if (zcrit<0.75) zcrit = 0.75;
#ifdef DEBUG
  
  printf("z crit %f at m=%d, |l(l+1)|=%f \n",(float)zcrit, m, cabs(llp1));
  
#endif  
  
  if (creal(llp1)<-0.25) 
    l = -0.5 + I * csqrt(-0.25 - llp1);
  else 
    l = -0.5 - csqrt(0.25 + llp1);

  // Cache some vaiues for use with hyp_lmz
  if (m>0) gamma_ratio0 = gamma_ratio(l,m);

  psi0 = cpsi(1) + cpsi(1+m) - cpsi(m-l) -  cpsi(m+l+1);

  for (i=0; i<nz; ++i)
    {

      if (zvals[i]  < zcrit)
	{
	  // Use the series definition for small z
	  out[i] = hyp_llpz(llp1, m, zvals[i]);
	}
      else
	{
	  // Psi series expansion when z close to 1
	  out[i] = hyp_lmz(l, m, zvals[i], gamma_ratio0, psi0);
#ifdef DEBUG
	  if (!used_psi)
	    {
	      printf("First use of psi function at idx=%d\n", i);
	      used_psi=1;
	    }
#endif

	}
    }

  return 0;
}
