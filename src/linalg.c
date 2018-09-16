/*
  Linear algebra (for when numpy.linalg is too slow)
 */


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
