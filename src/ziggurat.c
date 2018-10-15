/*
  A Ziggurat method for making random normals (following Marsaglia & Tsang 2000)

  Uses a 256-component table for drawing float precision random normals
 */

#include <stdint.h>
#include <math.h>
#include "zig.h" // Auto-generated Ziggurat constants

int zigg(const int num_needed, int num_ints, const uint32_t *restrict rand_ints, 
	 float *restrict out)
{
  /*
    num_needed - number of float normals wanted (size of out)
    num_ints   - number of 32-bit random integers provided
    rand_ints  - 32 bit random integers
    out        - to store normals

    If we fail to produce enough normals (because the Ziggurat method 
    occasionally uses >1 integer per normal) then return num_needed-num_done, 
    and out[:num_done] is filled. Otherwise return 0.
    
   */
  int todo = num_needed;
  while (todo)
    {

      if (!(num_ints--))
	return todo; // Consumed all our random integers, but still need to make more normals

      const unsigned int r = *rand_ints++, // next random integer
	tab_idx = r & 0xff, // 8 bits for lookup table
	sign = (r >> 8) & 1, // 1 bit for sign
	r23 = (r >> 9) & 0x7FFFFF; // 23 bit random integer
      
      const float x = r23 * to_float[tab_idx]; // Floating point from 23 bit random int

      if (r23 < in_block[tab_idx])
	{
	  // Default case, just linear
	  *out++ = sign ? -x : x;
	  todo--;
	  continue;
	}
      if (tab_idx == 0) 
	{
	  // Largest r, this corresponds to the final tail. Repeat until success

	  // Tail of the distribution
	  float xx, yy;
	  do {
	    if (num_ints<=1) // Exhausted random integers
	      return todo; 

	    num_ints -= 2;
	    const unsigned int x_u = *rand_ints++,
	      y_u = *rand_ints++;

	    // 1.0f+ since dont want log of zero
	    xx = inv_tail * (log32-logf(1.0f +(float)x_u)); 
	    yy = log32 - logf(1.0f + (float)y_u);

	  } while (yy + yy <= xx * xx);
	  
	  
	  // Found a point in the tail:
	  xx += tail;
	  *out++ = sign ? xx : -xx;
	  todo--;
	  continue;
	}

      if (!(num_ints--))
	return todo; // Exausted random integers
      
      // sample curve at edge of ziggurat
      if (expf(-0.5f*x*x) > y_seg[tab_idx-1]*(*rand_ints++) + y[tab_idx-1])
	{
	  *out++ = sign ? x : -x;
	  todo--;
	}
    }
  // All done.
  return 0;
}
