#ifndef BOUNDARY_KERNEL_CUH
#define BOUNDARY_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
/*#include <math.h>*/

#include <sph_boundary_particles/helper_functions.h>
#include <sph_boundary_particles/helper_math.h>
#include <sph_boundary_particles/helper_cuda.h>

__device__ SReal Wpoly(SVec3 ij, SReal h)
{
	SReal poly = 315.0 / (M_PI*powf(h,9));
	SReal len = length(ij);

	if (len > h) return 0.0 ;

	return (poly* (powf(h*h - len*len,3)));
}

__device__ __host__ SReal Wmonaghan(SVec3 r, SReal h)
{
	SReal value = 0.f;
	SReal m_invH = 1.f  / h;
	SReal m_v = 1.0/(4.0*M_PI*h*h*h);
    SReal q = length(r)*m_invH;
    if( q >= 0 && q < 1 )
    {
        value = m_v*( (2-q)*(2-q)*(2-q) - 4.0f*(1-q)*(1-q)*(1-q));
    }
    else if ( q >=1 && q < 2 )
    {
        value = m_v*( (2-q)*(2-q)*(2-q) );
    }
    else
    {
        value = 0.0f;
    }
    return value;
}

__global__ void computeVbi(SVec4 * bpos, SReal* vbi, SReal ir, unsigned int num_boundaries)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < num_boundaries) 
	{
		SVec3 p1 = make_SVec3(bpos[index]);

		SReal res = 0.0;
		for (int i = 0; i < num_boundaries; ++i) 
		{
			if (index != i) 
			{
				SVec3 p2 = make_SVec3(bpos[i]);
				SVec3 p1p2 = p1 - p2;
#if KERNEL_SET == 1
				res += 1.0 / Wpoly(p1p2,ir);
#elif KERNEL_SET == 0
				res += 1.0 / Wmonaghan(p1p2, ir);
#endif
			}	
		}
		vbi[index] = res;
	}
}

#endif /* ifndef BOUNDARY_KERNEL */
