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
	SReal poly = 315.f / (M_PI*powf(h,9));
	SReal len = length(ij);

	if (len > h) return 0.f ;

	return (poly* (powf(h*h - len*len,3)));
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
				SReal kpol = Wpoly(p1p2, ir);
				res += Wpoly(p1p2,ir);
			}	
		}
		vbi[index] = 1.0 / res;
	}
}

#endif /* ifndef BOUNDARY_KERNEL */
