#ifndef BOUNDARY_KERNEL_CUH
#define BOUNDARY_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

__device__ float Wpoly(SVec3 ij, float h)
{
	float poly = 315.f / (M_PI*powf(h,9));
	float len = length(ij);

	if (len > h) return 0.f ;

	return (poly* (powf(h*h - len*len,3)));
}

__global__ void computeVbi(SVec4 * bpos, float* vbi, float ir, unsigned int num_boundaries)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < num_boundaries) 
	{
		SVec3 p1 = glm::vec3(bpos[index]);

		float res = 0.f;
		for (int i = 0; i < num_boundaries; ++i) 
		{
			if (index != i) 
			{
				SVec3 p2 = glm::vec3(bpos[i]);
				SVec3 p1p2 = p1 - p2;
				float kpol = Wpoly(p1p2, ir);
				res += Wpoly(p1p2,ir);
			}	
		}
		vbi[index] = 1.f / res;
	}
}

#endif /* ifndef BOUNDARY_KERNEL */
