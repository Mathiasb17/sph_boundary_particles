#ifndef BOUNDARY_KERNEL_CUH
#define BOUNDARY_KERNEL_CUH

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

__device__ float Wpoly(float3 ij, float h)
{
	float poly = 315.f / (M_PI*powf(h,9));
	float len = length(ij);

	if (len > h) return 0.f ;

	return (poly* (powf(h*h - len*len,3)));
}

__global__ void computeVbi(float4 * bpos, float* vbi, float ir, unsigned int num_boundaries)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < num_boundaries) 
	{
   /*     float3 pos1 = make_float3(bpos[index]);*/

		/*float res = 0.f;*/
		/*for (int i = 0; i < num_boundaries; ++i) */
		/*{*/
			/*if (index != i) */
			/*{*/
				/*float3 pos2 = make_float3(bpos[i]);*/
				/*float3 p1p2 = pos1 - pos2;*/
				/*res += Wpoly(p1p2,ir);*/
			/*}	*/
		/*}*/
		/*vbi[index] = res;*/
	}
}

#endif /* ifndef BOUNDARY_KERNEL */
