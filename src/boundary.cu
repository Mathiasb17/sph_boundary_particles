#include "boundary.cuh"

#include "boundary_kernel.cuh"

extern "C"
{
	unsigned int iDivU(unsigned int a, unsigned int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	void computeGridSiz(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivU(n, numThreads);
	}

	void updateVbi(float* boundary_pos, float* vbi, float ir, unsigned int num_boundaries)
	{
		float* d_boundary_pos;
		float* d_vbi;

		cudaMalloc((void**)&d_boundary_pos, num_boundaries*sizeof(float)*4);
		cudaMalloc((void**)&d_vbi, num_boundaries*sizeof(float));

		//cudaMemcpy
		cudaMemcpy(d_vbi, vbi, num_boundaries*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_boundary_pos, boundary_pos, num_boundaries*sizeof(float)*4, cudaMemcpyHostToDevice);

		/*//kernel call*/
		
		unsigned int numThreads, numBlocks;
		computeGridSiz(num_boundaries, 256, numBlocks, numThreads);

		computeVbi<<<numBlocks, numThreads>>>((float4*)d_boundary_pos, d_vbi, ir,num_boundaries);

		/*//transfer back to host mem*/
		cudaMemcpy(vbi, d_vbi, num_boundaries*sizeof(float), cudaMemcpyDeviceToHost);

		/*//cudaFree*/
		cudaFree(d_boundary_pos);
		cudaFree(d_vbi);
	}
}
