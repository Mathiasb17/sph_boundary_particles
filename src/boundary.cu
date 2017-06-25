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

	void updateVbi(SReal* boundary_pos, SReal* vbi, SReal ir, unsigned int num_boundaries)
	{
		SReal* d_boundary_pos;
		SReal* d_vbi;

		cudaMalloc((void**)&d_boundary_pos, num_boundaries*sizeof(SReal)*4);
		cudaMalloc((void**)&d_vbi, num_boundaries*sizeof(SReal));

		//cudaMemcpy
		cudaMemcpy(d_vbi, vbi, num_boundaries*sizeof(SReal), cudaMemcpyHostToDevice);
		cudaMemcpy(d_boundary_pos, boundary_pos, num_boundaries*sizeof(SReal)*4, cudaMemcpyHostToDevice);

		/*//kernel call*/
		
		unsigned int numThreads, numBlocks;
		computeGridSiz(num_boundaries, 256, numBlocks, numThreads);

		computeVbi<<<numBlocks, numThreads>>>((SVec4*)d_boundary_pos, d_vbi, ir,num_boundaries);

		/*//transfer back to host mem*/
		cudaMemcpy(vbi, d_vbi, num_boundaries*sizeof(SReal), cudaMemcpyDeviceToHost);

		/*//cudaFree*/
		cudaFree(d_boundary_pos);
		cudaFree(d_vbi);
	}
}
