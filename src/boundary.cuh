#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH 

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <math.h>

extern "C"
{
	void updateVbi(float* boundary_pos, float* vbi, float ir, unsigned int num_boundaries);
}

#endif /* ifndef BOUNDARY_CUH */
