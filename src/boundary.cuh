#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <math.h>

#include <sph_boundary_particles/common.h>

extern "C"
{
	void updateVbi(SReal* boundary_pos, SReal* vbi, SReal ir, unsigned int num_boundaries);
}

#endif /* ifndef BOUNDARY_CUH */
