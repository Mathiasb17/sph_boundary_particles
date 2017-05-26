#ifndef BOUNDARY_CUH
#define BOUNDARY_CUH 

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <math.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

extern "C"
{
	void updateVbi(glm::vec4* boundary_pos, float* vbi, float ir, unsigned int num_boundaries);
}

#endif /* ifndef BOUNDARY_CUH */
