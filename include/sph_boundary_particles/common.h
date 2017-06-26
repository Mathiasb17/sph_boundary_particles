#ifndef COMMON_H
#define COMMON_H 

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>


#ifdef DOUBLE_PRECISION
typedef double SReal;
typedef double2 SVec2;
typedef double3 SVec3;
typedef double4 SVec4;
#define make_SVec3 make_double3
#define make_SVec4 make_double4

#else
typedef float SReal;
typedef int2 SVec2i;
typedef float2 SVec2;
typedef float3 SVec3;
typedef float4 SVec4;
#define make_SVec3 make_float3
#define make_SVec4 make_float4

#endif

#endif /* ifndef COMMON_H */
