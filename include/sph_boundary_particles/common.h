#ifndef SYMBOL
#define SYMBOL value

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#if 1
typedef float SReal;
typedef glm::vec3 SVec3;
typedef glm::vec4 SVec4;
#else
typedef double SReal;
typedef glm::dvec3 SVec3;
typedef glm::dvec4 SVec4;
#endif

#endif /* ifndef SYMBOL */
