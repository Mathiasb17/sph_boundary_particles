#ifndef BOUNDARY_FORCES
#define BOUNDARY_FORCES

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE 
#endif /* ifndef GLM_SWIZZLE */

#include <vector>

#include <glm/glm.hpp>

#include "common.h"

namespace sample_spheres
{

class boundary_forces
{
public:
	boundary_forces ();
	virtual ~boundary_forces ();

	static void getVbi(std::vector<SReal> & vbi, std::vector< SVec4> boundary_spheres, SReal interaction_radius);

private:
	/* data */
};

} 

#endif /* ifndef BOUNDARY_FORCES */
