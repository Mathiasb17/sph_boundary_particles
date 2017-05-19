#ifndef BOUNDARY_FORCES
#define BOUNDARY_FORCES

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE 
#endif /* ifndef GLM_SWIZZLE */

#include <vector>

#include <glm/glm.hpp>

namespace sample_spheres
{

class boundary_forces
{
public:
	boundary_forces ();
	virtual ~boundary_forces ();

	static void getVbi(std::vector<float> & vbi, std::vector<glm::vec4> boundary_spheres, float interaction_radius);

private:
	/* data */
};

} 

#endif /* ifndef BOUNDARY_FORCES */
