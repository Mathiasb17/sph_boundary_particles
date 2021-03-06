#include <sph_boundary_particles/boundary_forces.h>
#include "boundary.cuh"

//#include <cmath>

namespace sample_spheres
{

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
boundary_forces::boundary_forces ()
{

}

//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
boundary_forces::~boundary_forces ()
{

}
//====================================================================================================  
//====================================================================================================  
//====================================================================================================  
void boundary_forces::getVbi(std::vector<SReal> & vbi, std::vector<SVec4> boundary_spheres, SReal interaction_radius)
{
	vbi.resize(boundary_spheres.size(),0);
	updateVbi((SReal*)boundary_spheres.data(), (SReal*)vbi.data(), interaction_radius, boundary_spheres.size());
}

} 
