#ifndef SS_H
#define SS_H

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE 
#endif /* ifndef GLM_SWIZZLE */
#include <glm/glm.hpp>

#include "common.h"

#include <iostream>
#include <vector>
#include <set>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace sample_spheres
{

struct mesh
{
	std::vector<SVec4 > vertices;
	std::set< std::pair< int , int > > edges;
	std::vector<unsigned int > faces;
};

class ss
{
public:
	ss ();
	virtual ~ss ();

	static void sampleMesh(std::string fname, std::vector<SVec4> &boundary_spheres, SReal radius);

	static void sampleMeshVertices(const mesh m, std::vector<SVec4> & spheres, SReal radius);
	static void sampleMeshEdges(mesh & m, std::vector<SVec4> & spheres, SReal radius);
	static void sampleMeshFaces(std::vector<SVec4> & spheres, SReal radius, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);

	static void sampleBox(std::vector<SVec4> & spheres, glm::vec3 center, glm::vec3 size, SReal radius);
};

};

#endif /* ifndef SS_H */
