#ifndef SS_H
#define SS_H

#ifndef GLM_SWIZZLE
#define GLM_SWIZZLE 
#endif /* ifndef GLM_SWIZZLE */
#include <glm/glm.hpp>

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
	std::vector<glm::vec4 > vertices;
	std::set< std::pair< int , int > > edges;
	std::vector<unsigned int > faces;
};

class ss
{
public:
	ss ();
	virtual ~ss ();

	static void sampleMesh(std::string fname, std::vector<glm::vec4> &boundary_spheres, float radius);

	static void sampleMeshVertices(const mesh m, std::vector<glm::vec4> & spheres, float radius);
	static void sampleMeshEdges(mesh & m, std::vector<glm::vec4> & spheres, float radius);
	static void sampleMeshFaces(const mesh m, std::vector<glm::vec4> & spheres, float radius);

	static void sampleBox(std::vector<glm::vec4> & spheres, glm::vec3 center, glm::vec3 size, float radius);
};

};

#endif /* ifndef SS_H */
