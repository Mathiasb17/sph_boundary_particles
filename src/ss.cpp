#include "ss.h"
#include <glm/gtx/string_cast.hpp>

#include <array>

namespace sample_spheres
{

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
ss::ss (){}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
ss::~ss (){}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleMesh(std::string fname, std::vector<glm::vec4> &boundary_spheres, float radius)
{
	mesh _mesh;
	const aiScene* scene = aiImportFile( fname.c_str(),
			aiProcess_CalcTangentSpace       |
			aiProcess_GenNormals             |
			aiProcess_Triangulate            |
			aiProcess_JoinIdenticalVertices  |
			aiProcess_SortByPType);

	if( !scene)
	{
		std::cout << aiGetErrorString() << std::endl;
		exit(1);
	}

	for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
	{
		//loop into current mesh for vertices
		for (unsigned int j = 0; j < scene->mMeshes[i]->mNumVertices; ++j)
		{
			glm::vec4 p = glm::vec4(scene->mMeshes[i]->mVertices[j].x,
					scene->mMeshes[i]->mVertices[j].y,
					scene->mMeshes[i]->mVertices[j].z, 1);

			_mesh.vertices.push_back(p);
		}

		sampleMeshVertices(_mesh, boundary_spheres, radius);

		//loop into current mesh for indices
		for (unsigned int j = 0; j < scene->mMeshes[i]->mNumFaces; ++j)
		{
			aiFace f = scene->mMeshes[i]->mFaces[j];
			for (unsigned int k = 0; k < f.mNumIndices; ++k)
			{
				_mesh.faces.push_back(f.mIndices[k]);
			}
		}

		sampleMeshEdges(_mesh, boundary_spheres, radius);
		sampleMeshFaces(_mesh, boundary_spheres, radius);
	}
	aiReleaseImport( scene);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleMeshVertices(const mesh m, std::vector<glm::vec4> & spheres, float radius)
{
	(void)radius;
	for (unsigned int i = 0; i < m.vertices.size(); ++i) 
	{
		spheres.push_back(m.vertices[i]);
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleMeshEdges(mesh & m, std::vector<glm::vec4> & spheres, float radius)
{
	//1ere etape : on recupere les aretes dans un ensemble pour garantir l'unicite des spheres generees
	for (unsigned int i = 0; i < m.faces.size(); i += 3) 
	{
		std::pair<std::set<std::pair<int, int> >::iterator,bool> ret1;
		std::pair<std::set<std::pair<int, int> >::iterator,bool> ret2;

		//on essaie d'inserer le segment a-b et le segment b-a
		ret1 = m.edges.insert(std::make_pair(m.faces[i], m.faces[i+1]));
		ret2 = m.edges.insert(std::make_pair(m.faces[i+1], m.faces[i]));

		//si les deux sont inseres, le segment est nouveau, on vire b-a
		if (ret1.second == true && ret2.second == true) 
		{
			m.edges.erase(ret2.first);
		}

		//si seul a-b est insere, b-a est deja present, on vire a-b
		if (ret1.second == true && ret2.second == false) 
		{
			m.edges.erase(ret1.first);
		}
		
		//si seul b-a est insere, a-b est deja present, on vire b-a
		if (ret1.second == false && ret2.second == true) 
		{
			m.edges.erase(ret2.first);
		}

		//on fait la meme chose avec b-c / c-b
		ret1 = m.edges.insert(std::make_pair(m.faces[i+1], m.faces[i+2]));
		ret2 = m.edges.insert(std::make_pair(m.faces[i+2], m.faces[i+1]));
		if (ret1.second == true && ret2.second == true) 
		{
			m.edges.erase(ret2.first);
		}
		if (ret1.second == true && ret2.second == false) 
		{
			m.edges.erase(ret1.first);
		}
		if (ret1.second == false && ret2.second == true) 
		{
			m.edges.erase(ret2.first);
		}
		
		//on fait la meme chose avec a-c / c-a
		ret1 = m.edges.insert(std::make_pair(m.faces[i], m.faces[i+2]));
		ret2 = m.edges.insert(std::make_pair(m.faces[i+2], m.faces[i]));
		if (ret1.second == true && ret2.second == true) 
		{
			m.edges.erase(ret2.first);
		}
		if (ret1.second == true && ret2.second == false) 
		{
			m.edges.erase(ret1.first);
		}
		if (ret1.second == false && ret2.second == true) 
		{
			m.edges.erase(ret2.first);
		}

	}

	//2e etape, on genere les spheres
	std::set<std::pair<int, int> >::iterator  it;
	for(it = m.edges.begin(); it != m.edges.end(); ++it)
	{
		
		glm::vec4 p1 = m.vertices[it->first];
		glm::vec4 p2 = m.vertices[it->second];

		glm::vec4 edge = p2-p1;
		float elen = glm::length(edge);
		int pNumber = std::floor(elen/(radius*2.f));
		glm::vec4 pe = edge/(float)pNumber, p;

		for(int i=1; i<pNumber; ++i)
		{
			p = p1 + (float)i*pe;
			spheres.push_back(p);
		}
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
bool LineLineIntersect(
        const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, const glm::vec3& p4, glm::vec3& pa, glm::vec3& pb,
        float & mua, float & mub)
{
	glm::vec3 p13,p43,p21;
    float d1343,d4321,d1321,d4343,d2121;
    float numer,denom;

    p13[0] = p1[0] - p3[0];
    p13[1] = p1[1] - p3[1];
    p13[2] = p1[2] - p3[2];
    p43[0] = p4[0] - p3[0];
    p43[1] = p4[1] - p3[1];
    p43[2] = p4[2] - p3[2];
    if (std::abs(p43[0]) < std::numeric_limits<float>::epsilon() && std::abs(p43[1]) < std::numeric_limits<float>::epsilon() && std::abs(p43[2]) < std::numeric_limits<float>::epsilon())
        return false;
    p21[0] = p2[0] - p1[0];
    p21[1] = p2[1] - p1[1];
    p21[2] = p2[2] - p1[2];
    if (std::abs(p21[0]) < std::numeric_limits<float>::epsilon() && std::abs(p21[1]) < std::numeric_limits<float>::epsilon() && std::abs(p21[2]) < std::numeric_limits<float>::epsilon())
        return false;

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

    denom = d2121 * d4343 - d4321 * d4321;
    if (std::abs(denom) < std::numeric_limits<float>::epsilon())
        return false;
    numer = d1343 * d4321 - d1321 * d4343;

    mua = numer / denom;
    mub = (d1343 + d4321 * (mua)) / d4343;

    pa[0] = p1[0] + mua * p21[0];
    pa[1] = p1[1] + mua * p21[1];
    pa[2] = p1[2] + mua * p21[2];
    pb[0] = p3[0] + mub * p43[0];
    pb[1] = p3[1] + mub * p43[1];
    pb[2] = p3[2] + mub * p43[2];

    return true;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleMeshFaces(const mesh m, std::vector<glm::vec4> & spheres, float radius)
{
	//sampling triangle interiors
	
	float particleDiameter = radius*2.f;
	for (unsigned int i = 0; i < m.faces.size(); i += 3) 
	{
		
		glm::vec3 p1 = m.vertices[m.faces[i]].xyz();
		glm::vec3 p2 = m.vertices[m.faces[i+1]].xyz();
		glm::vec3 p3 = m.vertices[m.faces[i+2]].xyz();

		std::array< glm::vec3, 3> v = {{p1, p2, p3}};
		std::array< glm::vec3, 3 > edgesV = {{v[1]-v[0], v[2]-v[1], v[0]-v[2]}};
		std::array< glm::vec2, 3 > edgesI = {{glm::vec2(0,1), glm::vec2(1,2), glm::vec2(2,0)}};
		std::array< float, 3> edgesL = {{ glm::length(edgesV[0]), glm::length(edgesV[1]), glm::length(edgesV[2]) }};
		//spheres.push_back(m.vertices[i]);
		
		//Edges
		int pNumber=0;
		glm::vec3 pe(0,0,0);
		for(int j=0; j<3; ++j)
		{
			pNumber = std::floor(edgesL[j]/particleDiameter);
			pe = edgesV[j]/(float)pNumber;
			for(int k=0; k<pNumber; ++k)
			{
				glm::vec3 p = v[edgesI[j][0]] + (float)k*pe;
				//samples.push_back(p);
			}
		}

		//Triangles
		int sEdge=-1,lEdge=-1;
		float maxL = -std::numeric_limits<float>::max();
		float minL = std::numeric_limits<float>::max();
		for(int i=0; i<3; ++i)
		{
			if(edgesL[i]>maxL)
			{
				maxL = edgesL[i];
				lEdge = i;
			}
			if(edgesL[i]<minL)
			{
				minL = edgesL[i];
				sEdge = i;
			}
		}
		glm::vec3 cross, normal;
		cross = glm::cross(edgesV[lEdge], edgesV[sEdge]);
		normal = glm::cross(edgesV[sEdge], cross);
		glm::normalize(normal);

		std::array<bool, 3> findVertex = {{true, true, true}};
		findVertex[edgesI[sEdge][0]] = false;
		findVertex[edgesI[sEdge][1]] = false;
		int thirdVertex = -1;
		for(size_t i=0; i<findVertex.size(); ++i)
			if(findVertex[i]==true)
				thirdVertex = i;
		glm::vec3 tmpVec  = v[thirdVertex] - v[edgesI[sEdge][0]];
		float sign = glm::dot(normal, tmpVec);
		if(sign<0)
			normal = -normal;

		float triangleHeight = std::abs(glm::dot(normal, edgesV[lEdge]));
		int sweepSteps = triangleHeight/particleDiameter;
		bool success = false;

		glm::vec3 sweepA, sweepB, i1, i2, o1, o2;
		float m1, m2;
		int edge1,edge2;
		edge1 = (sEdge+1)%3;
		edge2 = (sEdge+2)%3;

		for(int i=1; i<sweepSteps; ++i)
		{
			sweepA = v[edgesI[sEdge][0]] + (float)i*particleDiameter*normal;
			sweepB = v[edgesI[sEdge][1]] + (float)i*particleDiameter*normal;
			success = LineLineIntersect(v[edgesI[edge1][0]], v[edgesI[edge1][1]], sweepA, sweepB, o1, o2, m1, m2);
			i1 = o1;
			if(success == false)
			{
				std::cout << "Intersection 1 failed" << std::endl;
			}
			success = LineLineIntersect(v[edgesI[edge2][0]], v[edgesI[edge2][1]], sweepA, sweepB, o1, o2, m1, m2);
			i2 = o1;
			if(success == false)
			{
				std::cout << "Intersection 1 failed" << std::endl;
			}
			glm::vec3 s = i1-i2;
			int step = std::floor(s.length()/particleDiameter);
			glm::vec3 ps = s/((float)step);
			for(int j=1; j<step; ++j)
			{
				glm::vec3 p = i2 + (float)j*ps;
				glm::vec4 pfinal(p,1.f);
				spheres.push_back(pfinal);
			}
		}
		
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleBox(std::vector<glm::vec4> & spheres, glm::vec3 center, glm::vec3 size, float radius)
{
	float spacing = radius * 2.f;
	int epsilon = 0;
    int widthSize = std::floor(size[0]/spacing);
    int heightSize = std::floor(size[1]/spacing);
    int depthSize = std::floor(size[2]/spacing);

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
	{
		for(int j = -epsilon; j <= depthSize+epsilon; ++j)
		{
			spheres.push_back(glm::vec4(center[0]+i*spacing, center[1], center[2]+j*spacing, 1.f));
		}
	}

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= depthSize+epsilon; ++j)
        {
            spheres.push_back(glm::vec4(center[0]+i*spacing, center[1]+size[1], center[2]+j*spacing, 1.f));
        }
    }

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= heightSize+epsilon; ++j)
        {
            spheres.push_back(glm::vec4(center[0]+i*spacing, center[1]+j*spacing, center[2], 1.f));
        }
    }

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= heightSize-epsilon; ++j)
        {
            spheres.push_back(glm::vec4(center[0]+i*spacing, center[1]+j*spacing, center[2]+size[2], 1.f));
        }
    }

	for(int i = -epsilon; i <= heightSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= depthSize+epsilon; ++j)
        {
            spheres.push_back(glm::vec4(center[0], center[1]+i*spacing, center[2]+j*spacing, 1.f));
        }
    }

	for(int i = -epsilon; i <= heightSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= depthSize+epsilon; ++j)
        {
            spheres.push_back(glm::vec4(center[0]+size[0], center[1]+i*spacing, center[2]+j*spacing, 1.f));
        }
    }
}

} 
