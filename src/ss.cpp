#include <sph_boundary_particles/ss.h>

#include <array>
#include <sph_boundary_particles/helper_math.h>

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
void ss::sampleMesh(std::string fname, std::vector<SVec4> &boundary_spheres, SReal radius)
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
			SVec4 p = make_SVec4(scene->mMeshes[i]->mVertices[j].x,
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

		for (int i = 0; i < _mesh.faces.size(); i+=3)
		{
			const SVec3 p1 = make_SVec3(_mesh.vertices[_mesh.faces[i]]);
			const SVec3 p2 = make_SVec3(_mesh.vertices[_mesh.faces[i+1]]);
			const SVec3 p3 = make_SVec3(_mesh.vertices[_mesh.faces[i+2]]);
			sampleMeshFaces(boundary_spheres, radius, p1, p2, p3);
		}

	}
	aiReleaseImport( scene);
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleMeshVertices(const mesh m, std::vector<SVec4> & spheres, SReal radius)
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
void ss::sampleMeshEdges(mesh & m, std::vector<SVec4> & spheres, SReal radius)
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
		
		SVec4 p1 = m.vertices[it->first];
		SVec4 p2 = m.vertices[it->second];

		SVec4 edge = p2-p1;
		SReal elen = length(edge);
		int pNumber = std::floor(elen/(radius*2.f));
		SVec4 pe = edge/(SReal)pNumber, p;

		for(int i=1; i<pNumber; ++i)
		{
			p = p1 + (SReal)i*pe;
			spheres.push_back(p);
		}
	}
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
bool LineLineIntersect(
        const SVec3& p1, const SVec3& p2, const SVec3& p3, const SVec3& p4, SVec3& pa, SVec3& pb,
        SReal & mua, SReal & mub)
{
	SVec3 p13,p43,p21;
    SReal d1343,d4321,d1321,d4343,d2121;
    SReal numer,denom;

    p13.x = p1.x - p3.x;
    p13.y = p1.y - p3.y;
    p13.z = p1.z - p3.z;
    p43.x = p4.x - p3.x;
    p43.y = p4.y - p3.y;
    p43.z = p4.z - p3.z;
    if (std::abs(p43.x) < std::numeric_limits<SReal>::epsilon() && std::abs(p43.y) < std::numeric_limits<SReal>::epsilon() && std::abs(p43.z) < std::numeric_limits<SReal>::epsilon())
        return false;
    p21.x = p2.x - p1.x;
    p21.y = p2.y - p1.y;
    p21.z = p2.z - p1.z;
    if (std::abs(p21.x) < std::numeric_limits<SReal>::epsilon() && std::abs(p21.y) < std::numeric_limits<SReal>::epsilon() && std::abs(p21.z) < std::numeric_limits<SReal>::epsilon())
        return false;

    d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z;
    d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z;
    d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z;
    d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z;
    d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z;

    denom = d2121 * d4343 - d4321 * d4321;
    if (std::abs(denom) < std::numeric_limits<SReal>::epsilon())
        return false;
    numer = d1343 * d4321 - d1321 * d4343;

    mua = numer / denom;
    mub = (d1343 + d4321 * (mua)) / d4343;

    pa.x = p1.x + mua * p21.x;
    pa.y = p1.y + mua * p21.y;
    pa.z = p1.z + mua * p21.z;
    pb.x = p3.x + mub * p43.x;
    pb.y = p3.y + mub * p43.y;
    pb.z = p3.z + mub * p43.z;

    return true;
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleMeshFaces(std::vector<SVec4> & spheres, SReal radius, SVec3 p1, SVec3 p2, SVec3 p3)
{
	SReal particleDiameter = radius * 2.f;

    std::array< SVec3, 3> v = {{p1, p2, p3}};
    std::array< SVec3, 3 > edgesV = {{v[1]-v[0], v[2]-v[1], v[0]-v[2]}};
    std::array< int2, 3 > edgesI = {{make_int2(0,1), make_int2(1,2), make_int2(2,0)}};
    std::array< SReal, 3> edgesL = {{ length(edgesV[0]), length(edgesV[1]), length(edgesV[2]) }};
    //samples.clear();

    //Edges
    int pNumber=0;
	SVec3 pe = make_SVec3(0,0,0);
    for(int i=0; i<3; ++i)
    {
        pNumber = std::floor(edgesL[i]/particleDiameter);
        pe = edgesV[i]/(SReal)pNumber;
        for(int j=0; j<pNumber; ++j)
        {
			SVec3 p = v[edgesI[i].x] + (SReal)j*pe;
            //samples.push_back(p);
        }
    }

    //Triangles
    int sEdge=-1,lEdge=-1;
    SReal maxL = -std::numeric_limits<SReal>::max();
    SReal minL = std::numeric_limits<SReal>::max();
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
	SVec3 cros, normal;
    cros = cross(edgesV[lEdge], edgesV[sEdge]);
    normal = cross(edgesV[sEdge], cros);
    normal = normal / length(normal);

    std::array<bool, 3> findVertex = {{true, true, true}};
    findVertex[edgesI[sEdge].x] = false;
    findVertex[edgesI[sEdge].y] = false;
    int thirdVertex = -1;
    for(size_t i=0; i<findVertex.size(); ++i)
        if(findVertex[i]==true)
            thirdVertex = i;
	SVec3 tmpVec  = v[thirdVertex] - v[edgesI[sEdge].x];
    SReal sign = dot(normal, tmpVec);
    if(sign<0)
        normal = -normal;

    SReal triangleHeight = fabs(dot(normal, edgesV[lEdge]));
    int sweepSteps = triangleHeight/particleDiameter;
    bool success = false;

	SVec3 sweepA, sweepB, i1, i2, o1, o2;
    SReal m1, m2;
    int edge1,edge2;
    edge1 = (sEdge+1)%3;
    edge2 = (sEdge+2)%3;

    for(int i=1; i<sweepSteps; ++i)
    {
        sweepA = v[edgesI[sEdge].x] + (SReal)i*particleDiameter*normal;
        sweepB = v[edgesI[sEdge].y] + (SReal)i*particleDiameter*normal;
        success = LineLineIntersect(v[edgesI[edge1].x], v[edgesI[edge1].y], sweepA, sweepB, o1, o2, m1, m2);
        i1 = o1;
        if(success == false)
        {
            std::cout << "Intersection 1 failed" << std::endl;
        }
        success = LineLineIntersect(v[edgesI[edge2].x], v[edgesI[edge2].y], sweepA, sweepB, o1, o2, m1, m2);
        i2 = o1;
        if(success == false)
        {
            std::cout << "Intersection 1 failed" << std::endl;
        }
		SVec3 s = i1-i2;
        int step = floor(length(s)/particleDiameter);
		SVec3 ps = s/((SReal)step);
        for(int j=1; j<step; ++j)
        {
			SVec3 p = i2 + ( ps*(SReal)j );
            spheres.push_back(make_SVec4(p.x, p.y, p.z, 1.f));
        }
    }
}

//==================================================================================================== 
//==================================================================================================== 
//==================================================================================================== 
void ss::sampleBox(std::vector<SVec4> & spheres, SVec3 center, SVec3 size, SReal radius)
{
	SReal spacing = radius * 2.f;
	int epsilon = 0;
    int widthSize = std::floor(size.x/spacing);
    int heightSize = std::floor(size.y/spacing);
    int depthSize = std::floor(size.z/spacing);

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
	{
		for(int j = -epsilon; j <= depthSize+epsilon; ++j)
		{
			spheres.push_back(make_SVec4(center.x+i*spacing, center.y, center.z+j*spacing, 1.f));
		}
	}

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= depthSize+epsilon; ++j)
        {
            spheres.push_back(make_SVec4(center.x+i*spacing, center.y+size.y, center.z+j*spacing, 1.f));
        }
    }

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= heightSize+epsilon; ++j)
        {
            spheres.push_back(make_SVec4(center.x+i*spacing, center.y+j*spacing, center.z, 1.f));
        }
    }

	for(int i = -epsilon; i <= widthSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= heightSize-epsilon; ++j)
        {
            spheres.push_back(make_SVec4(center.x+i*spacing, center.y+j*spacing, center.z+size.z, 1.f));
        }
    }

	for(int i = -epsilon; i <= heightSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= depthSize+epsilon; ++j)
        {
            spheres.push_back(make_SVec4(center.x, center.y+i*spacing, center.z+j*spacing, 1.f));
        }
    }

	for(int i = -epsilon; i <= heightSize+epsilon; ++i)
    {
        for(int j = -epsilon; j <= depthSize+epsilon; ++j)
        {
            spheres.push_back(make_SVec4(center.x+size.x, center.y+i*spacing, center.z+j*spacing, 1.f));
        }
    }
}

} 
