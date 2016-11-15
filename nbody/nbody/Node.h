#pragma once

#include <vector>
#include <glm\glm.hpp>
#include "Renderer.h"
#include "Particle.h"

#define G 0.000000000066742

using namespace std;
using namespace glm;

class Node
{
private:
	float x, y, width, height, mass;
	vec3 centreOfMass;
	vector<Node*> quadrants;

	unsigned int particleCount = 0;
	Particle* existingParticle = nullptr;


public:
	Node() {}
	Node(float x, float y, float width, float height);

	void deleteTree();
	void insertParticle(Particle* p);
	unsigned int getQuadrant(Particle* p);
	void createQuadrant(unsigned int quadIndex);

	void computeMassDistribution();
	vec3 calcForceFromTree(Particle* p);

	void renderGrid();

	float getWidth() { return width; }
	float getHeight() { return height; }

	~Node();
};

