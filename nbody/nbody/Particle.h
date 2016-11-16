#pragma once

#include <glm\glm.hpp>

using namespace glm;

class Particle
{
public:

	vec3 pos;
	vec3 velocity;
	vec3 force;
	float mass;

	Particle();

	void update(float dt);
	void resetForce();
	void addForce(Particle& other);


	~Particle();
};

