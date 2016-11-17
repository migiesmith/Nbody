#pragma once

#include "Renderer.h"

using namespace glm;

#define GRAVITY 6.6742E-11


#define PHYSICS_TIME 10.0f
#define DAMPENING 0.99f
#define SIM_WIDTH 1000.0f
#define SIM_HEIGHT 1000.0f
#define SIM_DEPTH 1000.0f
#define EPS 3E4

#define PARTICLE_COUNT 8192


struct Particle {
	float pos[3];
	float velocity[3] = {0.0f, 0.0f, 0.0f};
};