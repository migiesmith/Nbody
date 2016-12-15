#pragma once

#include "Renderer.h"

using namespace glm;

struct Particle {
	// Position of the particle
	float pos[3];
	// Velocity of the particle
	float velocity[3] = { 0.0f, 0.0f, 0.0f };
};