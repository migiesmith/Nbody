#pragma once

#include "Renderer.h"

using namespace glm;

struct Particle {
	float pos[3];
	float velocity[3] = {0.0f, 0.0f, 0.0f};
};