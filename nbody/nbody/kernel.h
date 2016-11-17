#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#include "Particle.h"

void updateParticles(vector<Particle*> &particles);
__global__ void calcForce(const Particle *in, Particle *out);