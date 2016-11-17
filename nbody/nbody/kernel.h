#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include "SimulationInformation.h"
#include "Particle.h"


void updateParticlesCUDA(const vector<Particle> &particles);
__global__ void calcForce(const Particle *in, Particle *out);

void cudaInfo();
void setUpCUDA(const vector<Particle> &particles);
void endCUDA();