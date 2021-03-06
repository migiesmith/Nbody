#include "kernel.h"

using namespace std;

#ifndef KERNEL
	#define KERNEL
	// Buffers for GPU communication
	Particle *bufferIN, *bufferOUT;
	// Size of the simulation in particles
	auto dataSize = sizeof(Particle) * PARTICLE_COUNT;
#endif


// Calculate the forces applying to the particles
__global__ void calcForce(const Particle *in, Particle *out) {
	// Get the thread's unique ID  - (blockIDX * blockDIM) + threadIDX
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	Particle other; // Reference to another particle
	// Velocity to apply to p this iteration
	float vel[3] = { 0.0f, 0.0f, 0.0f };
	for (int j = 0; j < PARTICLE_COUNT; j++) {
		other = in[j];
		// Calculate the distance between the two particles
		float distVec[3] = {
			other.pos[0] - in[idx].pos[0],
			other.pos[1] - in[idx].pos[1],
			other.pos[2] - in[idx].pos[2]
		};
		// Dot product + softening
		float sqrDist = (distVec[0] * distVec[0] + distVec[1] * distVec[1] + distVec[2] * distVec[2]) + EPS;
		float invDist3 = pow(1.0f / sqrtf(sqrDist), 3);
		vel[0] += distVec[0] * invDist3;
		vel[1] += distVec[1] * invDist3;
		vel[2] += distVec[2] * invDist3;
	}

	// Update this particle's velocity
	out[idx].velocity[0] = in[idx].velocity[0] + PHYSICS_TIME * vel[0] * DAMPENING;
	out[idx].velocity[1] = in[idx].velocity[1] + PHYSICS_TIME * vel[1] * DAMPENING;
	out[idx].velocity[2] = in[idx].velocity[2] + PHYSICS_TIME * vel[2] * DAMPENING;
	// Update this particle's position
	out[idx].pos[0] = in[idx].pos[0] + out[idx].velocity[0];
	out[idx].pos[1] = in[idx].pos[1] + out[idx].velocity[1];
	out[idx].pos[2] = in[idx].pos[2] + out[idx].velocity[2];

	// Clamp the particle within the simulation bounds
	out[idx].pos[0] = min(max(out[idx].pos[0], -SIM_WIDTH / 2.0f), SIM_WIDTH / 2.0f);
	out[idx].pos[1] = min(max(out[idx].pos[1], -SIM_HEIGHT / 2.0f), SIM_HEIGHT / 2.0f);
	out[idx].pos[2] = min(max(out[idx].pos[2], -SIM_DEPTH / 2.0f), SIM_DEPTH / 2.0f);
}

// Swap the input and output buffers (saves passing data to the GPU every frame)
void swapBuffers() {
	Particle *tempBuffer = bufferIN;
	bufferIN = bufferOUT;
	bufferOUT = tempBuffer;	
}

// Update the particles on the gpu and store them in the passed in vector
void updateParticlesCUDA(const vector<Particle> &particles) {
	// Update the particles
	calcForce<<<PARTICLE_COUNT / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(bufferIN, bufferOUT);
	// Wait for the GPU to finish
	cudaDeviceSynchronize();
	// Copy the results back to the CPU
	cudaMemcpy((void*)&particles[0], bufferOUT, dataSize, cudaMemcpyDeviceToHost);
	
	// Swap the in and out buffers
	swapBuffers();
}

// http://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
// Gets the number of streaming processors on the GPU
int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

void cudaInfo() {
	// Get CUDA device
	int device;
	cudaGetDevice(&device);

	// Get CUDA device
	cudaDeviceProp properites;
	cudaGetDeviceProperties(&properites, device);

	// Display properties
	cout << "|-------------------------------" << endl;
	cout << "|Name: " << properites.name << endl;
	cout << "|CUDA Capability: " << properites.major << "." << properites.minor << endl;
	cout << "|Cores: " << properites.multiProcessorCount << endl;
	cout << "|SP Cores: " << getSPcores(properites) << endl;
	cout << "|Memory: " << properites.totalGlobalMem / (1024 * 1024) << "MB" << endl;
	cout << "|Clock freq: " << properites.clockRate / 1000 << "MHz" << endl;
	cout << "|-------------------------------" << endl;
}

void setUpCUDA(const vector<Particle> &particles) {
	cudaInfo();
	cudaMalloc((void**)&bufferIN, dataSize);
	cudaMalloc((void**)&bufferOUT, dataSize);
	cudaMemcpy(bufferIN, &particles.at(0), dataSize, cudaMemcpyHostToDevice);
}

// Delete the buffers
void endCUDA() {
	cudaFree(bufferIN);
	cudaFree(bufferOUT);
}