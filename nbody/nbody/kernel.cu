#include "kernel.h"

using namespace std;

__global__ void calcForce(const Particle *in, Particle *out) {
	// Get block index
	unsigned int blockIDX = blockIdx.x;
	// Get thread index
	unsigned int threadIDX = threadIdx.x;
	// Get the number of threads per block
	unsigned int blockDIM = blockDim.x;
	// Get the thread's unique ID  - (blockIDX * blockDIM) + threadIDX
	unsigned int idx = (blockIDX * blockDIM) + threadIDX;

	Particle other;

	float vel[3] = { 0.0f, 0.0f, 0.0f };
	for (int j = 0; j < 2048; j++) {
		other = in[j];
		if (idx == j)
			continue;
		float distVec[3] = {
			other.pos[0] - in[idx].pos[0],
			other.pos[1] - in[idx].pos[1],
			other.pos[2] - in[idx].pos[2]
		};
		// Dot product + softening
		float dist = (distVec[0] * distVec[0] + distVec[1] * distVec[1] + distVec[2] * distVec[2]) + EPS;
		if (dist > 0.1f) {
			float invDist3 = pow(1.0f / sqrtf(dist), 3);
			vel[0] += distVec[0] * invDist3;
			vel[1] += distVec[1] * invDist3;
			vel[2] += distVec[2] * invDist3;
		}
	}
	
	out[idx].velocity[0] = in[idx].velocity[0] + PHYSICS_TIME * vel[0] * DAMPENING;
	out[idx].velocity[1] = in[idx].velocity[1] + PHYSICS_TIME * vel[1] * DAMPENING;
	out[idx].velocity[2] = in[idx].velocity[2] + PHYSICS_TIME * vel[2] * DAMPENING;
	out[idx].pos[0] = in[idx].pos[0] + out[idx].velocity[0];
	out[idx].pos[1] = in[idx].pos[1] + out[idx].velocity[1];
	out[idx].pos[2] = in[idx].pos[2] + out[idx].velocity[2];

	out[idx].pos[0] = min(max(out[idx].pos[0], -SIM_WIDTH / 2.0f), SIM_WIDTH / 2.0f);
	out[idx].pos[1] = min(max(out[idx].pos[1], -SIM_HEIGHT / 2.0f), SIM_HEIGHT / 2.0f);
	out[idx].pos[2] = min(max(out[idx].pos[2], -SIM_DEPTH / 2.0f), SIM_DEPTH / 2.0f);
}

void updateParticles(vector<Particle*> &particles) {
	const int ELEMENTS = particles.size();
	auto dataSize = sizeof(Particle) * ELEMENTS;
	vector<Particle> in(ELEMENTS);
	vector<Particle> out(ELEMENTS);

	for (unsigned int i = 0; i < ELEMENTS; i++) {
		in[i] = *particles[i];
	}

	Particle *bufferIN, *bufferOUT;

	cudaMalloc((void**)&bufferIN, dataSize);
	cudaMalloc((void**)&bufferOUT, dataSize);

	cudaMemcpy(bufferIN, &in[0], dataSize, cudaMemcpyHostToDevice);

	calcForce<<<ELEMENTS, 1>>>(bufferIN, bufferOUT);

	cudaDeviceSynchronize();

	cudaMemcpy(&out[0], bufferOUT, dataSize, cudaMemcpyDeviceToHost);

	cudaFree(bufferIN);
	cudaFree(bufferOUT);
	for (unsigned int i = 0; i < ELEMENTS; i++) {
		for (unsigned int xx = 0; xx < 3; xx++) {
			particles[i]->pos[xx] = out[i].pos[xx];
			particles[i]->velocity[xx] = out[i].velocity[xx];
		}
	}
}
