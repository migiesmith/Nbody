#pragma once

#define USE_CUDA false

#define USE_OPEN_MP true && !USE_CUDA
#define CPU_THREADS 8

#define RUN_FOR_RESULTS true
#define ITERATIONS 100

#define PHYSICS_TIME 2.0f
#define DAMPENING 0.99f
#define SIM_WIDTH 1000.0f
#define SIM_HEIGHT 1000.0f
#define SIM_DEPTH 1000.0f
#define EPS 3E4


#define RENDER !RUN_FOR_RESULTS

/*
 If these values are modified, the project must
 be rebuilt for the kernel to receive the changes
 */
#define PARTICLE_COUNT 8192*4
#define THREADS_PER_BLOCK 1024

