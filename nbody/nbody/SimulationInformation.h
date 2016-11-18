#pragma once

#define GRAVITY 6.6742E-11

#define USE_CUDA true
#define RENDER true

#define CPU_THREADS 8

#define RUN_FOR_RESULTS false
#define ITERATIONS 100

#define PHYSICS_TIME 2.0f
#define DAMPENING 0.99f
#define SIM_WIDTH 1000.0f
#define SIM_HEIGHT 1000.0f
#define SIM_DEPTH 1000.0f
#define EPS 3E4

#define PARTICLE_COUNT 8192
#define THREADS_PER_BLOCK 1024