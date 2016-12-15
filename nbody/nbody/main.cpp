
// Include the required libraries
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "FreeImage.lib")
#pragma comment(lib, "cudart.lib")

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationInformation.h"
#include "kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <windows.h>
#include <vector>
#include <map>
#include <sstream>
#include <chrono>
#include <random>

#include <omp.h>

#include "Renderer.h"
#include "Shader.h"
#include "Util.h"


using namespace std;
using namespace std::chrono;
using namespace glm;
using namespace Util;

// Colour Passthrough Shader
Shader colourShader;

// Vector storing all of the particles
vector<Particle> particles;

// Poisition of the camera
vec3 camPos = vec3(-SIM_WIDTH*1.5f, 0.0f, SIM_DEPTH*0.6f);
// Keep track of the time that has past to rotate the camera
float timePast = 0.0f;

// Load the colour passthrough shader
void init() {
	colourShader.addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.frag", GL_FRAGMENT_SHADER);
	colourShader.addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.vert", GL_VERTEX_SHADER);
	colourShader.build();
	colourShader = colourShader;
}

// Render the outline of a cube at the origin with dimensions: width, height, and depth
void renderCubeOutline(const float width, const float height, const float depth) {
	// Back Face
	glBegin(GL_LINE_LOOP);
	glVertex3f(-width / 2, -height / 2, -depth / 2);
	glVertex3f(-width / 2, height / 2, -depth / 2);
	glVertex3f(width / 2, height / 2, -depth / 2);
	glVertex3f(width / 2, -height / 2, -depth / 2);
	glEnd();

	// Front Face
	glBegin(GL_LINE_LOOP);
	glVertex3f(-width / 2, -height / 2, depth / 2);
	glVertex3f(-width / 2, height / 2, depth / 2);
	glVertex3f(width / 2, height / 2, depth / 2);
	glVertex3f(width / 2, -height / 2, depth / 2);
	glEnd();

	// Lines connecting the faces
	glBegin(GL_LINES);
		// Bottom Left
		glVertex3f(-width / 2, -height / 2, -depth / 2);
		glVertex3f(-width / 2, -height / 2, depth / 2);
		// Top Left
		glVertex3f(-width / 2, height / 2, -depth / 2);
		glVertex3f(-width / 2, height / 2, depth / 2);
		// Bottom Right
		glVertex3f(width / 2, -height / 2, -depth / 2);
		glVertex3f(width / 2, -height / 2, depth / 2);
		// Bottom Right
		glVertex3f(width / 2, height / 2, -depth / 2);
		glVertex3f(width / 2, height / 2, depth / 2);
	glEnd();
}

// Output the progress of result taking to the console
void show_percent(int it) {
	int dashes = (int)((it*0.5f / (float)ITERATIONS) * 100);
	cout << std::string(dashes+10, '\b') << "|" << std::string(dashes, '-') << "| " << dashes*2 << "%";
}

// Calculate the forces applying to the particles
void calcForce() {
#if USE_OPEN_MP
#pragma omp parallel for num_threads(CPU_THREADS) schedule(dynamic,1)
#endif
	for (int i = 0; i < particles.size(); i++) {
		Particle& p = particles.at(i);
		Particle* other;
		// Velocity to apply to p this iteration
		float vel[3] = { 0.0f, 0.0f, 0.0f };
		for (int j = 0; j < particles.size(); j++) {
			// Don't calculate against myself
			if (i == j)
				continue;
			// Reference to another particle
			other = &particles.at(j);
			// Calculate the distance between the two particles
			float distVec[3] = { 
				other->pos[0] - p.pos[0],
				other->pos[1] - p.pos[1],
				other->pos[2] - p.pos[2]
			};
			// Dot product + softening
			float sqrDist = (distVec[0] * distVec[0] + distVec[1] * distVec[1] + distVec[2] * distVec[2]) + EPS;
			float invDist3 = pow(1.0f / sqrtf(sqrDist), 3);
			// Increment the velocity
			vel[0] += distVec[0] * invDist3;
			vel[1] += distVec[1] * invDist3;
			vel[2] += distVec[2] * invDist3;
		}
		// Update this particle's velocity
		p.velocity[0] += PHYSICS_TIME * vel[0] * DAMPENING;
		p.velocity[1] += PHYSICS_TIME * vel[1] * DAMPENING;
		p.velocity[2] += PHYSICS_TIME * vel[2] * DAMPENING;
		// Update this particle's position
		p.pos[0] += p.velocity[0];
		p.pos[1] += p.velocity[1];
		p.pos[2] += p.velocity[2];

		// Clamp the particle within the simulation bounds
		p.pos[0] = min(max(p.pos[0], -SIM_WIDTH / 2.0f), SIM_WIDTH / 2.0f);
		p.pos[1] = min(max(p.pos[1], -SIM_HEIGHT / 2.0f), SIM_HEIGHT / 2.0f);
		p.pos[2] = min(max(p.pos[2], -SIM_DEPTH / 2.0f), SIM_DEPTH / 2.0f);
	}
}

// Rotate the camera around the simulation
void updateCamera(float deltaTime) {
	// Rotate the camera
	timePast += deltaTime;
	vec4 newCamPos = vec4(camPos, 1.0f) * glm::rotate(mat4(1.0f), pi<float>() * deltaTime * 0.25f, vec3(0, 1, 0));
	camPos.x = newCamPos.x;
	camPos.y = newCamPos.y;
	camPos.z = newCamPos.z;
}

void update(float deltaTime) {
#if RUN_FOR_RESULTS
	// Create a list to store the results
	static vector<float> results(ITERATIONS);
	// Get the start time for this result
	auto start = system_clock::now();
	static int iteration = 0;
#endif

	// Calculate the forces
	calcForce();

#if RUN_FOR_RESULTS
	// Calculate the time taken
	auto end = system_clock::now();
	auto timeTaken = end - start;
	// Store the result
	results[iteration] = duration_cast<milliseconds>(timeTaken).count();
	iteration++;

	// If we have all of the results then save them
	if (iteration == ITERATIONS) {
		std::stringstream ss;
		ss << "..\\results\\cpu_data_" << PARTICLE_COUNT << "_" << CPU_THREADS << ".csv";
		std::string fileName = ss.str();

		// Calculate the average timing and save the results to a csv file
		ofstream data(fileName, ofstream::out);
		unsigned int average = 0;
		for (unsigned int i = 0; i < ITERATIONS; i++) {
			data << i << "," << results[i] << endl;
			average += results[i];
		}
		average /= ITERATIONS;

		data << "Average," << average << endl;
		data.close();

		// Close the program
		glfwSetWindowShouldClose(Renderer::getWindow(), GL_TRUE);
	}

	show_percent(iteration);
#endif
#if RENDER
	updateCamera(deltaTime);
#endif
}

void updateCUDA(float deltaTime) {
#if RUN_FOR_RESULTS
	// Create a list to store the results
	static vector<float> results(ITERATIONS);
	// Get the start time for this result
	auto start = high_resolution_clock::now();
	static int iteration = 0;
#endif

	updateParticlesCUDA(particles);

#if RUN_FOR_RESULTS
	// Calculate the time taken
	auto end = high_resolution_clock::now();
	duration<float> timeTaken = end - start;
	results[iteration] = timeTaken.count() * 1000.0f;
	iteration++;

	// If we have all of the results then save them
	if (iteration == ITERATIONS) {
		std::stringstream ss;
		ss << "..\\results\\gpu_data_" << PARTICLE_COUNT << "_" << THREADS_PER_BLOCK << ".csv";
		std::string fileName = ss.str();

		// Calculate the average timing and save the results to a csv file
		ofstream data(fileName, ofstream::out);
		float average = 0;
		for (unsigned int i = 0; i < ITERATIONS; i++) {
			data << i << "," << results[i] << endl;
			average += results[i];
		}
		average /= ITERATIONS;

		data << "Average," << average << endl;
		data.close();

		// Close the program
		glfwSetWindowShouldClose(Renderer::getWindow(), GL_TRUE);
	}
	show_percent(iteration);
#endif
#if RENDER
	updateCamera(deltaTime);
#endif
}


void render() {
	// Bind the passthrough shader
	Renderer::getInstance().bindShader(&colourShader);

	// Get the active shader from the renderer
	Shader* currentShader = Renderer::getInstance().getActiveShader();

	// Get the screen ratio to calculate the perspective matrix
	float ratio = (float)Renderer::getWidth() / (float)Renderer::getHeight();
	float w = SIM_WIDTH;
	float h = SIM_HEIGHT * (1.0f / ratio);

	// Set the MVP matrix
	mat4 MVP = glm::perspective(45.0f, ratio, 0.5f, 5000.0f) * glm::lookAt(camPos + vec3(0.0f, sin(timePast*0.25f) * SIM_HEIGHT * 0.5f, 0.0f), vec3(0, 0, 0), vec3(0, 1, 0));
	glUniformMatrix4fv(currentShader->getUniformLoc("MVP"), 1, false, value_ptr(MVP));
	
	glEnable(GL_DEPTH_TEST);

	// Get the colour for each particle
	vector<float> colours;
	colours.reserve(particles.size() * 4);
	for (Particle p : particles) {
		float colour = abs(p.velocity[0]) * abs(p.velocity[1]) * abs(p.velocity[2]) + 0.1f;
		colours.push_back(colour);
		colours.push_back(colour);
		colours.push_back(colour);
		colours.push_back(0.8f);
	}

	// Create a buffer for the particle position
	GLuint posVBO = 0;
	glGenBuffers(1, &posVBO);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(float)*6, &particles[0], GL_STATIC_DRAW);

	// Create a buffer for the particle Colour
	GLuint colVBO = 0;
	glGenBuffers(1, &colVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colVBO);
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(float) * 4, &colours[0], GL_STATIC_DRAW);
	
	// Prepare the buffer for rendering
	GLuint vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, colVBO);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// Enable the vertex attributes
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	// Render the particles
	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, particles.size());

	// Delete the buffers
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffersARB(1, &posVBO);
	glDeleteBuffersARB(1, &colVBO);



	// Render the simulation bounds
	renderCubeOutline(SIM_WIDTH, SIM_HEIGHT, SIM_DEPTH);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {	
	// If F1 is pressed then close the window
	if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

// Generate a number of particles
void generateParticles(int particleCount) {
	particles.reserve(particleCount);
	std::mt19937 rng;
	rng.seed(std::random_device()());
	// Randomly distribute them within the simulation
	std::uniform_real_distribution<float> distW(0, 2);
	std::uniform_real_distribution<float> distH(0, 2);
	std::uniform_real_distribution<float> distD(0, 2);
	
	for (unsigned int i = 0; i < particleCount; i++) {
		Particle p;
		p.pos[0] = (distW(rng) - 1.0f) * SIM_WIDTH*0.5f;
		p.pos[1] = (distH(rng) - 1.0f) * SIM_HEIGHT*0.5f;
		p.pos[2] = (distD(rng) - 1.0f) * SIM_DEPTH*0.5f;
		particles.push_back(p);
	}
}


int main(){
	// Fill the simulation with particles
	generateParticles(PARTICLE_COUNT);	

	// Set the functions for the renderer to call
	Renderer::getInstance().setInit(init);

#if USE_CUDA
	// Initialise CUDA
	setUpCUDA(particles);
	cout << "|CUDA using " << PARTICLE_COUNT << " Particles." << endl << "|-------------------------------" << endl << endl;
	Renderer::getInstance().setUpdate(updateCUDA);
#else
	cout << "|-------------------------------" << endl << "|CPU using " << PARTICLE_COUNT << " Particles." << endl << "|-------------------------------" << endl;
	Renderer::getInstance().setUpdate(update);
#endif

#if RENDER
	Renderer::getInstance().setRender(render);
	Renderer::getInstance().setKeyCallback(keyCallback);
#endif

#if RUN_FOR_RESULTS
	cout << "Running for Results.." << endl;
#else
	cout << "!!! Not Running for Results !!!" << endl;
	cout << "This will run indefinitely and not gather any data" << endl;
#endif

	// Create the window and start the simulation
	Renderer::getInstance().start(1280, 720);
	
#if USE_CUDA
	// Clean up CUDA
	endCUDA();
#endif
    return 0;
}

