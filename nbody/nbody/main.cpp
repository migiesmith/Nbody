// nbody.cpp : Defines the entry point for the console application.
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "FreeImage.lib")
#pragma comment(lib, "cudart.lib")

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <chrono>
#include <random>

#include <omp.h>

#include "Renderer.h"
#include "Shader.h"
#include "Util.h"
#include "Particle.h"


using namespace std;
using namespace std::chrono;
using namespace glm;
using namespace Util;


map<string, unsigned int> textures;
map<string, Shader*> shaders;


vector<Particle*> particles;


vec3 camPos = vec3(-SIM_WIDTH*1.5f, 0.0f, SIM_DEPTH*0.6f);


void init() {
	Shader* colourShader = new Shader();
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.frag", GL_FRAGMENT_SHADER);
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.vert", GL_VERTEX_SHADER);
	colourShader->build();
	shaders["colourPass"] = colourShader;
}

void renderCube(const float width, const float height, const float depth) {
	Shader* currentShader = Renderer::getInstance().getActiveShader();
	glUniform4f(currentShader->getUniformLoc("colour"), 0.5f, 0.0f, 0.0f, 1.0f);
	glBegin(GL_LINE_LOOP);
	glVertex3f(-width / 2, -height / 2, -depth / 2);
	glVertex3f(-width / 2, height / 2, -depth / 2);
	glVertex3f(width / 2, height / 2, -depth / 2);
	glVertex3f(width / 2, -height / 2, -depth / 2);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glVertex3f(-width / 2, -height / 2, depth / 2);
	glVertex3f(-width / 2, height / 2, depth / 2);
	glVertex3f(width / 2, height / 2, depth / 2);
	glVertex3f(width / 2, -height / 2, depth / 2);
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(-width / 2, -height / 2, -depth / 2);
	glVertex3f(-width / 2, -height / 2, depth / 2);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(-width / 2, height / 2, -depth / 2);
	glVertex3f(-width / 2, height / 2, depth / 2);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(width / 2, -height / 2, -depth / 2);
	glVertex3f(width / 2, -height / 2, depth / 2);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(width / 2, height / 2, -depth / 2);
	glVertex3f(width / 2, height / 2, depth / 2);
	glEnd();
}

void calcForce() {
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particles.size(); i++) {
		Particle* p = particles[i];
		Particle* other;
		float vel[3] = { 0.0f, 0.0f, 0.0f };
		for (int j = 0; j < particles.size(); j++) {
			other = particles[j];
			if (i == j)
				continue;
			float distVec[3] = { 
				other->pos[0] - p->pos[0],
				other->pos[1] - p->pos[1],
				other->pos[2] - p->pos[2]
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
		p->velocity[0] += PHYSICS_TIME * vel[0] * DAMPENING;
		p->velocity[1] += PHYSICS_TIME * vel[1] * DAMPENING;
		p->velocity[2] += PHYSICS_TIME * vel[2] * DAMPENING;
		p->pos[0] += p->velocity[0];
		p->pos[1] += p->velocity[1];
		p->pos[2] += p->velocity[2];

		p->pos[0] = min(max(p->pos[0], -SIM_WIDTH / 2.0f), SIM_WIDTH / 2.0f);
		p->pos[1] = min(max(p->pos[1], -SIM_HEIGHT / 2.0f), SIM_HEIGHT / 2.0f);
		p->pos[2] = min(max(p->pos[2], -SIM_DEPTH / 2.0f), SIM_DEPTH / 2.0f);
	}
}


void update(float deltaTime) {
	calcForce();

	vec4 newCamPos = vec4(camPos, 1.0f) * glm::rotate(mat4(1.0f), pi<float>() * 0.001f, vec3(0, 1, 0));
	camPos.x = newCamPos.x;
	camPos.y = newCamPos.y;
	camPos.z = newCamPos.z;
}

void updateCUDA(float deltaTime) {
	updateParticles(particles);

	vec4 newCamPos = vec4(camPos, 1.0f) * glm::rotate(mat4(1.0f), pi<float>() * 0.001f, vec3(0, 1, 0));
	camPos.x = newCamPos.x;
	camPos.y = newCamPos.y;
	camPos.z = newCamPos.z;
}

void render() {

	Renderer::getInstance().bindShader(shaders["colourPass"]);

	// Get the active shader
	Shader* currentShader = Renderer::getInstance().getActiveShader();

	float ratio = (float)Renderer::getWidth() / (float)Renderer::getHeight();
	float w = SIM_WIDTH;
	float h = SIM_HEIGHT * (1.0f / ratio);

	// Set the MVP matrix
	mat4 MVP = glm::perspective(45.0f, ratio, 0.5f, 5000.0f) * glm::lookAt(camPos, vec3(0, 0, 0), vec3(0, 1, 0));
	glUniformMatrix4fv(currentShader->getUniformLoc("MVP"), 1, false, value_ptr(MVP));

	glDisable(GL_CULL_FACE);

	
	glPointSize(2.0f);
	for (auto p : particles) {
		float speed = abs(p->velocity[0]) * abs(p->velocity[1]) * abs(p->velocity[2]) + 0.2f;
		glUniform4f(currentShader->getUniformLoc("colour"), speed, speed, speed, 1);
		glBegin(GL_POINTS);
		glVertex3f(p->pos[0], p->pos[1], p->pos[2]);
		glEnd();
	}

	renderCube(SIM_WIDTH, SIM_HEIGHT, SIM_DEPTH);

}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	
	if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {

	}
	/*else if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
		renderLensFlare = !renderLensFlare;
	}

	if (scene)
		scene->keyCallback(window, key, scancode, action, mods);
	*/
}

void generateParticles(int particleCount) {
	particles.reserve(particleCount);
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> distW(0, SIM_WIDTH*0.9f);
	std::uniform_int_distribution<std::mt19937::result_type> distH(0, SIM_HEIGHT*0.9f);
	std::uniform_int_distribution<std::mt19937::result_type> distD(0, SIM_DEPTH*0.9f);

	for (unsigned int i = 0; i < particleCount; i++) {
			Particle* p = new Particle();
			p->pos[0] = distW(rng) - SIM_WIDTH*0.5f;
			p->pos[1] = distH(rng) - SIM_HEIGHT*0.5f;
			p->pos[2] = distD(rng) - SIM_DEPTH*0.5f;
			particles.push_back(p);
	}
}

void cudaInfo() {
	// Get CUDA device
	int device;
	cudaGetDevice(&device);

	// Get CUDA device
	cudaDeviceProp properites;
	cudaGetDeviceProperties(&properites, device);

	// Display properties
	cout << "-----------------------" << endl;
	cout << "Name: " << properites.name << endl;
	cout << "CUDA Capability: " << properites.major << "." << properites.minor << endl;
	cout << "Core: " << properites.multiProcessorCount << endl;
	cout << "Memory: " << properites.totalGlobalMem / (1024 * 1024) << "MB" << endl;
	cout << "Clock freq: " << properites.clockRate / 1000 << "MHz" << endl;
	cout << "-----------------------" << endl;
}

int main(){
	bool useCUDA = true;
	generateParticles(PARTICLE_COUNT);
	

	// Set the functions for the renderer to call
	Renderer::getInstance().setInit(init);

	if (useCUDA) {
		// Initialise CUDA
		cudaSetDevice(0);
		cudaInfo();
		Renderer::getInstance().setUpdate(updateCUDA);
	}else{
		Renderer::getInstance().setUpdate(update);
	}

	Renderer::getInstance().setRender(render);
	Renderer::getInstance().setKeyCallback(keyCallback);


	// Start the renderer and return any error codes on completion
	return Renderer::getInstance().start(1280, 720);
	cin.get();
    return 0;
}

