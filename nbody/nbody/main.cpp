// nbody.cpp : Defines the entry point for the console application.

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "FreeImage.lib")

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <chrono>

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

const float GRAVITY = 6.6742E-11;

long lastFrame;
float deltaTime;

const float PHYSICS_TIME = 0.001f;

// http://www.browndeertechnology.com/docs/BDT_OpenCL_Tutorial_NBody-rev3.html#opencl
// https://github.com/larsendt/opencl-nbody

/* Barnes-Hut */
// http://codereview.stackexchange.com/questions/95932/barnes-hut-n-body-simulator
// http://www.andrew.cmu.edu/user/esp/

void init() {
	Shader* colourShader = new Shader();
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.frag", GL_FRAGMENT_SHADER);
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.vert", GL_VERTEX_SHADER);
	colourShader->build();
	shaders["colourPass"] = colourShader;
}

void calcForce() {
//#pragma omp parallel for num_threads(6)
	for (int i = 0; i < particles.size(); i++) {
		for (int j = i+1; j < particles.size(); j++) {
			particles[i]->addForce(*particles[j]);
		}
	}
	//Then, loop again and update the bodies using timestep dt
//#pragma omp parallel for num_threads(6)
	for (int i = 0; i < particles.size(); i++) {
		particles[i]->update(PHYSICS_TIME);
	}
}

void update(float deltaTime) {
	calcForce();
}

void render() {

	Renderer::getInstance().bindShader(shaders["colourPass"]);

	// Set the MVP matrix
	Shader* currentShader = Renderer::getInstance().getActiveShader();

	float ratio = (float)Renderer::getWidth() / (float)Renderer::getHeight();
	float w = 100.0f;
	float h = 100.0f * (1.0f / ratio);
	mat4 MVP = glm::translate(glm::ortho(-w,w,h,-h, 0.5f, 1000.0f), vec3(-50.0f, -50.0f, -100.0f));
	//mat4 MVP = glm::translate(glm::perspective(90.0f, ratio, 0.5f, 5000.0f), vec3(0,0,-100.0f));

	glUniformMatrix4fv(currentShader->getUniformLoc("MVP"), 1, false, value_ptr(MVP));
	glUniform4f(currentShader->getUniformLoc("colour"), 1,0,0,1);

	glDisable(GL_CULL_FACE);


	glUniform4f(currentShader->getUniformLoc("colour"), 0, 1, 0, 1);
	glPointSize(2.0f);
	for (auto p : particles) {
		glBegin(GL_POINTS);
		glVertex3f(p->pos.x, p->pos.y, p->pos.z);
		glEnd();
	}
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

void createGridOfParticles(float startX, float startY, float width, float height, int nodes) {
	int sqrtNodeCount = sqrt(nodes);
	for (unsigned int yy = 0; yy < sqrtNodeCount; yy++) {
		for (unsigned int xx = 0; xx < sqrtNodeCount; xx++) {
			Particle* p = new Particle();
			p->pos = vec3(startX + xx * width / sqrtNodeCount + (width / sqrtNodeCount) / 2, startY + yy * height / sqrtNodeCount + (height / sqrtNodeCount) / 2, 0);
			p->mass = 1.98892e18;
			particles.push_back(p);
		}
	}
}

int main()
{
	createGridOfParticles(0, 0, 100, 100, 16*16);

	/*
	createGridOfParticles(0, 0, 20, 20, 32);
	createGridOfParticles(0, 0, 100, 100, 64);
	*/

	// Set the functions for the renderer to call
	Renderer::getInstance().setInit(init);
	Renderer::getInstance().setUpdate(update);
	Renderer::getInstance().setRender(render);
	Renderer::getInstance().setKeyCallback(keyCallback);

	lastFrame = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();

	// Start the renderer and return any error codes on completion
	return Renderer::getInstance().start(1280, 720);
	cin.get();
    return 0;
}

