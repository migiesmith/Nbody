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
#include <random>

#include <omp.h>

#include "Renderer.h"
#include "Shader.h"
#include "Util.h"


using namespace std;
using namespace std::chrono;
using namespace glm;
using namespace Util;


map<string, unsigned int> textures;
map<string, Shader*> shaders;

struct Particle {
	vec3 pos;
	vec3 velocity;
};

vector<Particle*> particles;

const float GRAVITY = 6.6742E-11;


const float PHYSICS_TIME = 10.0f;
const float DAMPENING = 0.99f;
const float SIM_WIDTH = 1000.0f;
const float SIM_HEIGHT = 1000.0f;


void init() {
	Shader* colourShader = new Shader();
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.frag", GL_FRAGMENT_SHADER);
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.vert", GL_VERTEX_SHADER);
	colourShader->build();
	shaders["colourPass"] = colourShader;
}

void calcForce() {
	float EPS = 3E4;
#pragma omp parallel for num_threads(6)
	for (int i = 0; i < particles.size(); i++) {
		vec3 vel = vec3(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < particles.size(); j++) {
			if (i == j)
				continue;

			vec3 distVec = particles[j]->pos - particles[i]->pos;
			float dist = dot(distVec, distVec) + EPS;
			if (dist > 0.1f) {
				vel += distVec * pow(1.0f / sqrtf(dist), 3);
			}
		}
		particles[i]->velocity += PHYSICS_TIME * vel * DAMPENING;
		particles[i]->pos += particles[i]->velocity;

		particles[i]->pos.x = min(max(particles[i]->pos.x, -SIM_WIDTH / 2.0f), SIM_WIDTH / 2.0f);
		particles[i]->pos.y = min(max(particles[i]->pos.y, -SIM_HEIGHT / 2.0f), SIM_HEIGHT / 2.0f);
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
	float w = SIM_WIDTH;
	float h = SIM_HEIGHT * (1.0f / ratio);
	mat4 MVP = glm::translate(glm::ortho(-w,w,h,-h, 0.5f, 1000.0f), vec3(0.0f, 0.0f, -100.0f));
	//mat4 MVP = glm::translate(glm::perspective(90.0f, ratio, 0.5f, 5000.0f), vec3(0,0,-100.0f));

	glUniformMatrix4fv(currentShader->getUniformLoc("MVP"), 1, false, value_ptr(MVP));

	glDisable(GL_CULL_FACE);

	
	glPointSize(2.0f);
	for (auto p : particles) {
		float speed = abs(p->velocity.x) * abs(p->velocity.y) + 0.2f;
		glUniform4f(currentShader->getUniformLoc("colour"), speed, speed, speed, 1);
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

void generateParticles(int particleCount) {
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> distW(0, SIM_WIDTH*0.9f);
	std::uniform_int_distribution<std::mt19937::result_type> distH(0, SIM_HEIGHT*0.9f);

	for (unsigned int i = 0; i < particleCount; i++) {
			Particle* p = new Particle();
			p->pos = vec3(distW(rng) - SIM_WIDTH*0.5f, distH(rng) - SIM_HEIGHT*0.5f, 0);
			particles.push_back(p);
	}
}

int main()
{
	generateParticles(2048);
	

	// Set the functions for the renderer to call
	Renderer::getInstance().setInit(init);
	Renderer::getInstance().setUpdate(update);
	Renderer::getInstance().setRender(render);
	Renderer::getInstance().setKeyCallback(keyCallback);
	
	// Start the renderer and return any error codes on completion
	return Renderer::getInstance().start(1280, 720);
	cin.get();
    return 0;
}

