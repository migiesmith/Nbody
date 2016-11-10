// nbody.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>


#include "Renderer.h"
#include "Shader.h"
#include "Util.h"


using namespace std;
using namespace glm;
using namespace Util;


map<string, unsigned int> textures;
map<string, Shader*> shaders;

// http://www.browndeertechnology.com/docs/BDT_OpenCL_Tutorial_NBody-rev3.html#opencl
// https://github.com/larsendt/opencl-nbody

void init() {
	Shader* colourShader = new Shader();
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.frag", GL_FRAGMENT_SHADER);
	colourShader->addShaderFile("..\\resources\\shaders\\colour_passthrough\\colour_passthrough.vert", GL_VERTEX_SHADER);
	colourShader->build();
	shaders["colourPass"] = colourShader;
}

void update(float deltaTime) {

}

void render() {

	Renderer::getInstance().bindShader(shaders["colourPass"]);

	// Set the MVP matrix
	glUniformMatrix4fv(Renderer::getInstance().getActiveShader()->getUniformLoc("MVP"), 1, false, value_ptr(mat4()));

	glBegin(GL_TRIANGLES);
	glVertex3f(-1,0,0);
	glVertex3f(0, 2, 0);
	glVertex3f(1, 0, 0);
	glEnd();


}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	/*
	if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
		renderSSAO = !renderSSAO;
	}
	else if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
		renderLensFlare = !renderLensFlare;
	}

	if (scene)
		scene->keyCallback(window, key, scancode, action, mods);
	*/
}



int main()
{
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

