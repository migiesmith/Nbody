#pragma once

#include <gl\glew.h>
#include <GLFW/glfw3.h>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include "Shader.h"


class Renderer
{
private:
	Shader* _activeShader;
	GLFWwindow* _window;
	int _width, _height;

	// The initialisation function
	void(*initFunction)();
	// The update function
	void(*updateFunction)(float);
	// The render function
	void(*renderFunction)();
	// The render function
	void(*keyCallbackFunction)(GLFWwindow*, int, int, int, int); 
	

public:



	// Create and run a new window with dimensions width and height
	int start(const int width, const int height);

	int setupWindow();

	// Sets the current shader
	void bindShader(Shader* shader) {
		// Set the active shader for the renderer
		_activeShader = shader;
		// Bind the shader
		glUseProgram(shader->getID());
	}

	// Returns the currently bound shader - only works if shaders are bound using bindShader(..)
	Shader* getActiveShader() {
		return _activeShader;
	}

	// Returns the rendering window
	static GLFWwindow* getWindow() { return getInstance()._window; }
	// Returns the width of the window
	static int getWidth() { return getInstance()._width; }
	// Returns the height of the window
	static int getHeight() { return getInstance()._height; }


	// Set the update function
	void setInit(void(*function)()) { initFunction = function; }
	// Set the update function
	void setUpdate(void(*function)(float)) { updateFunction = function; }
	// Set the render function
	void setRender(void(*function)()) { renderFunction = function; }
	// Set the key callback function
	void setKeyCallback(void(*function)(GLFWwindow*, int, int, int, int)) {
		keyCallbackFunction = function;
	}

	// Sets the render target
	void setTarget(const int buffer = 0) {
		glBindFramebuffer(GL_FRAMEBUFFER, buffer);
	}

	// Singleton parts below here

	// Returns the only instance of renderer
	static Renderer& getInstance()
	{
		static Renderer _instance;
		// Instantiated on first use.
		return _instance;
	}
private:
	// Singleton constructor
	Renderer() {};

	// Singleton destructors
public:
	Renderer(Renderer const&) = delete;
	void operator=(Renderer const&) = delete;
};