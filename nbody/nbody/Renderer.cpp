#include "Renderer.h"

int Renderer::start(const int width, const int height) {
	this->_width = width;
	this->_height = height;

	if (setupWindow() != 0) {
		throw runtime_error("Could not create window");
		return -1;
	}

	// Run the init function
	if (initFunction)
		initFunction();

	glfwSetKeyCallback(Renderer::getWindow(), keyCallbackFunction);


	// TODO get real delta
	float deltaTime = 1.0f / 60.0f;
	// Set the clear colour
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	do {
		Renderer::setTarget();
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (updateFunction)
			updateFunction(deltaTime);
		glfwPollEvents();

		if (renderFunction)
			renderFunction();
		glfwSwapBuffers(this->_window);

	} while (glfwWindowShouldClose(this->_window) == false);
	glfwTerminate();

	return 0;
}


int Renderer::setupWindow() {
	if (glfwInit() == false) {
		fprintf(stderr, "GLFW failed to initialise");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	glShadeModel(GL_SMOOTH);
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	//glfwWindowHint(GLFW_OPENGL_CORE_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	this->_window = glfwCreateWindow(this->_width, this->_height, "N Body", NULL, NULL);

	if (!_window) {
		fprintf(stderr, "Window failed to create");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(this->_window);

	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "GLFW failed to initialise");
		glfwTerminate();
		return -1;
	}

	glEnable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);

	return 0;
}
