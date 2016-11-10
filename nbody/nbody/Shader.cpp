#include "Shader.h"


Shader::Shader()
{
}

void Shader::addShaderFile(const char* filePath, GLenum type) {
	// Check file exists
	assert(checkFileExists(filePath));

	// Load the contents of the shader file
	string content = loadFileAsString(filePath);

	// Create shader with OpenGL
	GLuint shaderID = glCreateShader(type);

	// Check for OpenGL errors
	if (GL_ERROR_CHECK) {
		cerr << "ERROR - loading shader " << filePath << endl;
		cerr << "OpenGL could not create shader object" << endl;
		throw runtime_error("Error adding shader file");
	}

	// Get shader source and attach to the shader object
	const char* source = content.c_str();
	glShaderSource(shaderID, 1, &source, 0);

	// Compile the shader
	glCompileShader(shaderID);

	// Check for OpenGL errors
	if (GL_ERROR_CHECK) {
		cerr << "ERROR - could not load shader " << filePath << endl;
		cerr << "Problem attaching and compiling source" << endl;
		throw runtime_error("Error adding shader file to shader");
	}

	// Check the compile status
	GLint compiled;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compiled);

	// Check if compiled
	if (!compiled) {
		// The shader failed to compile, get the log and output it
		GLsizei length;
		glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &length);
		// Create vector to the store log
		vector<char> errorLog(length);
		// Get the log
		glGetShaderInfoLog(shaderID, length, &length, &errorLog[0]);

		// Display the log
		cerr << "ERROR - Could not compile shader file " << filePath << endl;
		cerr << &errorLog[0] << endl;

		// Remove the shader object
		glDeleteShader(shaderID);

		throw runtime_error("Error adding shader file to shader");
	}

	_shaderFiles.push_back(shaderID);

}


void Shader::addShaderFile(const vector<string> &filePaths, GLenum type) {
	// Check that there is at least one filename
	assert(filePaths.size() > 0);

	// Check that each file exists
	for (const string &path : filePaths)
		checkFileExists(path.c_str());

	// Holds the contents of the file
	vector<string> fileContents;

	// Read in the file contents
	for (const string &path : filePaths)
	{
		// Load file contents
		string content = loadFileAsString(path.c_str());
		// Add to vector
		fileContents.push_back(content);
	}
	// Create shader with OpenGL
	GLuint shaderID = glCreateShader(type);

	// Check for OpenGL errors
	if (GL_ERROR_CHECK)
	{
		cerr << "ERROR - loading shader:" << endl;
		for (const string &name : filePaths)
			std::cerr << "\t" << name << std::endl;
		std::cerr << "Could not create shader object with OpenGL" << std::endl;
		throw std::runtime_error("Error adding shader to effect");
	}

	// Get shader source and attach to shader object
	const char **source = new const char*[fileContents.size()];
	for (unsigned int i = 0; i < fileContents.size(); ++i)
		source[i] = fileContents[i].c_str();

	// Add source to shader
	glShaderSource(shaderID, fileContents.size(), source, 0);

	// Compile the shader
	glCompileShader(shaderID);

	// Delete the sources
	delete[] source;

	// Check for OpenGL errors
	if (GL_ERROR_CHECK)
	{
		cerr << "ERROR - loading shader:" << endl;
		for (const string &name : filePaths)
			cerr << "\t" << name << endl;
		cerr << "Problem attaching and compiling source" << endl;
		throw runtime_error("Error adding shader to effect");
	}
	// We have tried to compile the shader.  Now check if compilation was
	// successful.
	// Get compile status
	GLint compiled;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compiled);

	// Check if compiled
	if (!compiled)
	{
		// The shader failed to compile, get the log and output it
		GLsizei length;
		glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &length);
		// Create vector to the store log
		vector<char> errorLog(length);
		// Get the log
		glGetShaderInfoLog(shaderID, length, &length, &errorLog[0]);

		// Display the log
		cerr << "ERROR - Could not compile shader file:" << endl;
		for (const string &name : filePaths)
			cerr << "\t" << name << endl;
		cerr << &errorLog[0] << endl;

		// Remove shader object from OpenGL
		glDeleteShader(shaderID);

		throw runtime_error("Error adding shader to effect");
	}

	_shaderFiles.push_back(shaderID);
}


void Shader::build() {
	// Create program
	_id = glCreateProgram();

	// Check for OpenGL errors
	if (GL_ERROR_CHECK) {
		cerr << "ERROR - building shader " << endl;
		cerr << "OpenGL could not create shader object" << endl;
		throw runtime_error("Error adding shader file");
	}

	for (GLuint &id : _shaderFiles) {
		// Add the shader file to the shader
		glAttachShader(_id, id);

		// Check for OpenGL errors
		if (GL_ERROR_CHECK) {
			cerr << "ERROR - adding shader file to shader" << endl;
			throw runtime_error("Error adding shader file to shader");
		}
	}

	// Attempt to link program
	glLinkProgram(_id);

	// Check for OpenGL errors
	if (GL_ERROR_CHECK)
	{
		// Display error
		cerr << "ERROR - Problem linking program " << endl;
		// Detach and delete shaders
		for (GLuint& s : _shaderFiles) {
			glDetachShader(_id, s);
			glDeleteShader(s);
		}
		// Delete program
		glDeleteProgram(_id);

		throw runtime_error("Error building shader");
	}

	// Check if linked successfully
	GLint linked;
	glGetProgramiv(_id, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		// The shader failed to link, get the log and output it
		GLsizei length;
		glGetProgramiv(_id, GL_INFO_LOG_LENGTH, &length);
		// Create vector to the store log
		vector<char> errorLog(length);
		// Get the log
		glGetProgramInfoLog(_id, length, &length, &errorLog[0]);
		// Display error
		cerr << "ERROR - building shader, problem linking program" << endl;
		cerr << &errorLog[0] << endl;
		// Detach shaders
		for (GLuint& s : _shaderFiles) {
			glDetachShader(_id, s);
			glDeleteShader(s);
		}
		// Delete program
		glDeleteProgram(_id);
		// Throw exception
		throw runtime_error("Error building shader");
	}

	// Detach and delete shaders
	for (GLuint& s : _shaderFiles) {
		glDetachShader(_id, s);
		glDeleteShader(s);
	}
}

Shader::~Shader()
{
}
