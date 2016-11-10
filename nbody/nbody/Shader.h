#pragma once

#include "Util.h"
#include <vector>

using namespace Util;

class Shader
{
private:
	// The IDs of the shader files that create this shader
	vector<GLuint> _shaderFiles;
	GLuint _id;

public:
	Shader();

	// Add a shader file to this shader
	void addShaderFile(const char* filePath, GLenum type);
	// Add multiple shader files of one type to this shader
	void addShaderFile(const vector<string> &filePaths, GLenum type);

	// Build the shader
	void build();

	// Returns the location of a uniform using its name
	GLint getUniformLoc(const std::string &name) const { return glGetUniformLocation(_id, name.c_str()); }
	GLuint getID() const { return _id; }

	~Shader();
};

