#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <gl\glew.h>
#include <gl\freeglut.h>
#include <assert.h> 

#include <FreeImage\FreeImage.h>

using namespace std;

/**
Utility class, mostly for handling the loading of files
@author Grant Smith
@version 1.0 9/03/16
*/
namespace Util
{

	// Check if a string contains another string
	bool doesStringContain(string strString, string strSubString);

	/**
	Load the contents of a file as a string
	@param filePath the path to the file to be loaded
	@return the contents of the file as a string
	*/
	string loadFileAsString(const char* filePath);
	bool checkFileExists(const char* filePath);


	// Loads an image from a file and stores it as an OpenGL texture
	void loadTexture(unsigned int &texture, const char* strFileName);


	// Checks if there are any OpenGL errors
	inline bool glErrorCheck(int line, const string& file) {
		// Get the error from OpenGL
		GLenum error = glGetError();
		// Display a message if there is an error
		if (error) {
			cerr << "OpenGL Error: " << gluErrorString(error) << endl;
			cerr << "In file " << file << "at line" << line << endl;
			return true;
		}
		return false;
	}

#define GL_ERROR_CHECK glErrorCheck(__LINE__, __FILE__)

};

