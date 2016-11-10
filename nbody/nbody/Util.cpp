#include "Util.h"

using namespace Util;


bool Util::doesStringContain(string strString, string strSubString) {
	// Make sure both of these strings are valid, return false if either is empty
	if (strString.length() <= 0 || strSubString.length() <= 0) return false;

	// grab the starting index where the sub string is in the original string
	unsigned int index = strString.find(strSubString);

	// Make sure the index returned was valid
	if (index >= 0 && index < strString.length())
		return true;

	// The sub string does not exist in strString.
	return false;
}

string Util::loadFileAsString(const char* filePath) {
	// Check file exists
	assert(checkFileExists(filePath));

	// Create a variable to store the contents
	string contents = "";
	ifstream stream(filePath, std::ios::in);

	// If the file is open read the contents line by line
	if (stream.is_open()) {
		string line = "";
		while (getline(stream, line)) {
			contents += "\n" + line;
		}
		// Close the file
		stream.close();
	}
	// Return the contents
	return contents;
}

void Util::loadTexture(unsigned int &texture, const char* strFileName) {

	// Check if file exists
	fprintf(stderr, strFileName);
	assert(checkFileExists(strFileName));

	// Get format of image
	auto format = FreeImage_GetFileType(strFileName);
	// Load image data
	auto image = FreeImage_Load(format, strFileName, 0);
	// Convert image to 32bit format
	auto temp = image;
	image = FreeImage_ConvertTo32Bits(image);
	FreeImage_Unload(temp);

	// Get image details
	auto width = FreeImage_GetWidth(image);
	auto height = FreeImage_GetHeight(image);

	// Get pixel data
	auto pixel_data = FreeImage_GetBits(image);

	// Generate texture with OpenGL
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	// Check for any errors with OpenGL
	if (GL_ERROR_CHECK)
	{
		// Problem creating texture object
		std::cerr << "ERROR - loading texture " << strFileName << std::endl;
		std::cerr << "Could not allocate texture with OpenGL" << std::endl;
		// Unload FreeImage data
		FreeImage_Unload(image);
		// Set id to 0
		texture = 0;
		// Throw exception
		throw std::runtime_error("Error creating texture");
	}

	// Basic scaling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


	// Now set texture data
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, pixel_data);

	// Check if error
	if (GL_ERROR_CHECK)
	{
		// Error loading texture data into OpenGL
		std::cerr << "ERROR - loading texture " << strFileName << std::endl;
		std::cerr << "Could not load texture data in OpenGL" << std::endl;
		// Unload FreeImage data
		FreeImage_Unload(image);
		// Unallocate image with OpenGL
		glDeleteTextures(1, &texture);
		texture = 0;
		// Throw exception
		throw std::runtime_error("Error creating texture");
	}



	// Unload image data
	FreeImage_Unload(image);

}


bool Util::checkFileExists(const char* filePath) {
	ifstream file(filePath);
	return file.good();
}

