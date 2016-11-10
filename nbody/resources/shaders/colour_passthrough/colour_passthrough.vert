#version 410

// Model view projection matrix
uniform mat4 MVP;


// Incoming tex_coords
layout(location = 10) in vec2 tex_coord_in;

// Incoming position
layout(location = 0) in vec3 position;

void main()
{
	// Calculate screen position of vertex
	gl_Position = MVP * vec4(position, 1.0);
}