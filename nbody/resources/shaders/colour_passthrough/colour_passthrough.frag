#version 410

// Input colour
uniform vec4 colour;

// Outgoing colour
layout(location = 0) out vec4 out_colour;

void main()
{
	// Set the outgoing colour to the colour on the texture at the location tex_coord
	out_colour = colour;
}