#include "Particle.h"


#include <iostream>


Particle::Particle(){

}

void Particle::update(float dt) {
	velocity += dt * force / mass;
	pos += dt * velocity;

	velocity *= 0.99f;
	resetForce();
}

void Particle::resetForce() {
	force = vec3(0.0f, 0.0f, 0.0f);
}

void Particle::addForce(Particle& other){
	float EPS = 3E3;      // softening parameter (just to avoid infinities)
	vec3 delta = other.pos - pos;
	float dist = distance(other.pos, pos);
	float F = (6.6742E-11 * mass * other.mass) / (dist*dist + EPS*EPS);
	force += F * delta / dist;
	other.force -= F * delta / dist;
}

Particle::~Particle(){

}
