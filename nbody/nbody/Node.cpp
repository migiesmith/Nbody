#include "Node.h"

#include <iostream>

using namespace std;

Node::Node(float x, float y, float width, float height){
	this->x = x;
	this->y = y;
	this->width = width;
	this->height = height;
	this->particleCount = 0;
	quadrants.resize(4);
	for(unsigned int i = 0; i < 4; i++)
		quadrants[i] = NULL;
}

void Node::deleteTree() {
	for (unsigned int i = 0; i < 4; i++) {
		if (quadrants[i] != NULL) {
			quadrants[i]->deleteTree();
			delete (quadrants[i]);
		}
	}
	quadrants.clear();
	quadrants.resize(4);
	particleCount = 0;
	existingParticle = NULL;
	mass = 0;
	centreOfMass = vec3(0.0f, 0.0f, 0.0f);
}

void Node::insertParticle(Particle* p) {
	if (particleCount > 1) {
		unsigned int quadIndex = getQuadrant(p);
		// If sub node does not exist, create it
		if (quadrants[quadIndex] == NULL) {
			createQuadrant(quadIndex);
		}
		quadrants[quadIndex]->insertParticle(p);

	}else if (particleCount == 1) {
		unsigned int quadIndex = getQuadrant(existingParticle);
		// If sub node does not exist, create it
		if (quadrants[quadIndex] == NULL) {
			createQuadrant(quadIndex);
		}
		quadrants[quadIndex]->insertParticle(existingParticle);

		quadIndex = getQuadrant(p);
		// If sub node does not exist, create it
		if (quadrants[quadIndex] == NULL) {
			createQuadrant(quadIndex);
		}
		quadrants[quadIndex]->insertParticle(p);
	}else{
		existingParticle = p;
	}

	particleCount++;
}
	
unsigned int Node::getQuadrant(Particle* p){
	if (p->pos.x < (x + (width / 2))){
		if (p->pos.y < (y + (height / 2))){
			return 0;  // Top Left
		}else{
			return 2;  // Bottom Left
		}
	}else{
		if (p->pos.y < (y + (height / 2))){
			return 1;  // Top Right
		}else{
			return 3;  // Bottom Right
		}
	}
}

void Node::createQuadrant(unsigned int quadIndex) {
	float xPos = x;
	float yPos = y;
	float w = width / 2.0f;
	float h = height / 2.0f;

	if (quadIndex == 1 || quadIndex == 3) {
		xPos += w;
	}
	if (quadIndex == 2 || quadIndex == 3) {
		yPos += h;
	}
	quadrants[quadIndex] = new Node(xPos, yPos, w, h);
}

void Node::computeMassDistribution(){
	if (particleCount = 1) {
		centreOfMass = existingParticle->pos;
		mass = existingParticle->mass;
	}else{
		centreOfMass = vec3(0.0f, 0.0f, 0.0f);
		for (unsigned int i = 0; i < 4; i++) {
			if (quadrants[i] != NULL) {
				quadrants[i]->computeMassDistribution();
				mass += quadrants[i]->mass;
				centreOfMass += quadrants[i]->centreOfMass;
			}
			centreOfMass /= mass;
		}
	}
}

vec3 Node::calcForceFromTree(Particle* p){
	vec3 force = vec3(0.0f, 0.0f, 0.0f);
	if (particleCount == 1) {
		if (p == existingParticle) {
			force = vec3(0.0f, 0.0f, 0.0f);
		} else {
			float d = distance(existingParticle->pos, p->pos) + 0.000001f;
			force = normalize(existingParticle->pos - p->pos) * 0.000000000066742f * (p->mass * existingParticle->mass) / pow(d, 2);
		}

	} else {
		float r = distance(centreOfMass, p->pos) + 0.000001f;
		float d = width;
		if (d / r < 0.1) {
			force = normalize(centreOfMass - p->pos) * 0.000000000066742f * (p->mass * mass) / pow(r, 2);
		} else {
			for (unsigned int i = 0; i < 4; i++) {
				if (quadrants[i] != NULL) {
					force += quadrants[i]->calcForceFromTree(p);
				}
			}
		}
	}
	return force;
}

void Node::renderGrid(){
	glBegin(GL_LINE_LOOP);
	glVertex3f(x, y, 0.0f);
	glVertex3f(x + width, y, 0.0f);
	glVertex3f(x + width, y + height, 0.0f);
	glVertex3f(x, y + height, 0.0f);
	glEnd();
	for (unsigned int i = 0; i < 4; i++) {
		if (quadrants[i] != NULL) {
			quadrants[i]->renderGrid();
		}
	}
}


Node::~Node(){

}
