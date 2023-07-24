//
// Created by Khurram Javed on 2023-03-12.
//


#include "../../../include/nn/networks/vertex.h"
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

// Vertex implementation
Vertex::Vertex() {
  age = 0;
  is_output = false;
  value = 0;
  sum_of_outgoing_weights = 0;
  id = id_generator;
  this->utility_trace = 1;
  id_generator++;
  this->max_value = 10;
  this->min_value = -10;
  this->utility = 0;
  this->type = "linear";
  this->new_util = 1;
};

float Vertex::forward() {
  return this->value;
}

float Vertex::forward_with_val(float value) {
  return value;
}
float Vertex::backward(float val) {
  return 1;
}

float Vertex::get_value() {
  return this->value;
}

int Vertex::id_generator = 0;

ReluVertex::ReluVertex() : Vertex() {
  this->type = "relu";
  this->max_value = 10;
  this->min_value = 0;
}

ReluVertex::ReluVertex(int seed) : Vertex() {
  this->type = "relu";
  this->max_value = 10;
  this->min_value = 0;
  this->seed = seed;
}

LeakyReluVertex::LeakyReluVertex() : Vertex() {
  this->type = "leakyrelu";
  this->max_value = 10;
  this->min_value = -10;
}

SigmoidVertex::SigmoidVertex() : Vertex() {
  this->type = "sigmoid";
  this->max_value = 1;
  this->min_value = 0;
}

TanHVertex::TanHVertex() : Vertex() {
  this->type = "tanh";
  this->max_value = 1;
  this->min_value = -1;
}

NormalizedRelu::NormalizedRelu() {
  this->type = "normalizedrelu";
  this->max_value = 10;
  this->min_value = 0;
  this->mean = 0;
  this->variance = 1;
  this->decay_rate = 0.999997;
}

float NormalizedRelu::forward() {
  float temp_value = 0;
  if(this->value > 0){
    temp_value = this->value;
  }
  if(this->value > 3)
    this->value = 3;
  this->mean = this->mean*this->decay_rate + (1-this->decay_rate)*temp_value;
  this->variance = this->variance*this->decay_rate + (1-this->decay_rate)*(this->value - this->mean)*(this->value - this->mean);
  float normalized_val =  (temp_value - this->mean)/float(sqrt(this->variance + 1e-6));
  if(this->variance < 0.2)
    this->variance = 0.2;
//  if(this->mean != 0)
//    this->mean = 0;
//  if(this->mean < -0.001 || this->mean > 0.001)
//    this->mean = 0;
//  std::cout << "Mean val = " << this->mean << std::endl;
//  if(this->variance < 0.2){
//  std::cout << "Mean = " << this->mean;
//  std::cout << " Var = " << this->variance << std::endl;
//  }
//  if(this->variance < 0.05)
//    std::cout << "Variance is " << this->variance << std::endl;
  return normalized_val;
}

float NormalizedRelu::forward_with_val(float val) {
  float temp_value = 0;
  if(val > 0){
    temp_value = val;
  }
  return (temp_value - this->mean)/float(sqrt(this->variance + 1e-6));
}

float NormalizedRelu::backward(float val) {
  if(val > 0 and val < 3) {
    return 1.0f/float(sqrt(this->variance + 1e-6));
  }
  return 0;
}

float ReluVertex::forward_with_val(float val) {
  if (val > 0)
    return val;
  return 0;
}
float ReluVertex::forward() {
  this->age++;
  if (this->value > 0)
    return this->value;
  return 0;
}

float ReluVertex::backward(float val) {
  if (val > 0)
    return 1;
  return 0;
}

float LeakyReluVertex::forward_with_val(float val) {
  if (val > 0)
    return val;
  return 0.1 * val;
}
float LeakyReluVertex::forward() {
  if (this->value > 0)
    return this->value;
  return 0.1 * this->value;
}

float LeakyReluVertex::backward(float val) {
  if (val > 0)
    return 1;
  return 0.1;
}

float SigmoidVertex::sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float SigmoidVertex::forward_with_val(float val) {
  return sigmoid(val);
}

float SigmoidVertex::forward() {
  return sigmoid(this->value);
}

float SigmoidVertex::backward(float val) {
  float temp = sigmoid(val);
  return temp * (1 - temp);
}

float TanHVertex::tanh(float x) {
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float TanHVertex::forward_with_val(float val) {
  return tanh(val);
}

float TanHVertex::forward() {
  return tanh(this->value);
}

float TanHVertex::backward(float val) {
  float temp = tanh(val);
  return (1 - temp * temp);
}

Vertex *VertexFactory::get_vertex(const std::string &type) {
  if (type == "linear") {
    return new Vertex();
  } else if (type == "relu") {
    return new ReluVertex();
  }
  else if (type == "normalizedrelu") {
    return new NormalizedRelu();
  }
  else if (type == "leakyrelu") {
    return new LeakyReluVertex();
  } else if (type == "sigmoid") {
    return new SigmoidVertex();
  } else if (type == "binary") {
    return new BinaryVertex();
  } else if (type == "tanh") {
    return new TanHVertex();
  }
  return nullptr;
}

Vertex *VertexFactory::get_vertex_with_seed(const std::string &type, int seed) {
  if (type == "linear") {
    return new Vertex();
  } else if (type == "relu") {
    return new ReluVertex(seed);
  }
  else if (type == "normalizedrelu") {
    return new NormalizedRelu();
  }
  else if (type == "leakyrelu") {
    return new LeakyReluVertex();
  } else if (type == "sigmoid") {
    return new SigmoidVertex();
  } else if (type == "binary") {
    return new BinaryVertex();
  } else if (type == "tanh") {
    return new TanHVertex();
  }
  return nullptr;
}

BinaryVertex::BinaryVertex() {
  this->max_value = 1;
  this->min_value = -1;
}

float BinaryVertex::forward() {
  if (this->value > 0)
    return 1;
  else
    return 0;
}

float BinaryVertex::forward_with_val(float val) {
  if (val > 0)
    return 1;
  else
    return 0;
}

float BinaryVertex::backward(float val) {
  return 0;
}