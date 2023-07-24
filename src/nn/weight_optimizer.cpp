//
// Created by Khurram Javed on 2023-03-12.
//

#include "../../include/nn/weight_optimizer.h"
#include <iostream>
#include <math.h>
#include <string>
SGD::SGD() {}


void SGD::update_weights(Graph *mygraph) {
  int counter = 0;
  for (auto &v : mygraph->list_of_vertices) {
    float incoming_edges = v->incoming_edges.size();
    for (auto &incoming : v->incoming_edges) {
      incoming.weight -=
          (incoming.step_size / incoming_edges) * incoming.gradient;
    }
    counter++;
  }
}

Adam::Adam(float step_size, float b1, float b2, float epsilon, Graph *mygraph)
    : SGD() {
  this->step_size = step_size;
  b1_value = b1;
  b2_value = b2;
  b1_normalizer = 1;
  b2_normalizer = 1;
  this->epsilon = epsilon;
  this->t = 0;
  for (auto &v : mygraph->list_of_vertices) {
    for (auto &incoming : v->incoming_edges) {
      this->b1.push_back(0);
      this->b2.push_back(0);
    }
  }
}

void Adam::update_weights(Graph *mygraph) {
  int counter = 0;
  b1_normalizer *= b1_value;
  b2_normalizer *= b2_value;
  this->t++;
  for (auto &v : mygraph->list_of_vertices) {
    float incoming_edges = v->incoming_edges.size();
    for (auto &incoming : v->incoming_edges) {
      this->b1[counter] =
          this->b1[counter] * b1_value + (1 - b1_value) * incoming.gradient;
      this->b2[counter] = this->b2[counter] * b2_value + (1 - b2_value) *
                                                             incoming.gradient *
                                                             incoming.gradient;
      float b1_grad = this->b1[counter] / (1 - b1_normalizer);
      float b2_grad = this->b2[counter] / (1 - b2_normalizer);
      incoming.weight -= (incoming.step_size ) * b1_grad /
                         (sqrt(b2_grad) + epsilon);
      counter++;
    }
  }
}

float SGD::get_average_gradient(Graph *mygraph, std::string node_type) {
  float sum_of_gradients = 0;
  int counter = 0;
  for (int i = mygraph->list_of_vertices.size() - 1; i >= 0; i--) {
    if (mygraph->list_of_vertices[i]->type == node_type) {
      for (auto &e : mygraph->list_of_vertices[i]->incoming_edges) {
        sum_of_gradients += std::abs(e.gradient);
        counter += 1;
      }
    }
  }
  return sum_of_gradients / counter;
}

float Adam::get_average_gradient(Graph *mygraph, std::string node_type) {
  float sum_of_gradients = 0;
  int counter = 0;
  int counter_inner = 0;
  for (auto &v : mygraph->list_of_vertices) {
    for (auto &incoming : v->incoming_edges) {
      if (v->type == node_type) {
        float b1_grad = this->b1[counter] / (1 - b1_normalizer);
        float b2_grad = this->b2[counter] / (1 - b2_normalizer);
        sum_of_gradients += std::abs(b1_grad / (sqrt(b2_grad) + epsilon));
        counter_inner++;
      }
      counter += 1;
    }
  }
  return sum_of_gradients / counter_inner;
}
