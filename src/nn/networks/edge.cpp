
#include "../../../include/nn/networks/graph.h"
#include <random>
#include <iostream>
#include <string>
#include <vector>


//Edge implementation
Edge::Edge(float weight, int from, int to, float step_size) {
  this->edge_id = edge_id_generator++;
  this->from = from;
  this->step_size = step_size;
  this->h = 0;
  this->to = to;
  this->gradient = 0;
  this->temp_gradient = 0;
  this->weight = weight;
  this->local_utility = 0;
  this->utility = 0;
}

int Edge::edge_id_generator = 0;