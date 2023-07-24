//
// Created by Khurram Javed on 2023-03-12.
//


#include "../../include/nn/weight_initializer.h"


WeightInitializer::WeightInitializer(float lower, float higher, int seed) :weight_sampler(lower, higher), mt(seed){
  this->lower = lower;
  this->higher = higher;
}

Graph *WeightInitializer::initialize_weights(Graph *mygraph) {
  int counter = 0;
  for (auto &v: mygraph->list_of_vertices) {
    for (auto &incoming: v->incoming_edges) {
      incoming.weight = weight_sampler(this->mt);
    }
    counter++;
  }
  return mygraph;
}