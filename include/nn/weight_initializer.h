//
// Created by Khurram Javed on 2023-03-12.
//


#ifndef INCLUDE_NN_WEIGHT_INITIALIZER_H_
#define INCLUDE_NN_WEIGHT_INITIALIZER_H_
#include "networks/graph.h"

class WeightInitializer{
 protected:
  float lower;
  std::mt19937 mt;
  std::uniform_real_distribution<float> weight_sampler;
  float higher;
 public:
  WeightInitializer(float lower, float higher, int seed);
  Graph* initialize_weights(Graph* mygraph);
};

#endif //INCLUDE_NN_WEIGHT_INITIALIZER_H_
