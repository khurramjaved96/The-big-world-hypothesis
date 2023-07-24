//
// Created by Khurram Javed on 2023-03-12.
//


#ifndef INCLUDE_NN_WEIGHT_OPTIMIZER_H_
#define INCLUDE_NN_WEIGHT_OPTIMIZER_H_

#include <vector>
#include "networks/graph.h"
#include <string>


class Optimizer {
 public:
  virtual void update_weights(Graph *mygraph) = 0;
  virtual float get_average_gradient(Graph *mygraph, std::string node_type) = 0;
};

class SGD : public Optimizer {
 protected:
  float step_size;
 public:
  SGD();
  virtual void update_weights(Graph *mygraph) override;
  float get_average_gradient(Graph *mygraph, std::string node_type) override;
};

class Adam : public SGD {
 protected:
  float b1_value;
  float b2_value;
  std::vector<float> b1;
  std::vector<float> b2;
  float epsilon;
  float t;
  float b1_normalizer;
  float b2_normalizer;
 public:
  Adam(float step_size, float b1, float b2, float epsilon, Graph *my_graph);
  void update_weights(Graph *mygraph) override;
  float get_average_gradient(Graph *mygraph, std::string node_type) override;

};
#endif //INCLUDE_NN_WEIGHT_OPTIMIZER_H_
