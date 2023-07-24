//
// Created by Khurram Javed on 2023-03-12.
//


#ifndef INCLUDE_NN_OPTIMIZER_FACTORY_H_
#define INCLUDE_NN_OPTIMIZER_FACTORY_H_


#include "networks/graph.h"
#include "weight_optimizer.h"
#include <string>
#include "../../include/experiment/Experiment.h"

class OptimizerFactory {
 public:
  static Optimizer* get_optimizer(Graph* mygraph,  Experiment* myexp);
};


#endif //INCLUDE_NN_OPTIMIZER_FACTORY_H_
