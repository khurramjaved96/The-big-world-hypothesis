//
// Created by Khurram Javed on 2023-03-12.
//

#include <string>
#include "../../include/nn/optimizer_factory.h"
#include "../../include/nn/networks/graph.h"
#include "../../include/experiment/Experiment.h"
#include <exception>

Optimizer *OptimizerFactory::get_optimizer(Graph *mygraph, Experiment *myexp) {
  if(myexp->get_string_param("optimizer") == "sgd"){
    return new SGD();
  }
  else if(myexp->get_string_param("optimizer") == "adam"){
    return new Adam(myexp->get_float_param("step_size"), myexp->get_float_param("b1"), myexp->get_float_param("b2"), myexp->get_float_param("epsilon"), mygraph);
  }

  throw std::invalid_argument("Optimizer not implemented");
  Optimizer *temp = nullptr;
  return temp;
}