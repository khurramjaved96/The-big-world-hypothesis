//
// Created by Khurram Javed on 2022-11-14.
//

#ifndef INCLUDE_ENVIRONMENTS_ENVIRONMENT_FACTORY_H_
#define INCLUDE_ENVIRONMENTS_ENVIRONMENT_FACTORY_H_

#include "input_distribution.h"
#include <string>
#include "../../include/experiment/Experiment.h"

class EnvironmentFactory{
 public:
  static Environment* get_environment(std::string env_name, Experiment *my_experiment, int seed);
};
#endif //INCLUDE_ENVIRONMENTS_ENVIRONMENT_FACTORY_H_
