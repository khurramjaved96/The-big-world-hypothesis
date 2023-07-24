//
// Created by Khurram Javed on 2023-03-12.
//

#include "../../include/environments/environment_factory.h"
#include "../../include/environments/input_distribution.h"
#include "../../include/experiment/Experiment.h"
#include <string>

Environment *EnvironmentFactory::get_environment(std::string env_name, Experiment *my_experiment, int seed) {
  if(env_name == "xor")
    return new XOREnvironment(seed);
  else if(env_name == "pattern")
    return new PatternEnvironment(my_experiment->get_int_param("input_vertices"),
                                  my_experiment->get_float_param("target_mean"),
                                  my_experiment->get_float_param("target_range"),
                                  seed);
  else if(env_name == "mnist")
    return new MNISTEnviroment(my_experiment->get_int_param("seed"));
  return nullptr;
}