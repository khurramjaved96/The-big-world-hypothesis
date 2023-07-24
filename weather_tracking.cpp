//
// Created by Khurram Javed on 2023-03-12.
//

#include "include/environments/input_distribution.h"
#include "include/environments/weather_prediction.h"
#include "include/nn/networks/graph.h"
#include "include/nn/networks/vertex.h"
#include <iostream>
#include <random>
#include <vector>

#include "include/environments/environment_factory.h"
#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/weather_prediction.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include "include/nn/architure_initializer.h"
#include "include/nn/buffer.h"
#include "include/nn/graphfactory.h"
#include "include/nn/optimizer_factory.h"
#include "include/nn/weight_initializer.h"
#include "include/nn/weight_optimizer.h"
#include "include/utils.h"
#include <random>
#include <string>

int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric =
      Metric(my_experiment->database_name, "error",
             std::vector<std::string>{"run", "step", "lifetime_error"},
             std::vector<std::string>{"int", "int", "real"},
             std::vector<std::string>{"run", "step"});

  WeatherPrediction *env = new WeatherPrediction();
  int win = 0;

  Buffer buff(my_experiment->get_int_param("buffer_size"), 0);
  std::cout << "Buff created\n";
  int seed = my_experiment->get_int_param("seed");
  std::mt19937 mt(seed);
  Graph *network = GraphFactory::get_graph(
      "", my_experiment, my_experiment->get_int_param("seed"));

  auto network_initializer = ArchitectureInitializer();

  if (my_experiment->get_string_param("network") == "multilayer") {
    network = network_initializer.initialize_sprase_networks(
        network,
        my_experiment->get_int_param("parameters") /
            my_experiment->get_int_param("batch_size"),
        my_experiment->get_int_param("density"),
        my_experiment->get_string_param("non_linearity"),
        my_experiment->get_float_param("step_size"),
        my_experiment->get_int_param("seed"));
  } else if (my_experiment->get_string_param("network") == "celu") {
    network = network_initializer.initialize_CELU_single_layer_network(
        network,
        my_experiment->get_int_param("parameters") /
            my_experiment->get_int_param("batch_size"),
        my_experiment->get_int_param("density"),
        my_experiment->get_string_param("non_linearity"),
        my_experiment->get_float_param("step_size"),
        my_experiment->get_int_param("seed"));
  }
  Optimizer *opti = OptimizerFactory::get_optimizer(network, my_experiment);
  double lifetime_error = 0;
  long long int life_time_step = 1;
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
    auto inps = env->step();
//    std::cout << "inp len = " << inps.size() << std::endl;
    network->set_input_values(inps);
    float prediction = network->update_values();
    float target = env->get_target();

    lifetime_error =
        lifetime_error +
        (-log(network->prediction_probabilites[target]) - lifetime_error) /
            life_time_step;
    life_time_step++;
    if (i % my_experiment->get_int_param("frequency") ==
        my_experiment->get_int_param("frequency") - 1) {
      std::cout << "Error = " << lifetime_error << std::endl;
      std::vector<std::string> val;
      val.push_back(std::to_string(my_experiment->get_int_param("run")));
      val.push_back(std::to_string(i));
      val.push_back(std::to_string(lifetime_error));
      error_metric.record_value(val);
      error_metric.commit_values();
    }

    buff.addSample(inps, target);
    network->decay_gradient(0);
    if (i > 100) {
      for (int batch = 0; batch < my_experiment->get_int_param("batch_size");
           batch++) {
        auto d = buff.sample();
        network->set_input_values(d.first);
        float prediction = network->update_values();
        network->estimate_gradient(d.second);
      }
      opti->update_weights(network);
    }
  }
  error_metric.commit_values();
}
