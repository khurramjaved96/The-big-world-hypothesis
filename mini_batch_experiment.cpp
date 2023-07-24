//
// Created by Khurram Javed on 2023-03-12.
//

#include "include/environments/input_distribution.h"
#include "include/nn/networks/graph.h"
#include "include/nn/networks/vertex.h"
#include <iostream>
#include <random>
#include <vector>

#include "include/environments/environment_factory.h"
#include "include/environments/mnist/mnist_reader.hpp"
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

  Metric error_metric = Metric(
      my_experiment->database_name, "error",
      std::vector<std::string>{"run", "step", "train_error", "train_accuracy",
                               "test_error", "test_accuracy"},
      std::vector<std::string>{"int", "int", "real", "real", "real", "real"},
      std::vector<std::string>{"run", "step"});

  MNISTEnviroment *env =
      new MNISTEnviroment(my_experiment->get_int_param("seed"));
  MNISTTestEnviroment *test_env =
      new MNISTTestEnviroment(my_experiment->get_int_param("seed"));
  int win = 0;
  Buffer buff(10000, 0);
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
  } else if (my_experiment->get_string_param("network") == "fixed_features") {
    network = network_initializer.initialize_single_layer_network(
        network,
        my_experiment->get_int_param("parameters") /
            my_experiment->get_int_param("batch_size"),
        my_experiment->get_int_param("density"),
        my_experiment->get_string_param("non_linearity"),
        my_experiment->get_float_param("step_size"),
        my_experiment->get_int_param("seed"));
  } else if (my_experiment->get_string_param("network") == "linear_learning") {
    network = network_initializer.initialize_linear_learning_network(
        network,
        my_experiment->get_int_param("parameters") /
            my_experiment->get_int_param("batch_size"),
        my_experiment->get_int_param("density"),
        my_experiment->get_string_param("non_linearity"),
        my_experiment->get_float_param("step_size"),
        my_experiment->get_int_param("seed"));
  }

  Optimizer *opti = OptimizerFactory::get_optimizer(network, my_experiment);
  WeightInitializer weight_initializer(my_experiment->get_float_param("lower"),
                                       my_experiment->get_float_param("higher"),
                                       seed);
  network = weight_initializer.initialize_weights(network);
  float train_error = 5;
  bool check_flag = false;
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
    env->step();
    auto inps = env->get_features();
    float target = env->get_target();
    buff.addSample(inps, target);
    network->decay_gradient(0);
    if (i > 100) {
      for (int batch = 0; batch < my_experiment->get_int_param("batch_size");
           batch++) {
        auto d = buff.sample();
        auto inp_t = d.first;
        float target_t = d.second;
        network->set_input_values(inp_t);
        float prediction = network->update_values();
        network->estimate_gradient(target_t);
      }
      opti->update_weights(network);
    }

    if (i % my_experiment->get_int_param("frequency") ==
        my_experiment->get_int_param("frequency") - 1) {
      check_flag = false;
      //      Evaluate error on the train set
      float correct = 0;
      auto list_of_x = env->get_all_x();
      auto list_of_targets = env->get_all_y();
      float total_error = 0;
      for (int temp = 0; temp < list_of_x.size(); temp++) {
        auto temp_x = list_of_x[temp];
        float temp_target = list_of_targets[temp];
        network->set_input_values(temp_x);
        float temp_prediction = network->update_values();
        int target = int(list_of_targets[temp]);
        total_error += -log(network->prediction_probabilites[target]);
        if (temp_prediction == target)
          correct++;
      }
      total_error /= list_of_x.size();
      train_error = total_error;
//      std::cout << "Train error " << train_error << std::endl;
      std::vector<std::string> val;
      val.push_back(std::to_string(my_experiment->get_int_param("run")));
      val.push_back(std::to_string(i));
      val.push_back(std::to_string(total_error));
      val.push_back(std::to_string(correct / list_of_x.size()));

      correct = 0;
      list_of_x = test_env->get_all_x();
      list_of_targets = test_env->get_all_y();
      total_error = 0;
      for (int temp = 0; temp < list_of_x.size(); temp++) {
        auto temp_x = list_of_x[temp];
        float temp_target = list_of_targets[temp];
        network->set_input_values(temp_x);
        float temp_prediction = network->update_values();
        int target = int(list_of_targets[temp]);
        total_error += -log(network->prediction_probabilites[target]);
        if (temp_prediction == target)
          correct++;
      }
      total_error /= list_of_x.size();
      val.push_back(std::to_string(total_error));
      val.push_back(std::to_string(correct / list_of_x.size()));
      error_metric.record_value(val);
      error_metric.commit_values();
    }
  }

  error_metric.commit_values();
}
