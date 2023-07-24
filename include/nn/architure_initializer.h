//
// Created by Khurram Javed on 2023-03-18.
//

#ifndef INCLUDE_NN_ARCHITURE_INITIALIZER_H_
#define INCLUDE_NN_ARCHITURE_INITIALIZER_H_
#include "networks/graph.h"
#include <string>

class ArchitectureInitializer {
public:
  ArchitectureInitializer() = default;
  Graph *initialize_sprase_networks(Graph *my_graph, int total_parameters,
                                    int parameters_per_feature,
                                    std::string vertex_type, float step_size,
                                    int seed);
  Graph *initialize_single_layer_network(Graph *my_graph, int total_parameters,
                                         int parameters_per_feature,
                                         std::string vertex_type,
                                         float step_size, int seed);
  Graph *initialize_linear_learning_network(Graph *my_graph,
                                            int total_parameters,
                                            int parameters_per_feature,
                                            std::string vertex_type,
                                            float step_size, int seed);
  Graph *initialize_linear_learning_network_list_of_seeds(
      Graph *my_graph, std::string vertex_type, float step_size,  float lower, float higher,
      std::vector<int> seeds);

  Graph *initialize_CELU_single_layer_network(Graph *my_graph, int total_parameters,
                                         int parameters_per_feature,
                                         std::string vertex_type,
                                         float step_size, int seed);


  Graph *initialize_fixed_features_network_list_of_seeds(
      Graph *my_graph, std::string vertex_type, float step_size, float lower, float higher,
      std::vector<int> seeds);
};

Graph *update_feature(Graph *my_graph, int index, int seed, float step_size,
                      float lower, float higher);

Graph *freeze_feature(Graph *my_graph, int index,
                      float lower, float higher);

#endif // INCLUDE_NN_ARCHITURE_INITIALIZER_H_
