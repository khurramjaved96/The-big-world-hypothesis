//
// Created by Khurram Javed on 2023-03-18.
//

#include "../../include/nn/architure_initializer.h"
#include <iostream>
#include <random>

Graph *ArchitectureInitializer::initialize_CELU_single_layer_network(
    Graph *my_graph, int total_parameters, int parameters_per_feature,
    std::string vertex_type, float step_size, int seed) {
  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.02, 0.02);
  int total_features = int(total_parameters / parameters_per_feature);
  int input_features = my_graph->list_of_vertices.size();
  auto sampler = std::uniform_int_distribution<int>(0, input_features - 1);
  for (int i = 0; i < total_features; i++) {
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex("linear"));

    for (int j = 0; j < parameters_per_feature; j++) {
      my_graph->add_edge(weight_sampler(mt), sampler(mt), my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  // Add CELU features

  int total_vertices_features = my_graph->list_of_vertices.size();
  for(int i = input_features; i < total_vertices_features; i++){
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex("relu"));
    my_graph->add_edge(1, i, my_graph->list_of_vertices.size() - 1,
                       0);
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex("relu"));
    my_graph->add_edge(-1, i, my_graph->list_of_vertices.size() - 1,
                       0);
  }
  int total_vertices_features_after = total_vertices_features;
  total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = total_vertices_features_after; j < total_vertices_features; j++) {
      my_graph->add_edge(weight_sampler(mt), j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}

Graph *
ArchitectureInitializer::initialize_linear_learning_network_list_of_seeds(
    Graph *my_graph, std::string vertex_type, float step_size, float lower,
    float higher, std::vector<int> seeds) {

  std::uniform_real_distribution<float> weight_sampler(lower, higher);
  int total_features = seeds.size();
  int input_features = my_graph->list_of_vertices.size();
  auto sampler = std::uniform_int_distribution<int>(0, input_features - 1);
  for (int i = 0; i < total_features; i++) {
    std::mt19937 mt_connections(seeds[i]);
    std::mt19937 mt_weights(seeds[i] * 2 + 2);
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex(vertex_type));

    for (int j = 0; j < 50; j++) {
      int conn = sampler(mt_connections);
      float weight = weight_sampler(mt_weights);
      //      std::cout << conn << ", " << weight << "\n";
      my_graph->add_edge(weight, conn, my_graph->list_of_vertices.size() - 1,
                         0);
    }
  }
  int total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = input_features; j < total_vertices_features; j++) {
      my_graph->add_edge(0, j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}

Graph *update_feature(Graph *my_graph, int index, int seed, float step_size,
                      float lower, float higher) {
  std::uniform_real_distribution<float> weight_sampler(lower, higher);
  int input_features = my_graph->list_of_vertices.size();
  auto sampler = std::uniform_int_distribution<int>(0, input_features - 1);
  std::mt19937 mt_connections(seed);
  std::mt19937 mt_weights(seed * 2 + 2);
  my_graph->list_of_vertices[index] =
      VertexFactory::get_vertex_with_seed("relu", seed);

  for (int j = 0; j < 50; j++) {
    int conn = sampler(mt_connections);
    float weight = weight_sampler(mt_weights);
    //      std::cout << conn << ", " << weight << "\n";
    my_graph->add_edge(weight, conn, index, step_size);
  }
  //  for (int i = my_graph->list_of_vertices.size() - 10; i <
  //  my_graph->list_of_vertices.size(); i++) {
  //    if(my_graph->list_of_vertices[i]->is_output == false){
  //      std::cout << "Bug in the code\n";
  //      exit(1);
  //    }
  //    for (auto &e : my_graph->list_of_vertices[i]->incoming_edges) {
  //      if(e.from == index){
  //        e.weight = 0;
  //        e.utility = 0;
  //        e.local_utility = 0;
  //      }
  //    }
  //  }

  for (int i = my_graph->list_of_vertices.size() - 10;
       i < my_graph->list_of_vertices.size(); i++) {
    if (my_graph->list_of_vertices[i]->is_output == false) {
      std::cout << "Bug in the code\n";
      exit(1);
    }
    for (auto &e : my_graph->list_of_vertices[i]->incoming_edges) {
      e.weight = 0;
      e.utility = 0;
      e.local_utility = 0;
    }
  }

  //  std::cout << "Feature added\n";
  return my_graph;
}

Graph *ArchitectureInitializer::initialize_fixed_features_network_list_of_seeds(
    Graph *my_graph, std::string vertex_type, float step_size, float lower,
    float higher, std::vector<int> seeds) {

  std::uniform_real_distribution<float> weight_sampler(lower, higher);
  int total_features = seeds.size();
  int input_features = my_graph->list_of_vertices.size();
  auto sampler = std::uniform_int_distribution<int>(0, input_features - 1);
  for (int i = 0; i < total_features; i++) {
    std::mt19937 mt_connections(seeds[i]);
    std::mt19937 mt_weights(seeds[i] * 2 + 2);
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex_with_seed(vertex_type, seeds[i]));

    for (int j = 0; j < 50; j++) {
      int conn = sampler(mt_connections);
      float weight = weight_sampler(mt_weights);
      //      std::cout << conn << ", " << weight << "\n";
      my_graph->add_edge(weight, conn, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  int total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = input_features; j < total_vertices_features; j++) {
      my_graph->add_edge(0, j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}

Graph *ArchitectureInitializer::initialize_sprase_networks(
    Graph *my_graph, int total_parameters, int parameters_per_feature,
    std::string vertex_type, float step_size, int seed) {
  std::mt19937 mt(seed);
  int total_features = int(total_parameters / parameters_per_feature);
  for (int i = 0; i < total_features; i++) {
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex(vertex_type));
    auto sampler = std::uniform_int_distribution<int>(
        0, my_graph->list_of_vertices.size() - 2);
    for (int j = 0; j < parameters_per_feature; j++) {
      my_graph->add_edge(0, sampler(mt), my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  int total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = 0; j < total_vertices_features; j++) {
      my_graph->add_edge(0, j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}

Graph *ArchitectureInitializer::initialize_single_layer_network(
    Graph *my_graph, int total_parameters, int parameters_per_feature,
    std::string vertex_type, float step_size, int seed) {
  std::mt19937 mt(seed);
  int total_features = int(total_parameters / parameters_per_feature);
  //  int total_features = 100;
  int input_features = my_graph->list_of_vertices.size();
  auto sampler = std::uniform_int_distribution<int>(0, input_features - 1);
  for (int i = 0; i < total_features; i++) {
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex(vertex_type));

    for (int j = 0; j < parameters_per_feature; j++) {
      my_graph->add_edge(0, sampler(mt), my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  int total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = input_features; j < total_vertices_features; j++) {
      my_graph->add_edge(0, j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}

Graph *ArchitectureInitializer::initialize_linear_learning_network(
    Graph *my_graph, int total_parameters, int parameters_per_feature,
    std::string vertex_type, float step_size, int seed) {
  std::mt19937 mt(seed);
  int total_features = int(total_parameters / parameters_per_feature);
  //  int total_features = 100;
  int input_features = my_graph->list_of_vertices.size();
  auto sampler = std::uniform_int_distribution<int>(0, input_features - 1);
  for (int i = 0; i < total_features; i++) {
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex(vertex_type));

    for (int j = 0; j < parameters_per_feature; j++) {
      my_graph->add_edge(0, sampler(mt), my_graph->list_of_vertices.size() - 1,
                         0);
    }
  }
  int total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = input_features; j < total_vertices_features; j++) {
      my_graph->add_edge(0, j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}