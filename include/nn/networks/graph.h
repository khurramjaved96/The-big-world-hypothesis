//
// Created by Khurram Javed on 2023-03-12.
//

#ifndef INCLUDE_NN_NETWORKS_GRAPH_H_
#define INCLUDE_NN_NETWORKS_GRAPH_H_
#include "vertex.h"
#include <random>
#include <string>
#include <vector>

class Vertex;

class Edge {
public:
  int edge_id;
  static int edge_id_generator;
  int from;
  int to;
  float gradient;
  float utility;
  float temp_gradient;
  float h = 0;
  float local_utility;
  float step_size;
  float weight;
  Edge(float weight, int from, int to, float step_size);
  float get_weight();
  int get_from();
  int get_to();
  bool is_recurrent();
};

class Graph {
protected:
  int input_vertices;

public:
  int GetInputVertices() const;
  void SetInputVertices(int input_vertices);
  std::mt19937 mt;
  int output_vertex_index;
  std::vector<float> prediction_logits;
  float softmax_noralization_term;
  std::vector<float> prediction_probabilites;
  int output_vertices;
  float prediction;
  std::vector<Vertex *> list_of_vertices;
  std::vector<int> get_distribution_of_values();
  void print_utility();
  Graph();
  Graph(int input_vertices, int seed);
  void add_edge(float weight, int from, int to, float step_size);
  void print_graph();
  void estimate_gradient(float error);
  float get_average_gradient(float);
  void set_input_values(std::vector<float> inp);
  float update_values();
  void decay_gradient(float decay_rate);
  void prune_weight();
  float get_prediction();
  void update_weights();
  void reset_feature();
  float compute_cross_entropy(std::vector<float> labels);
  std::string serialize_graph();
  virtual void update_utility() = 0;
};

class GraphLinearAssumptionUtility : public Graph {
protected:
  float utility_decay_rate;

public:
  GraphLinearAssumptionUtility(int input_vertices, int seed,

                               float utility_decay_rate);
  void update_utility() override;
};

class GradientUtility : public Graph {
protected:
  float utility_decay_rate;

public:
  GradientUtility(int input_vertices, int seed, float utility_decay_rate);
  void update_utility() override;
};

class GradientLocalUtility : public Graph {
protected:
  float utility_decay_rate;

public:
  GradientLocalUtility(int input_vertices, int seed,

                       float utility_decay_rate);
  void update_utility() override;
};

class UtilityPropagation : public Graph {
protected:
  float utility_decay_rate;

public:
  UtilityPropagation(int input_vertices, int seed, float utility_decay_rate);
  void update_utility() override;
};

class WeightUtility : public Graph {
public:
  WeightUtility(int input_vertices, int seed);
  void update_utility() override;
};

class RandomUtility : public Graph {
public:
  RandomUtility(int input_vertices, int seed);
  void update_utility() override;
};

class ActivationTrace : public Graph {
protected:
  float utility_decay_rate;

public:
  ActivationTrace(int input_vertices, int seed, float utility_decay_rate);
  void update_utility() override;
};

class GraphLocalUtility : public Graph {
protected:
  float utility_decay_rate;

public:
  GraphLocalUtility(int input_vertices, int seed, float utility_decay_rate);
  void update_utility() override;
};

#endif // INCLUDE_NN_NETWORKS_GRAPH_H_
