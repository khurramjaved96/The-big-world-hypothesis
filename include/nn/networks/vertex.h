//
// Created by Khurram Javed on 2023-03-12.
//

#ifndef INCLUDE_NN_NETWORKS_VERTEX_H_
#define INCLUDE_NN_NETWORKS_VERTEX_H_
#include "graph.h"
#include <string>
#include <vector>

class Edge;
class Vertex {
protected:
public:
  int age;
  int id;
  float value;
  std::string type;
  bool is_output;
  float sum_of_outgoing_weights;
  float utility;
  float new_util;
  float utility_trace;
  float d_out_d_vertex;
  float d_out_d_vertex_before_non_linearity;
  static int id_generator;
  Vertex();
  std::vector<Edge> incoming_edges;
  virtual float forward_with_val(float value);
  virtual float forward();
  float get_value();
  virtual float backward(float val);
  float max_value;
  float min_value;
};

class ReluVertex : public Vertex {

public:
  int seed;
  ReluVertex();
  ReluVertex(int seed);
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class NormalizedRelu : public Vertex {
public:
  float mean;
  float variance;
  float decay_rate;
  NormalizedRelu();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class SigmoidVertex : public Vertex {
  static float sigmoid(float x);

public:
  SigmoidVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class TanHVertex : public Vertex {
  static float tanh(float x);

public:
  TanHVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class LeakyReluVertex : public Vertex {
public:
  LeakyReluVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class BinaryVertex : public Vertex {
public:
  BinaryVertex();
  float forward_with_val(float val) override;
  float forward() override;
  float backward(float val) override;
};

class VertexFactory {
public:
  static Vertex *get_vertex(const std::string &type);
  static Vertex *get_vertex_with_seed(const std::string &type, int seed);
};

#endif // INCLUDE_NN_NETWORKS_VERTEX_H_
