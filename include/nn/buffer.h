//
// Created by Khurram Javed on 2023-07-13.
//

#ifndef INCLUDE_NN_BUFFER_H_
#define INCLUDE_NN_BUFFER_H_
#include <vector>
#include <random>

class Buffer{
protected:
  std::vector<std::vector<float>> observations;
  std::vector<float> targets;
  int current_index;
  int size;
  int seed;
  std::mt19937 mt;
  std::uniform_int_distribution<int> index_sampler;
public:
  Buffer(int size, int seed);
  void addSample(std::vector<float>& obs, float &target);
  std::pair<std::vector<float>, float> sample();
};

#endif // INCLUDE_NN_BUFFER_H_
