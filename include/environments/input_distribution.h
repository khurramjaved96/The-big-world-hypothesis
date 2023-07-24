//
// Created by Khurram Javed on 2022-11-08.
//

#ifndef INCLUDE_NN_INPUT_DISTRIBUTIO_H_
#define INCLUDE_NN_INPUT_DISTRIBUTIO_H_

#include <vector>
#include <random>

#include <string>
#include <vector>


class Environment {
 protected:
  std::vector<float> target_list;
  std::vector<std::vector<float>> input_list;
  int current_sample;
  int number_of_samples;
  std::mt19937 mt;
  std::uniform_int_distribution<int> sample_index;
 public:
  Environment(int total_samples, int seed);
  virtual std::vector<float> get_features();
  virtual void step();
  virtual std::vector<std::vector<float>> get_all_x();
  virtual std::vector<float> get_all_y();
  virtual float get_target();
};

class MNISTEnviroment : public Environment{
public:
  MNISTEnviroment(int seed);
  std::vector<float> get_one_hot_target();
};

class MNISTEnviromentOnline : public MNISTEnviroment{
  int p1;
  int p2;
  long long int T;
  void update_targets();
public:
  MNISTEnviromentOnline(int seed);
  void step() override;
};


class MNISTTestEnviroment : public Environment{
public:
  MNISTTestEnviroment(int seed);
  std::vector<float> get_one_hot_target();

};

class PatternEnvironment : public Environment {
 protected:
  int number_of_features;
  std::uniform_real_distribution<float> sampler;
 public:
  PatternEnvironment(int number_of_samples, float target_mean, float range, int seed);

};

class XOREnvironment : public Environment {
 public:
  XOREnvironment(int seed);
};


#endif //INCLUDE_NN_INPUT_DISTRIBUTIO_H_
