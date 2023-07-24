//
// Created by Khurram Javed on 2023-03-12.
//


#include "../../include/environments/input_distribution.h"
#include "../../include/environments/mnist/mnist_reader.hpp"
#include "math.h"
PatternEnvironment::PatternEnvironment(int number_of_samples, float target_mean,
                                       float range, int seed)
    : sampler(target_mean - range, target_mean + range),
      Environment(number_of_samples, seed) {

  for (int i = 0; i < number_of_samples; i++) {
    std::vector<float> x(number_of_samples, 0);
    x[i] = 1;
    this->input_list.push_back(x);
    this->target_list.push_back(sampler(this->mt));
  }
}

void Environment::step() { this->current_sample = sample_index(this->mt); }


std::vector<float> Environment::get_features() {
  return this->input_list[this->current_sample];
}

float Environment::get_target() {
  return this->target_list[this->current_sample];
}

std::vector<std::vector<float>> Environment::get_all_x() {
  return this->input_list;
}

std::vector<float> Environment::get_all_y() { return this->target_list; }

Environment::Environment(int total_samples, int seed)
    : mt(seed), sample_index(0, total_samples - 1) {
  this->number_of_samples = total_samples;
  this->current_sample = 0;
}

XOREnvironment::XOREnvironment(int seed) : Environment(4, seed) {
  this->input_list.push_back(std::vector<float>{1, 0, 1});
  this->input_list.push_back(std::vector<float>{1, 1, 0});
  this->input_list.push_back(std::vector<float>{1, 1, 1});
  this->input_list.push_back(std::vector<float>{1, 0, 0});
  this->target_list.push_back(11);
  this->target_list.push_back(11);
  this->target_list.push_back(10);
  this->target_list.push_back(10);
}

std::vector<float> MNISTEnviroment::get_one_hot_target() {
  int t = int(this->get_target());
  std::vector<float> target_vector(10, 0);
  target_vector[t] = 1;
  return target_vector;
}

MNISTEnviroment::MNISTEnviroment(int seed) : Environment(60000, seed) {
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");
  for (int counter = 0; counter < 60000; counter++) {
    std::vector<float> x_temp;
    for (auto inner : dataset.training_images[counter]) {
      x_temp.push_back(float(unsigned(inner)) / 255);
    }
    this->input_list.push_back(x_temp);
    this->target_list.push_back(
        float(unsigned(dataset.training_labels[counter])));
  }
}

MNISTEnviromentOnline::MNISTEnviromentOnline(int seed): MNISTEnviroment(seed) {
  this->T = 0;
  update_targets();
}

void MNISTEnviromentOnline::update_targets() {
  this->p1 = sample_index(this->mt)%10;
  this->p2 = sample_index(this->mt)%10;
  while(this->p1 == this->p2)
    this->p2 = sample_index(this->mt)%10;
}

void MNISTEnviromentOnline::step() {
  this->T++;
  this->current_sample = sample_index(this->mt);
  while(this->target_list[this->current_sample] != p1 && this->target_list[this->current_sample] != p2)
    this->current_sample = sample_index(this->mt);
  if(T%10000 == 0)
    update_targets();
//  }
}


std::vector<float> MNISTTestEnviroment::get_one_hot_target() {
  int t = int(this->get_target());
  std::vector<float> target_vector(10, 0);
  target_vector[t] = 1;
  return target_vector;
}

MNISTTestEnviroment::MNISTTestEnviroment(int seed) : Environment(10000, seed) {
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");
  for (int counter = 0; counter < 10000; counter++) {
    std::vector<float> x_temp;
    for (auto inner : dataset.test_images[counter]) {
      x_temp.push_back(float(unsigned(inner)) / 255);
    }
    this->input_list.push_back(x_temp);
    this->target_list.push_back(float(unsigned(dataset.test_labels[counter])));
  }
}
