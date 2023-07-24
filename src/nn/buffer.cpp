//
// Created by Khurram Javed on 2023-07-13.
//

#include "../../include/nn/buffer.h"

Buffer::Buffer(int size, int seed) : index_sampler(0, size - 1), mt(seed) {
  this->size = size;
  this->current_index = 0;
}

void Buffer::addSample(std::vector<float> &obs, float &target) {
  if (observations.size() < this->size) {
    observations.push_back(obs);
    targets.push_back(target);
  } else {
    observations[this->current_index] = obs;
    targets[this->current_index] = target;
  }
  this->current_index = (this->current_index + 1) % this->size;
}

std::pair<std::vector<float>, float> Buffer::sample() {
  std::pair<std::vector<float>, float> sample;
  int data_index = index_sampler(mt);
  if(observations.size() < this->size){
    data_index = data_index%observations.size();
  }
  sample.first = this->observations[data_index];
  sample.second = this->targets[data_index];
  return sample;
}

