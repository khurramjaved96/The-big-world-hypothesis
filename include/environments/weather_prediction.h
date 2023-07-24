//
// Created by Khurram Javed on 2023-07-11.
//

#ifndef INCLUDE_ENVIRONMENTS_WEATHER_PREDICTION_H_
#define INCLUDE_ENVIRONMENTS_WEATHER_PREDICTION_H_

#include <fstream>
#include <string>
#include <vector>

class WeatherPrediction {
private:
  std::vector<float> observation;
  std::vector<std::vector<float>> dataset;
  std::ifstream myfile;

  std::vector<float> col;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> hour;
  std::vector<float> day;
  std::vector<float> month ;
  std::vector<float> year ;
public:
  int time;
  float gamma;
  WeatherPrediction();
  std::vector<float> get_state();
  std::vector<float> step();
  float get_target();
};

#endif // INCLUDE_ENVIRONMENTS_WEATHER_PREDICTION_H_
