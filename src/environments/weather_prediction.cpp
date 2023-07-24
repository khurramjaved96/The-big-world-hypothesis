//
// Created by Khurram Javed on 2023-07-11.
//

#include "../../include/environments/weather_prediction.h"
#include "../../include/rapidcsv.h"
#include "../../include/utils.h"
#include <fstream>
#include <iostream>
#include <string>

WeatherPrediction::WeatherPrediction() {
  std::cout << "Gets here\n";

  rapidcsv::Document doc("./weather_data/data.csv");
  std::vector<std::string> col = doc.GetColumn<std::string>("tempreture");
  std::vector<float> x = doc.GetColumn<float>("x");
  std::vector<float> y = doc.GetColumn<float>("y");
  std::vector<float> hour = doc.GetColumn<float>("time");
  std::vector<float> day = doc.GetColumn<float>("day");
  std::vector<float> month = doc.GetColumn<float>("month");
  std::vector<float> year = doc.GetColumn<float>("year");
  float max_temp = -200;
  float min_year = 3000;
  float min_temp = 200;
  for (int i = 0; i < col.size(); i++) {
    if (col[i].size() > 0) {
      std::vector<float> obs;
      float t = std::stof(col[i]);
      obs.push_back(std::stof(col[i]));
      if(t > max_temp)
        max_temp = t;
      if(t<min_temp)
        min_temp = t;
      if(year[i] < min_year)
        min_year = year[i];
      obs.push_back(year[i]);
      obs.push_back(month[i]);
      obs.push_back(day[i]);
      obs.push_back(hour[i]);
      dataset.push_back(obs);
    }
  }
  std::cout << "Dataset size = " << dataset.size() << std::endl;
  std::cout << max_temp << " " << min_temp << " " << min_year << std::endl;
  time = 0;
}

std::vector<float> WeatherPrediction::step() {
  this->time++;
  return this->get_state();
}

std::vector<float> WeatherPrediction::get_state() {
  std::vector<float> obs(230, 0);
  int year = 90;
  int month = 160;
  int day = 172;
  int hour = 203;
  obs[int(round(dataset[time][0]) + 50)] = 1;
  obs[year + int(dataset[time][1] - 1953)] = 1;
  obs[month + int(dataset[time][2])] = 1;
  obs[day + int(dataset[time][3])] = 1;
  obs[hour + int(dataset[time][4])] = 1;
  return obs;
}

float WeatherPrediction::get_target() {
  return round(dataset[time+1][0]) + 50;
}