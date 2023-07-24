//
// Created by Khurram Javed on 2023-03-12.
//

#include "../../include/nn/graphfactory.h"
#include "../../include/experiment/Experiment.h"
#include "../../include/nn/networks/graph.h"
#include <exception>
#include <string>

Graph *GraphFactory::get_graph(std::string graph_name,
                               Experiment *my_experiment, int seed) {
  //  return new Graph(my_experiment->get_int_param("vertices"),
  //  my_experiment->get_int_param("edges"),
  //  my_experiment->get_int_param("input_vertices"),
  //  my_experiment->get_int_param("seed"),
  //  my_experiment->get_string_param("non_linearity"));

  return new GraphLinearAssumptionUtility(
      my_experiment->get_int_param("input_vertices"), seed,
      my_experiment->get_float_param("utility_trace"));
}
