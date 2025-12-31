#pragma once

#include "tensor.hpp"
#include <map>
#include <string>
#include <vector>

class Model {
protected:
  std::map<std::string, Tensor> weights;

public:
  virtual ~Model() = default;
  // pure virtual function, force the derived class to implement `forward`
  virtual Tensor forward(const std::vector<int> &input_ids) = 0;
};
