#pragma once

#include "tensor.hpp"
#include <map>
#include <string>

class Model {
protected:
  std::map<std::string, Tensor> weights;

public:
  virtual ~Model() = default;
  // pure virtual function, force the derived class to implement `forward`
  virtual Tensor forward(int *input_ids, int seq_len) = 0;
};
