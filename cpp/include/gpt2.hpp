#pragma once

#include "tensor.hpp"
#include <map>
#include <string>
#include <vector>

class GPT2 {
private:
  int64_t num_layers = 12;
  int64_t num_heads = 12;
  int64_t hidden_size = 768;
  int64_t head_dim = hidden_size / num_heads;

  std::map<std::string, Tensor> weights;

  void attention_block(Tensor &x, int layer_idx);
  void mlp_block(Tensor &x, int layer_idx);

public:
  GPT2(const std::string &path);

  Tensor forward(const std::vector<int> &input_ids);
};
