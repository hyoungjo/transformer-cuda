#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <map>
#include <string>

class GPT2 {
private:
  int64_t num_layers = 12;
  int64_t num_heads = 12;
  int64_t hidden_size = 768;
  int64_t mlp_size = 3072;
  int64_t head_dim = hidden_size / num_heads;
  int64_t vocab_size = 50257;

  std::map<std::string, Tensor> weights;

  Tensor x_norm;
  Tensor qkv;
  Tensor attention_value;
  Tensor attention_output;
  Tensor x1;
  Tensor x2;

  void attention_block(Tensor &x, int layer_idx);
  void mlp_block(Tensor &x, int layer_idx);

public:
  GPT2(const std::string &path);

  Tensor forward(int *input_ids, int seq_len);
};
