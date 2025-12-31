#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

class LLaMA3_8B : public Model {
private:
  int64_t num_layers = 32;
  int64_t num_heads = 32;
  int64_t num_kv_heads = 8; // Grouped Query Attention (GQA)
  int64_t head_dim = 128;   // hidden_size / num_heads
  int64_t hidden_size = 4096;

  void attention_block(Tensor &x, int layer_idx);
  void mlp_block(Tensor &x, int layer_idx);

public:
  LLaMA3_8B(const std::string &path);

  Tensor forward(const std::vector<int> &input_ids);
};
