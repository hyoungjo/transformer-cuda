#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include <string>

class LLaMA3_8B : public Model {
private:
  int64_t num_layers = 32;
  int64_t num_heads = 32;
  int64_t num_kv_heads = 8; // Grouped Query Attention (GQA)
  int64_t head_dim = 128;   // hidden_size / num_heads
  int64_t hidden_size = 4096;

  int64_t mlp_size = 14336;
  int64_t vocab_size = 128256;

  Tensor x_norm;
  Tensor q, k, v;
  Tensor attention_value;
  Tensor attention_output;
  Tensor gate, up, down;

  void attention_block(Tensor &x, int layer_idx);
  void mlp_block(Tensor &x, int layer_idx);

public:
  LLaMA3_8B(const std::string &path);

  Tensor forward(int *input_ids, int seq_len);
};
