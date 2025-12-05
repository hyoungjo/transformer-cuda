#include "gpt2.hpp"
#include "operations.hpp"
#include "utils.hpp"
#include <cmath>
#include <iostream>

GPT2::GPT2(const std::string &path) {
  weights = utils::load_data(path);

  // transpose "lm_head.weight" for matmul
  operations::transpose(weights["lm_head.weight"]);
}

void GPT2::attention_block(Tensor &x, int layer_idx) {
  // std::cout << "[CPP][TRACE] Attention Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";

  Tensor x_norm = x;
  operations::layer_norm(x_norm, weights[prefix + "ln_1.weight"],
                         weights[prefix + "ln_1.bias"]);

  Tensor qkv; // seq_len x (3 * hidden_size)
  operations::matmul(qkv, x_norm, weights[prefix + "attn.c_attn.weight"]);
  operations::add_bias(qkv, weights[prefix + "attn.c_attn.bias"]);

  int64_t seq_len = x.shape[0];
  Tensor attention_value({seq_len, hidden_size});

  for (int64_t h = 0; h < num_heads; ++h) {
    /**
     * To follow the PyTorch implementation of the self-attention, the algorithm
     * splits the procedure token by token, instead of creating a separate Q, K,
     * and V matrices:
     *
     * 1. This avoids memory allocation for intermediate results. i.e.,
     * attention scores and attention coefficients.
     * 2. It uses explicit masking, which reduces the number of calculations in
     * half. However, this optimization is only effective to CPU-based
     * execution.
     */

    /**
     * The tensor `qkv` with dimension (seq_len, 3 * hidden_size) is a
     * concatenation of Q, K, and V matrices. Each Q, K, and V matrix is also a
     * concatenation of the h heads.
     *
     * [ Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h ]
     */
    int64_t q_offset = 0 * hidden_size + h * head_dim;
    int64_t k_offset = 1 * hidden_size + h * head_dim;
    int64_t v_offset = 2 * hidden_size + h * head_dim;

    for (int64_t t = 0; t < seq_len; ++t) {
      Tensor scores({t + 1});

      // t-th token query dot product with 0 ~ t-th token key
      for (int64_t i = 0; i <= t; ++i) {
        for (int64_t d = 0; d < head_dim; ++d) {
          float q_d = qkv(t, q_offset + d);
          float k_d = qkv(i, k_offset + d);
          scores(i) += q_d * k_d;
        }
        scores(i) /= std::sqrt((float)head_dim);
      }

      operations::softmax(scores);

      for (int64_t i = 0; i <= t; ++i) {
        for (int64_t d = 0; d < head_dim; ++d) {
          float v_d = qkv(i, v_offset + d);
          attention_value(t, h * head_dim + d) += scores(i) * v_d;
        }
      }
    }
  }

  Tensor attention_output;
  operations::matmul(attention_output, attention_value,
                     weights[prefix + "attn.c_proj.weight"]);
  operations::add_bias(attention_output, weights[prefix + "attn.c_proj.bias"]);

  operations::add(x, attention_output);
}

void GPT2::mlp_block(Tensor &x, int layer_idx) {
  // std::cout << "[CPP][TRACE] MLP Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";

  Tensor x_norm = x;
  operations::layer_norm(x_norm, weights[prefix + "ln_2.weight"],
                         weights[prefix + "ln_2.bias"]);

  Tensor x1;
  operations::matmul(x1, x_norm, weights[prefix + "mlp.c_fc.weight"]);
  operations::add_bias(x1, weights[prefix + "mlp.c_fc.bias"]);

  operations::gelu(x1);

  Tensor x2;
  operations::matmul(x2, x1, weights[prefix + "mlp.c_proj.weight"]);
  operations::add_bias(x2, weights[prefix + "mlp.c_proj.bias"]);

  operations::add(x, x2);
}

Tensor GPT2::forward(const std::vector<int> &input_ids) {
  // std::cout << "[CPP][TRACE] Beginning forward pass" << std::endl;
  int seq_len = static_cast<int>(input_ids.size());
  Tensor x({seq_len, hidden_size});

  Tensor &wte = weights["transformer.wte.weight"]; // vocab_size x hidden_size
  Tensor &wpe = weights["transformer.wpe.weight"]; // max_length x hidden_size
  if (seq_len > wpe.shape[0]) {
    std::cerr << "[CPP][ERROR] Input length " << seq_len
              << " exceeds max position embeddings " << wpe.shape[0]
              << std::endl;
    exit(1);
  }

  for (int t = 0; t < seq_len; ++t) {
    if (input_ids[t] < 0 || input_ids[t] >= wte.shape[0]) {
      std::cerr << "[CPP][ERROR] Token ID " << input_ids[t] << " out of bounds."
                << std::endl;
      exit(1);
    }
    for (int i = 0; i < hidden_size; ++i) {
      x(t, i) = wte(input_ids[t], i) + wpe(t, i);
    }
  }

  for (int i = 0; i < num_layers; ++i) {
    attention_block(x, i);
    mlp_block(x, i);
  }

  operations::layer_norm(x, weights["transformer.ln_f.weight"],
                         weights["transformer.ln_f.bias"]);

  Tensor prediction_token({1, hidden_size});
  for (int i = 0; i < hidden_size; ++i)
    prediction_token(i) = x(seq_len - 1, i);

  Tensor logits;
  operations::matmul(logits, prediction_token, weights["lm_head.weight"]);

  return logits;
}
