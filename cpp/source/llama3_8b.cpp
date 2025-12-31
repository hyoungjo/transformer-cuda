#include "llama3_8b.hpp"
#include "operations.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

LLaMA3_8B::LLaMA3_8B(const std::string &path) {
  weights = utils::load_data(path);
}

void LLaMA3_8B::attention_block(Tensor &x, int layer_idx) {
  // std::cout << "[CPP][TRACE] Attention Layer " << layer_idx << std::endl;
  std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  Tensor x_norm = x;
  operations::rms_norm(x_norm, weights[prefix + "input_layernorm.weight"]);

  /**
   * LLaMA uses separate query, key, value projections (without bias)
   */
  int64_t kv_dim = num_kv_heads * head_dim;
  Tensor q({seq_len, hidden_size});
  Tensor k({seq_len, kv_dim});
  Tensor v({seq_len, kv_dim});
  operations::matmul(q, x_norm, weights[prefix + "self_attn.q_proj.weight"],
                     true);
  operations::matmul(k, x_norm, weights[prefix + "self_attn.k_proj.weight"],
                     true);
  operations::matmul(v, x_norm, weights[prefix + "self_attn.v_proj.weight"],
                     true);

  operations::rope(q, head_dim);
  operations::rope(k, head_dim);

  /**
   * Grouped Query Attention (GQA)
   */
  Tensor attention_value({seq_len, hidden_size});

  int64_t group_size = num_heads / num_kv_heads;

  for (int64_t h = 0; h < num_heads; ++h) {
    int64_t kv_head = h / group_size;

    int64_t q_offset = h * head_dim;
    int64_t k_offset = kv_head * head_dim;
    int64_t v_offset = kv_head * head_dim;

    /**
     * The algorithm splits the procedure token by token, instead of creating a
     * separate Q, K, and V matrices:
     *
     * 1. This avoids memory allocation for intermediate results.
     *    i.e., attention scores and attention coefficients.
     * 2. It uses explicit masking i <= t, which reduces the number of
     *    calculations in half. However, this optimization is only effective to
     *    CPU-based execution.
     */
    for (int64_t t = 0; t < seq_len; ++t) {
      Tensor scores({t + 1}); // 0-indexed

      // t-th token query dot product with 0 ~ t-th token key
      for (int64_t i = 0; i <= t; ++i) {
        for (int64_t d = 0; d < head_dim; ++d) {
          float q_d = q(t, q_offset + d);
          float k_d = k(i, k_offset + d);
          scores(i) += q_d * k_d;
        }
        scores(i) /= std::sqrt((float)head_dim);
      }

      operations::softmax(scores);

      for (int64_t i = 0; i <= t; ++i) {
        for (int64_t d = 0; d < head_dim; ++d) {
          float v_d = v(i, v_offset + d);
          attention_value(t, h * head_dim + d) += scores(i) * v_d;
        }
      }
    }
  }

  Tensor attention_output({seq_len, hidden_size});
  operations::matmul(attention_output, attention_value,
                     weights[prefix + "self_attn.o_proj.weight"], true);

  operations::add(x, attention_output);
}

void LLaMA3_8B::mlp_block(Tensor &x, int layer_idx) {
  // std::cout << "[CPP][TRACE] MLP Layer " << layer_idx << std::endl;
  std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  Tensor x_norm({seq_len, hidden_size});
  operations::rms_norm(x_norm,
                       weights[prefix + "post_attention_layernorm.weight"]);

  Tensor gate({seq_len, mlp_size});
  Tensor up({seq_len, mlp_size});
  operations::matmul(gate, x_norm, weights[prefix + "mlp.gate_proj.weight"],
                     true);
  operations::matmul(up, x_norm, weights[prefix + "mlp.up_proj.weight"], true);

  operations::silu(gate);

  for (int64_t i = 0; i < gate.numel(); ++i) {
    gate(i) *= up(i); // element-wise, breaks linearity
  }

  Tensor down({seq_len, hidden_size});
  operations::matmul(down, gate, weights[prefix + "mlp.down_proj.weight"],
                     true);

  operations::add(x, down);
}

Tensor LLaMA3_8B::forward(const std::vector<int> &input_ids) {
  // std::cout << "[CPP][TRACE] Beginning forward pass" << std::endl;
  int seq_len = static_cast<int>(input_ids.size());
  Tensor x({seq_len, hidden_size});

  // vocab_size x hidden_size
  Tensor &embeddings = weights["model.embed_tokens.weight"];

  int vocab_size = embeddings.shape[0];
  for (int t = 0; t < seq_len; ++t) {
    if (input_ids[t] < 0 || input_ids[t] >= vocab_size) {
      std::cerr << "[CPP][ERROR] Token ID " << input_ids[t] << " out of bounds."
                << std::endl;
      exit(1);
    }
    for (int i = 0; i < hidden_size; ++i) {
      x(t, i) = embeddings(input_ids[t], i);
    }
  }

  for (int i = 0; i < num_layers; ++i) {
    attention_block(x, i);
    mlp_block(x, i);
  }

  operations::rms_norm(x, weights["model.norm.weight"]);

  Tensor prediction_token({1, hidden_size});
  for (int i = 0; i < hidden_size; ++i)
    prediction_token(i) = x(seq_len - 1, i);

  Tensor logits({1, vocab_size});
  operations::matmul(logits, prediction_token, weights["lm_head.weight"], true);

  return logits;
}
