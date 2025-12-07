#include "gpt2.hpp"
#include "operations.hpp"
#include "utils.hpp"
#include <cmath>
#include <cuda_runtime.h>

/**
 * ============================================================
 * =================== Helper CUDA Kernels ====================
 * ============================================================
 */

/**
 * Extract specific Q, K, V head from the QKV tensor
 * [Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h]
 *
 * section_idx: Q = 0, K = 1, V = 2
 */
__global__ void extract_head_kernel(float *out, const float *qkv, int seq_len,
                                    int hidden_size, int head_dim, int head_idx,
                                    int section_idx) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < head_dim) {
    int col_offset = section_idx * hidden_size + head_idx * head_dim;

    int input_idx = row * (3 * hidden_size) + col_offset + col;
    int output_idx = row * head_dim + col;

    out[output_idx] = qkv[input_idx];
  }
}

/**
 * Save the attention head's final value
 * [H_1, H_2, .., H_h]
 */
__global__ void insert_head_kernel(float *out, const float *head_out,
                                   int seq_len, int hidden_size, int head_dim,
                                   int head_idx) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < head_dim) {
    int col_offset = head_idx * head_dim;

    int input_idx = row * head_dim + col; // head
    int output_idx = row * hidden_size + col_offset + col;

    out[output_idx] = head_out[input_idx];
  }
}

/**
 * Apply causal mask to the attention scores and scale.
 *
 * Q * K^T / sqrt(d_k)
 */
__global__ void causal_mask_and_scale_kernel(float *scores, int seq_len,
                                             float scale) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < seq_len) {
    int idx = row * seq_len + col;
    if (row < col) {
      scores[idx] = -1e9f; // mask future tokens
    } else {
      scores[idx] *= scale;
    }
  }
}

/**
 * Extract the last token's hidden state
 */
__global__ void extract_last_token_kernel(float *out, const float *in,
                                          int seq_len, int hidden_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_size) {
    out[i] = in[(seq_len - 1) * hidden_size + i];
  }
}

/**
 * ============================================================
 * ================ GPT2 Class Implementation =================
 * ============================================================
 */

GPT2::GPT2(const std::string &path) {
  weights = utils::load_data(path, "gpu");

  // transpose "lm_head.weight" for matmul
  operations::transpose(weights["lm_head.weight"]);
}

void GPT2::attention_block(Tensor &x, int layer_idx) {
  // std::cout << "[CUDA][TRACE] Attention Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  Tensor x_norm = x;
  operations::layer_norm(x_norm, weights[prefix + "ln_1.weight"],
                         weights[prefix + "ln_1.bias"]);

  Tensor qkv({seq_len, 3 * hidden_size});
  operations::matmul(qkv, x_norm, weights[prefix + "attn.c_attn.weight"]);
  operations::add_bias(qkv, weights[prefix + "attn.c_attn.bias"]);

  /**
   * The tensor `qkv` with dimension (seq_len, 3 * hidden_size) is a
   * concatenation of Q, K, and V matrices. Each Q, K, and V matrix is also a
   * concatenation of the h heads.
   *
   * [ Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h ]
   */
  Tensor q_head({seq_len, head_dim});
  Tensor k_head({seq_len, head_dim});
  Tensor v_head({seq_len, head_dim});
  Tensor attention_head_scores({seq_len, seq_len});
  Tensor attention_head_values({seq_len, head_dim});

  Tensor attention_value({seq_len, hidden_size}); // heads concatenated

  dim3 block_dims(16, 16);
  dim3 head_grid_dims(1 + (seq_len - 1) / 16, 1 + (head_dim - 1) / 16);
  dim3 score_grid_dims(1 + (seq_len - 1) / 16, 1 + (seq_len - 1) / 16);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int h = 0; h < num_heads; ++h) {
    // 1. Extract Q, K, V heads
    extract_head_kernel<<<head_grid_dims, block_dims>>>(
        q_head.d_data, qkv.d_data, seq_len, hidden_size, head_dim, h, 0);
    extract_head_kernel<<<head_grid_dims, block_dims>>>(
        k_head.d_data, qkv.d_data, seq_len, hidden_size, head_dim, h, 1);
    extract_head_kernel<<<head_grid_dims, block_dims>>>(
        v_head.d_data, qkv.d_data, seq_len, hidden_size, head_dim, h, 2);

    // 2. Compute attention scores
    operations::transpose(k_head);
    operations::matmul(attention_head_scores, q_head,
                       k_head); // Score = Q * K^T
    operations::transpose(k_head);

    causal_mask_and_scale_kernel<<<score_grid_dims, block_dims>>>(
        attention_head_scores.d_data, seq_len, scale);

    // 3. Compute attention coefficients
    operations::softmax(attention_head_scores);

    // 4. Compute attention value
    operations::matmul(attention_head_values, attention_head_scores, v_head);

    // 5. Insert back into main buffer
    insert_head_kernel<<<head_grid_dims, block_dims>>>(
        attention_value.d_data, attention_head_values.d_data, seq_len,
        hidden_size, head_dim, h);
  }

  Tensor attention_output({seq_len, hidden_size});
  operations::matmul(attention_output, attention_value,
                     weights[prefix + "attn.c_proj.weight"]);
  operations::add_bias(attention_output, weights[prefix + "attn.c_proj.bias"]);

  operations::add(x, attention_output);
}

void GPT2::mlp_block(Tensor &x, int layer_idx) {
  // std::cout << "[CUDA][TRACE] MLP Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  Tensor x_norm = x;
  operations::layer_norm(x_norm, weights[prefix + "ln_2.weight"],
                         weights[prefix + "ln_2.bias"]);

  Tensor x1({seq_len, mlp_size});
  operations::matmul(x1, x_norm, weights[prefix + "mlp.c_fc.weight"]);
  operations::add_bias(x1, weights[prefix + "mlp.c_fc.bias"]);

  operations::gelu(x1);

  Tensor x2({seq_len, hidden_size});
  operations::matmul(x2, x1, weights[prefix + "mlp.c_proj.weight"]);
  operations::add_bias(x2, weights[prefix + "mlp.c_proj.bias"]);

  operations::add(x, x2);
}

Tensor GPT2::forward(const std::vector<int> &input_ids) {
  // std::cout << "[CUDA][TRACE] Beginning forward pass" << std::endl;
  int seq_len = static_cast<int>(input_ids.size());
  Tensor x({seq_len, hidden_size});
  operations::embedding(x, weights["transformer.wte.weight"],
                        weights["transformer.wpe.weight"], input_ids);

  for (int i = 0; i < num_layers; ++i) {
    attention_block(x, i);
    mlp_block(x, i);
  }

  operations::layer_norm(x, weights["transformer.ln_f.weight"],
                         weights["transformer.ln_f.bias"]);

  Tensor prediction_token({1, hidden_size});
  dim3 block_dims(256);
  dim3 grid_dims((hidden_size + 256 - 1) / 256);
  extract_last_token_kernel<<<grid_dims, block_dims>>>(
      prediction_token.d_data, x.d_data, seq_len, hidden_size);

  Tensor logits({1, vocab_size});
  operations::matmul(logits, prediction_token, weights["lm_head.weight"]);

  return logits;
}
