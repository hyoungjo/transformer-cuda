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
 * Calculates the attention output given the query, key, and value matrices.
 *
 * This is a fused multi-head attention kernel, that calculates the attention
 * score, applies causal masking and scaling, performs online softmax and value
 * aggregation.
 *
 * grid_dims: (seq_len, num_heads)
 * block_dims: (head_dim)
 *
 * A block calculates the attention output for a token, head pair.
 */
__global__ void multi_head_attention_kernel(float *output, const float *qkv,
                                            int seq_len, int hidden_size,
                                            int head_dim) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int dim_idx = threadIdx.x;

  // [ Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h ]
  int q_idx = token_idx * (3 * hidden_size) + 0 * hidden_size +
              head_idx * head_dim + dim_idx;
  float q_val = qkv[q_idx];
  float scale = 1.0f / sqrtf((float)head_dim);

  /**
   * Log-sum-exp trick is applied for numerical stability, and online softmax
   * is used to avoid storing the entire softmax matrix in memory.
   *
   * Let's iterate over past tokens t <= token_idx (Causal Mask)
   */
  float max_val = -1e9f; // maximum attention score
  float exp_sum = 0.0f;  // sum of expf(score_t - max_val)
  float out_val = 0.0f;  // sum of expf(score_t - max_val) * v_t

  for (int t = 0; t <= token_idx; ++t) {
    int k_idx =
        t * (3 * hidden_size) + 1 * hidden_size + head_idx * head_dim + dim_idx;
    float k_val = qkv[k_idx];
    float product = q_val * k_val;

    /**
     * Conduct warp-level reduction to obtain the attention score.
     * Calculate the sum of 'product' across the block (head_dim).
     */
    extern __shared__ float reduction_buffer[];

    unsigned int mask = 0xffffffff;
    float val = product;
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;
    if (lane_idx == 0) { // thread 0 of each warp
      reduction_buffer[warp_idx] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) { // aggregate warp results
      int num_warps = 1 + (head_dim - 1) / 32;
      for (int i = 1; i < num_warps; ++i) {
        reduction_buffer[0] += reduction_buffer[i];
      }
      reduction_buffer[0] *= scale;
    }
    __syncthreads();

    float score = reduction_buffer[0]; // all threads have the total score

    /**
     * Update max_val, exp_sum, and out_val for online softmax.
     * An update for exp_sum is also necessary to apply the new max_val.
     */
    float max_val_prev = max_val;
    max_val = fmaxf(max_val, score);
    float correction = expf(max_val_prev - max_val);
    float exp_score = expf(score - max_val);
    exp_sum = exp_sum * correction + exp_score;

    int v_idx =
        t * (3 * hidden_size) + 2 * hidden_size + head_idx * head_dim + dim_idx;
    float v_val = qkv[v_idx];
    out_val = out_val * correction + exp_score * v_val;
  }

  out_val /= exp_sum; // normalize

  /**
   * Save the attention output to global memory.
   * The output layout is [seq_len, hidden_size] (heads concatenated).
   */
  int out_idx = token_idx * hidden_size + head_idx * head_dim + dim_idx;
  output[out_idx] = out_val;
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
  Tensor out({hidden_size, vocab_size});
  operations::transpose(out, weights["lm_head.weight"]);
  weights["lm_head.weight"] = std::move(out);
}

void GPT2::attention_block(Tensor &x, int layer_idx) {
  // std::cout << "[CUDA][TRACE] Attention Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  Tensor x_norm = x;
  operations::layer_norm(x_norm, weights[prefix + "ln_1.weight"],
                         weights[prefix + "ln_1.bias"]);

  /**
   * The tensor `qkv` with dimension (seq_len, 3 * hidden_size) is a
   * concatenation of Q, K, and V matrices. Each Q, K, and V matrix is also a
   * concatenation of the h heads.
   *
   * [ Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h ]
   * where hidden_size = num_heads * head_dim
   */
  Tensor qkv({seq_len, 3 * hidden_size});
  operations::matmul(qkv, x_norm, weights[prefix + "attn.c_attn.weight"]);
  operations::add_bias(qkv, weights[prefix + "attn.c_attn.bias"]);

  Tensor attention_value({seq_len, hidden_size});

  dim3 block_dims(head_dim);
  dim3 grid_dims(seq_len, num_heads);
  int num_warps = 1 + (head_dim - 1) / 32;
  multi_head_attention_kernel<<<grid_dims, block_dims,
                                num_warps * sizeof(float)>>>(
      attention_value.d_data, qkv.d_data, seq_len, hidden_size, head_dim);

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

Tensor GPT2::forward(int *input_ids, int seq_len) {
  // std::cout << "[CUDA][TRACE] Beginning forward pass" << std::endl;
  Tensor x({seq_len, hidden_size});
  operations::embed(x, weights["transformer.wte.weight"],
                    weights["transformer.wpe.weight"], input_ids, seq_len,
                    hidden_size);

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
