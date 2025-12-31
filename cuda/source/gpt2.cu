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
 * Extract a query, key, or value head from the concatenated `qkv` tensor.
 *
 * [Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h]
 * section_idx: Q = 0, K = 1, V = 2
 */
static __global__ void extract_head_kernel(float *head, const float *qkv,
                                           int seq_len, int hidden_size,
                                           int head_dim, int head_idx,
                                           int section_idx) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < head_dim) {
    int row_offset = row * (3 * hidden_size);
    int head_offset = section_idx * hidden_size + head_idx * head_dim;
    int input_idx = row_offset + head_offset + col;
    int output_idx = row * head_dim + col;
    head[output_idx] = qkv[input_idx];
  }
}

/**
 * Insert an attention head's values to the output tensor.
 *
 * [H_1, H_2, .., H_h]
 */
static __global__ void insert_head_kernel(float *attention_values,
                                          const float *head, int seq_len,
                                          int hidden_size, int head_dim,
                                          int head_idx) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < head_dim) {
    int row_offset = row * hidden_size;
    int head_offset = head_idx * head_dim;
    int input_idx = row * head_dim + col;
    int output_idx = row_offset + head_offset + col;
    attention_values[output_idx] = head[input_idx];
  }
}

/**
 * Apply causal mask to disable attentions future tokens and apply scaling of
 * sqrt(d_k) for normalization.
 *
 * Q * K^T / sqrt(d_k)
 */
static __global__ void causal_mask_and_scale_kernel(float *scores, int seq_len,
                                                    float scale) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < seq_len) {
    int idx = row * seq_len + col;
    if (row < col) {
      scores[idx] = -1e9f;
    } else {
      scores[idx] *= scale;
    }
  }
}

/**
 * Calculates the attention output from query, key, and value matrices.
 * A block calculates the attention output for a token, head pair.
 *
 * The kernel uses a fused attention mechanism, where it integrates several
 * previously discrete operations into a single execution unit. It combines
 * matrix multiplication for calculating the attention scores, causal masking
 * and scaling, softmax via online log-sum-exp trick, and value aggregation.
 *
 * By fusing these operations, the implementation removes intermediate global
 * memory reads and writes between different kernel launches, resulting in
 * significant performance improvements over a naive implementation.
 *
 * A naive version is written to simply merge separate kernel operations, and
 * therefore each block runs with a single thread calculating the attention
 * output for a token, head pair.
 */

static __global__ void naive_fused_attention_kernel(float *output,
                                                    const float *qkv,
                                                    int hidden_size,
                                                    int head_dim) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  auto calculate_qkv_offset = [&](int token_idx, int qkv_idx, int head_idx,
                                  int head_dim_idx) {
    return token_idx * (3 * hidden_size) + qkv_idx * hidden_size +
           head_idx * head_dim + head_dim_idx;
  };

  float scale = 1.0f / sqrtf((float)head_dim);

  float max = -1e9f;
  float exp_sum = 0.0f;
  extern __shared__ float out_val[];

  for (int i = 0; i < head_dim; ++i)
    out_val[i] = 0.0f;

  for (int t = 0; t <= token_idx; ++t) {
    /**
     * Calculate the attention score for a query, key pair by iterating across
     * the block (head_dim).
     */
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      int q_idx = calculate_qkv_offset(token_idx, 0, head_idx, i);
      int k_idx = calculate_qkv_offset(t, 1, head_idx, i);
      score += qkv[q_idx] * qkv[k_idx];
    }
    score *= scale;

    float max_prev = max;
    max = fmaxf(max, score);
    float correction = expf(max_prev - max);
    float exp_score = expf(score - max);
    exp_sum = exp_sum * correction + exp_score;

    for (int i = 0; i < head_dim; ++i) {
      int v_idx = calculate_qkv_offset(t, 2, head_idx, i);
      out_val[i] = out_val[i] * correction + exp_score * qkv[v_idx];
    }
  }

  for (int i = 0; i < head_dim; ++i) {
    int out_idx = token_idx * hidden_size + head_idx * head_dim + i;
    output[out_idx] = out_val[i] / exp_sum;
  }
}

static __global__ void fused_attention_kernel(float *output, const float *qkv,
                                              int hidden_size) {
  extern __shared__ float reduction[];

  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int head_dim = blockDim.x;
  int head_dim_idx = threadIdx.x;

  int num_warps = 1 + (head_dim - 1) / 32;
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  /**
   * Calculate the offset in the qkv tensor for a given token, query/key/value
   * type, head, and dimension.
   *
   * [ Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h ]
   *
   * `int qkv_idx`: 0 for query, 1 for key, 2 for value
   */
  auto calculate_qkv_offset = [&](int token_idx, int qkv_idx, int head_idx,
                                  int head_dim_idx) {
    return token_idx * (3 * hidden_size) + qkv_idx * hidden_size +
           head_idx * head_dim + head_dim_idx;
  };

  int q_idx = calculate_qkv_offset(token_idx, 0, head_idx, head_dim_idx);
  float q_val = qkv[q_idx];

  float scale = 1.0f / sqrtf((float)head_dim);

  /**
   * Log-sum-exp trick is applied for numerical stability, and online softmax
   * is used to avoid storing the entire softmax matrix in memory.
   *
   * Let's iterate over past tokens t <= token_idx (Causal Mask)
   */
  float max = -1e9f;    // maximum attention score
  float exp_sum = 0.0f; // sum of expf(score - max)
  float out_val = 0.0f; // sum of expf(score - max) * v_val

  for (int t = 0; t <= token_idx; ++t) {
    /**
     * Calculate the sum of 'product' across the block (head_dim). It conducts
     * reduction to obtain the attention score for a query, key pair. The code
     * follows the warp-level reduction pattern from layer normalization and
     * softmax kernels of `operations.cu`.
     */
    float t_score = 0.0f;
    if (head_dim_idx < head_dim) {
      int k_idx = calculate_qkv_offset(t, 1, head_idx, head_dim_idx);
      float k_val = qkv[k_idx];
      t_score = q_val * k_val;
    }

    float warp_score = t_score;
    for (int offset = 16; offset > 0; offset /= 2)
      warp_score += __shfl_down_sync(0xffffffff, warp_score, offset);
    if (lane_idx == 0)
      reduction[warp_idx] = warp_score;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int w = 1; w < num_warps; ++w) {
        reduction[0] += reduction[w];
      }
    }
    __syncthreads();

    float score = reduction[0] * scale;

    /**
     * Update max, exp_sum, and out_val for online softmax.
     * An update for exp_sum is also necessary to apply the new max.
     */
    float max_prev = max;
    max = fmaxf(max, score);
    float correction = expf(max_prev - max);
    float exp_score = expf(score - max);
    exp_sum = exp_sum * correction + exp_score;

    int v_idx = calculate_qkv_offset(t, 2, head_idx, head_dim_idx);
    float v_val = qkv[v_idx];
    out_val = out_val * correction + exp_score * v_val;
  }

  /**
   * Save the attention output to global memory.
   * The output layout is [seq_len, hidden_size] (heads concatenated).
   */
  int out_idx = token_idx * hidden_size + head_idx * head_dim + head_dim_idx;
  output[out_idx] = out_val / exp_sum;
}

/**
 * The kernel implements the flash attention mechanism. It is based on the
 * second version of the algorithm, presented by the paper "Flash Attention 2:
 * Fast causal attention with efficient memory usage" by A. R. et al.
 *
 * The key optimizations compared to Flash Attention 1 is provided as follows:
 *  - The position of query block loop and key/value block loop has been
 *    switched: outer loop iterates over query blocks while the inner loop
 *    streams key and value blocks. This allows the accumulators for online
 *    softmax to stay in registers, not in shared memory.
 *  - The blocks are launched to parallelize across the sequence length
 *    dimension, not just batch and head dimensions (version 1) to maximize the
 *    GPU occupancy.
 */

#define QUERY_BLOCK_SIZE 32
#define KEY_VALUE_BLOCK_SIZE 64
#define HEAD_DIM 64
#define NUM_WARPS HEAD_DIM / 32

static __global__ void flash_attention_warp_kernel(float *output,
                                                   const float *qkv,
                                                   int seq_len, int hidden_size,
                                                   int head_dim) {
  /**
   * blockDim.x == 32
   * blockDim.y == QUERY_BLOCK_SIZE
   */
  int head_idx = blockIdx.x;
  int query_block_idx = blockIdx.y;
  int row = query_block_idx * QUERY_BLOCK_SIZE + threadIdx.y;
  int tx = threadIdx.x;

  /**
   * Calculate the offset in the qkv tensor for a given token, query/key/value
   * type, head, and dimension.
   *
   * [ Q_1, Q_2, .., Q_h | K_1, K_2, .., K_h | V_1, V_2, .., V_h ]
   *
   * `int qkv_idx`: 0 for query, 1 for key, 2 for value
   */
  auto calculate_qkv_offset = [&](int token_idx, int qkv_idx, int head_idx,
                                  int dim_idx) {
    return token_idx * (3 * hidden_size) + qkv_idx * hidden_size +
           head_idx * head_dim + dim_idx;
  };

  /**
   * Shared Memory Tiles
   *
   * The size of shared memory should be known at compile time. Therefore,
   * to avoid dynamic shared memory allocation and keep the code simple,
   * `head_dim` was hardcoded to HEAD_DIM.
   *
   * The query is stored as registers for optimization.
   */
  __shared__ float key_block[KEY_VALUE_BLOCK_SIZE][HEAD_DIM];
  __shared__ float value_block[KEY_VALUE_BLOCK_SIZE][HEAD_DIM];

  /**
   * Each thread within a warp holds NUM_WARPS elements of the query and the
   * output value.
   */
  float query[NUM_WARPS];
  float out_val[NUM_WARPS];
  for (int i = 0; i < NUM_WARPS; ++i) {
    query[i] = 0.0f;
    out_val[i] = 0.0f;
  }

  /**
   * Log-sum-exp trick is applied for numerical stability, and online softmax
   * is used to avoid storing the entire softmax matrix in memory.
   */
  float max = -1e9f;
  float exp_sum = 0.0f;
  float scale = 1.0f / sqrtf((float)head_dim);

  /**
   * Load the query block scattered into registers. Each thread `tx` loads `tx`,
   * `tx + 32`, `tx + 64`, .. (NUM_WARPS) into registers.
   *
   * In GPT2, NUM_WARPS = HEAD_DIM / 32 = 64 / 32 = 2 elements are processed by
   * each thread.
   */
  if (row < seq_len) {
    for (int i = 0; i < NUM_WARPS; ++i) {
      int head_dim_idx = tx + i * 32;
      int q_idx = calculate_qkv_offset(row, 0, head_idx, head_dim_idx);
      query[i] = qkv[q_idx];
    }
  }

  /**
   * For each key-value block pairs, the queries will be fetched and processed.
   * The loops are designed to process the ith query with jth key and value.
   */
  int num_steps = 1 + (seq_len - 1) / KEY_VALUE_BLOCK_SIZE;

  for (int j = 0; j < num_steps; ++j) {
    /**
     * Load the jth key and value blocks into shared memory.
     *
     * It is a collaborative load with block stride loop. All threads iterate
     * over the tile indices until the entire tile is loaded.
     */
    int t = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;
    int kv_block_offset = j * KEY_VALUE_BLOCK_SIZE;

    int tile_size = KEY_VALUE_BLOCK_SIZE * HEAD_DIM;
    for (int i = t; i < tile_size; i += stride) {
      int r = i / HEAD_DIM;
      int c = i % HEAD_DIM;

      int kv_idx = kv_block_offset + r;
      if (kv_idx < seq_len) {
        int k_idx = calculate_qkv_offset(kv_idx, 1, head_idx, c);
        int v_idx = calculate_qkv_offset(kv_idx, 2, head_idx, c);
        key_block[r][c] = qkv[k_idx];
        value_block[r][c] = qkv[v_idx];
      } else {
        key_block[r][c] = 0.0f;
        value_block[r][c] = 0.0f;
      }
    }
    __syncthreads();

    /**
     * Compute the attention output for the ith query with jth key and value.
     */
    for (int k = 0; k < KEY_VALUE_BLOCK_SIZE; ++k) {
      int kv_idx = kv_block_offset + k;
      if (kv_idx > row) // causal mask
        continue;

      /**
       * Compute the attention score (dot product) with the query and key_block.
       * The query is distributed accross threads, with each thread possessing
       * NUM_WARPS query values.
       */
      float t_score = 0.0f;
      for (int i = 0; i < NUM_WARPS; ++i) {
        int head_dim_idx = tx + i * 32;
        t_score += query[i] * key_block[k][head_dim_idx];
      }

      float warp_score = t_score;
      for (int offset = 16; offset > 0; offset /= 2)
        warp_score += __shfl_xor_sync(0xffffffff, warp_score, offset);
      float score = warp_score * scale;

      /**
       * Update max, exp_sum, and out_val for online softmax.
       * An update for exp_sum is also necessary to apply the new max.
       */
      float max_prev = max;
      max = fmaxf(max, score);
      float correction = expf(max_prev - max);
      float exp_score = expf(score - max);
      exp_sum = exp_sum * correction + exp_score;

      for (int i = 0; i < NUM_WARPS; ++i) {
        int head_dim_idx = tx + i * 32;
        out_val[i] *= correction;
        out_val[i] += exp_score * value_block[k][head_dim_idx];
      }
    }
    __syncthreads();
  }

  /**
   * Save the attention output to global memory.
   * The output layout is [seq_len, hidden_size] (heads concatenated).
   */
  if (row < seq_len) {
    int out_offset = row * hidden_size + head_idx * head_dim;
    for (int i = 0; i < NUM_WARPS; ++i) {
      int dim_idx = tx + i * 32;
      output[out_offset + dim_idx] = out_val[i] / exp_sum;
    }
  }
}

/**
 * Extract the last token's hidden state
 */
static __global__ void extract_last_token_kernel(float *out, const float *in,
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

GPT2::GPT2(const std::string &path) { weights = utils::load_data(path, "gpu"); }

void GPT2::attention_block(Tensor &x, int layer_idx) {
  // std::cout << "[CUDA][TRACE] Attention Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  // Tensor x_norm = x;
  cudaMemcpy(x_norm.d_data, x.d_data, x.numel() * sizeof(float),
             cudaMemcpyDeviceToDevice);
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
  // Tensor qkv({seq_len, 3 * hidden_size});
  operations::matmul(qkv, x_norm, weights[prefix + "attn.c_attn.weight"]);
  operations::add_bias(qkv, weights[prefix + "attn.c_attn.bias"]);

  // Tensor attention_value({seq_len, hidden_size});

  /**
   * A naive implementation of multi-head attention. It allocates Q, K, V for
   * each head and computes attention scores, coefficients, and values for
   * each head with kernels from `operations.cu`.
   */
  // Tensor q_head({seq_len, head_dim});
  // Tensor k_head({seq_len, head_dim});
  // Tensor k_head_transposed({head_dim, seq_len});
  // Tensor v_head({seq_len, head_dim});
  // Tensor attention_head_scores({seq_len, seq_len});
  // Tensor attention_head_values({seq_len, head_dim});

  // dim3 block_dims(16, 16);
  // dim3 head_grid_dims(1 + (head_dim - 1) / 16, 1 + (seq_len - 1) / 16);
  // dim3 score_grid_dims(1 + (seq_len - 1) / 16, 1 + (seq_len - 1) / 16);

  // float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // for (int h = 0; h < num_heads; ++h) {
  //   /**
  //    * Extract Q, K, V heads from joint matrix `qkv` and compute attention
  //    * scores, coefficients, and values.
  //    */
  //   extract_head_kernel<<<head_grid_dims, block_dims>>>(
  //       q_head.d_data, qkv.d_data, seq_len, hidden_size, head_dim, h, 0);
  //   extract_head_kernel<<<head_grid_dims, block_dims>>>(
  //       k_head.d_data, qkv.d_data, seq_len, hidden_size, head_dim, h, 1);
  //   extract_head_kernel<<<head_grid_dims, block_dims>>>(
  //       v_head.d_data, qkv.d_data, seq_len, hidden_size, head_dim, h, 2);

  //   operations::transpose(k_head_transposed, k_head);
  //   operations::matmul(attention_head_scores, q_head,
  //                      k_head_transposed); // Score = Q * K^T
  //   causal_mask_and_scale_kernel<<<score_grid_dims, block_dims>>>(
  //       attention_head_scores.d_data, seq_len, scale);

  //   operations::softmax(attention_head_scores);
  //   operations::matmul(attention_head_values, attention_head_scores, v_head);

  //   /**
  //    * Save the results to the attention_value tensor.
  //    */
  //   insert_head_kernel<<<head_grid_dims, block_dims>>>(
  //       attention_value.d_data, attention_head_values.d_data, seq_len,
  //       hidden_size, head_dim, h);
  // }

  /**
   * Implementation of naive and optimized fused attention kernel.
   */
  // dim3 block_dims(1);
  // dim3 grid_dims(seq_len, num_heads);
  // naive_fused_attention_kernel<<<grid_dims, block_dims,
  //                                head_dim * sizeof(float)>>>(
  //     attention_value.d_data, qkv.d_data, hidden_size, head_dim);
  // dim3 block_dims(head_dim);
  // dim3 grid_dims(seq_len, num_heads);
  // fused_attention_kernel<<<grid_dims, block_dims>>>(attention_value.d_data,
  //                                                   qkv.d_data, hidden_size);

  /**
   * Implementation of the flash attention kernel.
   */
  dim3 block_dims(32, QUERY_BLOCK_SIZE);
  dim3 grid_dims(num_heads, 1 + (seq_len - 1) / QUERY_BLOCK_SIZE);
  flash_attention_warp_kernel<<<grid_dims, block_dims>>>(
      attention_value.d_data, qkv.d_data, seq_len, hidden_size, head_dim);

  // Tensor attention_output({seq_len, hidden_size});
  operations::matmul(attention_output, attention_value,
                     weights[prefix + "attn.c_proj.weight"]);
  operations::add_bias(attention_output, weights[prefix + "attn.c_proj.bias"]);

  operations::add(x, attention_output);
}

void GPT2::mlp_block(Tensor &x, int layer_idx) {
  // std::cout << "[CUDA][TRACE] MLP Layer " << layer_idx << std::endl;
  std::string prefix = "transformer.h." + std::to_string(layer_idx) + ".";
  int64_t seq_len = x.shape[0];

  // Tensor x_norm = x;
  cudaMemcpy(x_norm.d_data, x.d_data, x.numel() * sizeof(float),
             cudaMemcpyDeviceToDevice);
  operations::layer_norm(x_norm, weights[prefix + "ln_2.weight"],
                         weights[prefix + "ln_2.bias"]);

  // Tensor up({seq_len, mlp_size});
  operations::matmul(up, x_norm, weights[prefix + "mlp.c_fc.weight"]);
  operations::add_bias(up, weights[prefix + "mlp.c_fc.bias"]);

  operations::gelu(up);

  // Tensor down({seq_len, hidden_size});
  operations::matmul(down, up, weights[prefix + "mlp.c_proj.weight"]);
  operations::add_bias(down, weights[prefix + "mlp.c_proj.bias"]);

  operations::add(x, down);
}

Tensor GPT2::forward(int *input_ids, int seq_len) {
  // std::cout << "[CUDA][TRACE] Beginning forward pass" << std::endl;
  Tensor x({seq_len, hidden_size});
  operations::embed(x, weights["transformer.wte.weight"], input_ids, seq_len,
                    hidden_size);
  operations::positional_encoding(x, weights["transformer.wpe.weight"], seq_len,
                                  hidden_size);

  /**
   * Pre-allocate all temporary tensors.
   *
   * While PyTorch manages the memory overhead using a caching memory allocator,
   * this implementation avoids runtime allocation and deallocation during the
   * forward pass.
   */
  x_norm = Tensor({seq_len, hidden_size});
  qkv = Tensor({seq_len, 3 * hidden_size});
  attention_value = Tensor({seq_len, hidden_size});
  attention_output = Tensor({seq_len, hidden_size});
  up = Tensor({seq_len, mlp_size});
  down = Tensor({seq_len, hidden_size});

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
  operations::matmul(logits, prediction_token, weights["lm_head.weight"], true);

  return logits;
}
