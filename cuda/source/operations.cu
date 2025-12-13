#include "operations.hpp"
#include "tensor.hpp"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define TILE_SIZE 16

namespace operations {

/**
 * ============================================================
 * =================== Helper CUDA Kernels ====================
 * ============================================================
 */

__inline__ __device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/**
 * Look up and configure embeddings.
 * Each thread handles a hidden dimension of a token.
 *
 * token_id = input_ids[t]
 * x[t, i] = wte[token_id, i] + wpe[t, i]
 */
__global__ void embedding_kernel(float *x, const float *wte, const float *wpe,
                                 const int *input_ids, int seq_len,
                                 int hidden_size) {
  int t = blockIdx.x;
  int i = threadIdx.x;

  if (t < seq_len && i < hidden_size) {
    int token_id = input_ids[t];
    x[t * hidden_size + i] =
        wte[token_id * hidden_size + i] + wpe[t * hidden_size + i];
  }
}

/**
 * Transpose a matrix.
 * Each thread handles an element.
 */
__global__ void transpose_kernel(float *out, const float *in, int rows,
                                 int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows) {
    out[x * rows + y] = in[y * cols + x];
  }
}

/**
 * Matrix multiplication.
 * Each thread handles an output element.
 */
__global__ void matrix_multiplication_kernel(float *out, const float *A,
                                             const float *B, int M, int N,
                                             int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Global row and column indices for this thread
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  /**
   * The loop iterates pairs of tiles from A and B to compute the result of a
   * specific tile in `out`.
   *
   * `val` holds the dot product result to be saved to `out[row][col]`.
   * `num_tiles` is the number of tile pairs to iterate over.
   */
  float val = 0.0f;
  int num_tiles = 1 + (K - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    /**
     * A collaborative load of tiles from A and B into the shared memory `As`
     * and `Bs`, respectively.
     *
     * The current thread will load `A[row][t * TILE_SIZE + tx]` to `As` and
     * `B[t * TILE_SIZE + ty][col]` to `Bs`.
     */
    int a_col = t * TILE_SIZE + tx;
    if (row < M && a_col < K) {
      As[ty][tx] = A[row * K + a_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    int b_row = t * TILE_SIZE + ty;
    if (b_row < K && col < N) {
      Bs[ty][tx] = B[b_row * N + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }
    __syncthreads();

    /**
     * Compute the dot product of the loaded tiles. The position used from `As`
     * and `Bs` is different from what each thread loads.
     */
    for (int k = 0; k < TILE_SIZE; ++k) {
      val += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    out[row * N + col] = val;
  }
}

/**
 * Add y to x element-wise.
 * Each thread handles one element.
 */
__global__ void add_kernel(float *x, const float *y, int64_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += y[i];
  }
}

/**
 * Add bias to each element.
 * Each thread handles one element.
 */
__global__ void add_bias_kernel(float *x, const float *bias, int hidden_size,
                                int64_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int col = i % hidden_size;
    x[i] += bias[col];
  }
}

/**
 * Apply GELU activation function to each element.
 * Each thread handles one element.
 */
__global__ void gelu_kernel(float *x, int64_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    const float c1 = 0.7978845608f;
    const float c2 = 0.044715f;
    float cube = val * val * val;
    x[i] = 0.5f * val * (1.0f + tanhf(c1 * (val + c2 * cube)));
  }
}

/**
 * Apply layer normalization over each sequence element.
 * Each block processes a row (sequence element).
 */
__global__ void layer_norm_kernel(float *x, const float *weight,
                                  const float *bias, int hidden_size,
                                  float eps) {
  extern __shared__ float warp_results[];

  int row_idx = blockIdx.x;
  float *row = x + row_idx * hidden_size;

  int num_warps = blockDim.x / 32;
  int tid = threadIdx.x;
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  /**
   * Find the sum (then mean) of the row with strided loop and warp reduce.
   *
   * The first thread of each warp stores the result in the shared memory, and
   * later aggregated by thread 0.
   */
  float local_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    local_sum += row[i];
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane_idx == 0) {
    warp_results[warp_idx] = local_sum;
  }
  __syncthreads();

  if (tid == 0) {
    for (int w = 1; w < num_warps; ++w) {
      warp_results[0] += warp_results[w];
    }
  }
  __syncthreads();

  float mean = warp_results[0] / hidden_size;

  /**
   * Compute the sum of squares (then variance) of the row with strided loop and
   * warp reduce.
   */
  float local_sum_of_squares = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    local_sum_of_squares += (row[i] - mean) * (row[i] - mean);
  }
  local_sum_of_squares = warp_reduce_sum(local_sum_of_squares);
  if (lane_idx == 0) {
    warp_results[warp_idx] = local_sum_of_squares;
  }
  __syncthreads();

  if (tid == 0) {
    for (int w = 1; w < num_warps; ++w) {
      warp_results[0] += warp_results[w];
    }
  }
  __syncthreads();

  float variance = warp_results[0] / hidden_size;
  float inv_std = rsqrtf(variance + eps);

  /**
   * Normalize and scale with weights and biases.
   */
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    row[i] = ((row[i] - mean) * inv_std) * weight[i] + bias[i];
  }
}

/**
 * Apply softmax over the last dimension.
 * Each block processes a flattened row.
 */
__global__ void softmax_kernel(float *x, int stride, int rows) {
  extern __shared__ float warp_results[];

  int row_idx = blockIdx.x;
  if (row_idx >= rows)
    return;

  float *row = x + row_idx * stride;

  int num_warps = blockDim.x / 32;
  int tid = threadIdx.x;
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  /**
   * Find the maximum value in the row with strided loop and warp reduce.
   *
   * The first thread of each warp stores the result in the shared memory, and
   * later aggregated by thread 0.
   */
  float local_max = -FLT_MAX;
  for (int i = tid; i < stride; i += blockDim.x) {
    local_max = fmaxf(local_max, row[i]);
  }
  local_max = warp_reduce_max(local_max);
  if (lane_idx == 0) {
    warp_results[warp_idx] = local_max;
  }
  __syncthreads();

  if (tid == 0) {
    for (int w = 1; w < num_warps; ++w) {
      warp_results[0] = fmaxf(warp_results[0], warp_results[w]);
    }
  }
  __syncthreads();

  float row_max = warp_results[0];

  /**
   * Compute the exponential sum of the row with strided loop and warp reduce.
   */
  float local_sum = 0.0f;
  for (int i = tid; i < stride; i += blockDim.x) {
    local_sum += expf(row[i] - row_max);
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane_idx == 0) {
    warp_results[warp_idx] = local_sum;
  }
  __syncthreads();

  if (tid == 0) {
    for (int w = 1; w < num_warps; ++w) {
      warp_results[0] += warp_results[w];
    }
  }
  __syncthreads();

  float row_sum = warp_results[0];

  /**
   * Normalize each element with strided loop
   */
  for (int i = tid; i < stride; i += blockDim.x) {
    row[i] = expf(row[i] - row_max) / row_sum;
  }
}

/**
 * ============================================================
 * =================== Launch CUDA Kernels ====================
 * ============================================================
 */

void embedding(Tensor &x, const Tensor &wte, const Tensor &wpe,
               const std::vector<int> &input_ids) {
  // std::cout << "[CUDA][TRACE] Operations embedding()" << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];

  int *d_input_ids;
  cudaMalloc(&d_input_ids, seq_len * sizeof(int));
  cudaMemcpy(d_input_ids, input_ids.data(), seq_len * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 block_dims(hidden_size);
  dim3 grid_dims(seq_len);
  embedding_kernel<<<grid_dims, block_dims>>>(
      x.d_data, wte.d_data, wpe.d_data, d_input_ids, seq_len, hidden_size);
  cudaFree(d_input_ids);
}

void transpose(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations transpose()" << std::endl;
  int64_t rows = x.shape[0];
  int64_t cols = x.shape[1];
  Tensor out({cols, rows});

  dim3 block_dims(16, 16);
  dim3 grid_dims(1 + (cols - 1) / 16, 1 + (rows - 1) / 16);
  transpose_kernel<<<grid_dims, block_dims>>>(out.d_data, x.d_data, rows, cols);

  x = std::move(out);
}

void matmul(Tensor &out, const Tensor &A, const Tensor &B) {
  // std::cout << "[CUDA][TRACE] Operations matmul()" << std::endl;
  int64_t M = A.shape[0];
  int64_t K = A.shape[1];
  int64_t N = B.shape[1];

  dim3 block_dims(TILE_SIZE, TILE_SIZE);
  dim3 grid_dims(1 + (N - 1) / TILE_SIZE, 1 + (M - 1) / TILE_SIZE);
  matrix_multiplication_kernel<<<grid_dims, block_dims>>>(out.d_data, A.d_data,
                                                          B.d_data, M, N, K);
}

void add(Tensor &x, const Tensor &y) {
  // std::cout << "[CUDA][TRACE] Operations add()" << std::endl;
  int64_t n = x.numel();

  dim3 block_dims(256);
  dim3 grid_dims(1 + (n - 1) / 256);
  add_kernel<<<grid_dims, block_dims>>>(x.d_data, y.d_data, n);
}

void add_bias(Tensor &x, const Tensor &bias) {
  // std::cout << "[CUDA][TRACE] Operations add_bias()" << std::endl;
  int64_t n = x.numel();
  int hidden_size = x.shape[1];

  dim3 block_dims(256);
  dim3 grid_dims(1 + (n - 1) / 256);
  add_bias_kernel<<<grid_dims, block_dims>>>(x.d_data, bias.d_data, hidden_size,
                                             n);
}

void gelu(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations gelu()" << std::endl;
  int64_t n = x.numel();

  dim3 block_dims(256);
  dim3 grid_dims(1 + (n - 1) / 256);
  gelu_kernel<<<grid_dims, block_dims>>>(x.d_data, n);
}

void layer_norm(Tensor &x, const Tensor &weight, const Tensor &bias,
                float eps) {
  // std::cout << "[CUDA][TRACE] Operations layer_norm()" << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];

  dim3 block_dims(256);
  dim3 grid_dims(seq_len);
  int num_warps = block_dims.x / 32; // should be a multiple of 32
  layer_norm_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
      x.d_data, weight.d_data, bias.d_data, hidden_size, eps);
}

void softmax(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations softmax()" << std::endl;
  int64_t last_dim = x.shape.back();
  int64_t flattened_rows = x.numel() / last_dim;

  dim3 block_dims(256);
  dim3 grid_dims(flattened_rows);
  int num_warps = block_dims.x / 32; // should be a multiple of 32
  softmax_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
      x.d_data, last_dim, flattened_rows);
}

} // namespace operations
