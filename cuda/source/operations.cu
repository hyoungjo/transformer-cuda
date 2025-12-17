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
 * Transform input ids to embeddings.
 * Each thread handles a hidden dimension of a token.
 *
 * token_id = input_ids[token]
 * x[token, i] = wte[token_id, i] + wpe[token, i]
 *
 * The kernel implements a strided loop to support arbitrary hidden size.
 */
__global__ void embed_kernel(float *x, const float *wte, const float *wpe,
                             const int *input_ids, int hidden_size) {
  int t = threadIdx.x;
  int token = blockIdx.x;

  for (int i = t; i < hidden_size; i += blockDim.x) {
    int token_id = input_ids[token];
    x[token * hidden_size + i] =
        wte[token_id * hidden_size + i] + wpe[token * hidden_size + i];
  }
}

/**
 * Transpose an input matrix and save to an output matrix.
 * Each thread handles an element.
 *
 * The kernel is designed to use a shared memory tile, to allow coalesced memory
 * access to the input and output matrices. To avoid shared memory bank
 * conflict, padding of 1 is added to the shared memory tile.
 */
__global__ void transpose_kernel(float *out, const float *in, int M, int N) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (col < N && row < M) {
    tile[ty][tx] = in[row * N + col];
  }
  __syncthreads();

  col = blockIdx.y * TILE_SIZE + threadIdx.x;
  row = blockIdx.x * TILE_SIZE + threadIdx.y;

  /**
   * `rows` is the number of columns in the output matrix and `cols` is the
   * number of rows in the output matrix.
   */
  if (col < M && row < N) {
    out[row * M + col] = tile[tx][ty]; // changed tx, ty position
  }
}

/**
 * Perform a general matrix multiplication of matrix A and B and save to out.
 * Each block handles a tile of the output matrix.
 *
 * The kernel implements a tiled matrix multiplication algorithm, using shared
 * memory to improve memory access patterns and reduce global memory access.
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
  for (int tile = 0; tile < num_tiles; ++tile) {
    /**
     * A collaborative load of tiles from A and B into the shared memory `As`
     * and `Bs`, respectively.
     *
     * Each thread loads `A[row][tile * TILE_SIZE + tx]` to `As` and `B[tile *
     * TILE_SIZE + ty][col]` to `Bs`.
     */
    int A_row = row;
    int A_col = tile * TILE_SIZE + tx;
    if (A_row < M && A_col < K) {
      As[ty][tx] = A[A_row * K + A_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    int B_row = tile * TILE_SIZE + ty;
    int B_col = col;
    if (B_row < K && B_col < N) {
      Bs[ty][tx] = B[B_row * N + B_col];
    } else {
      Bs[ty][tx] = 0.0f;
    }
    __syncthreads();

    /**
     * Compute the dot product of the loaded tiles. The position used from `As`
     * and `Bs` is different from what each thread loads.
     */
    for (int i = 0; i < TILE_SIZE; ++i) {
      val += As[ty][i] * Bs[i][tx];
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
  if (i < n)
    x[i] += y[i];
}

/**
 * Add bias to each element.
 * Each thread handles one element.
 */
__global__ void add_bias_kernel(float *x, const float *bias, int hidden_size,
                                int64_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int col = i % hidden_size;
  if (i < n)
    x[i] += bias[col];
}

/**
 * Apply GELU activation function to each element.
 * Each thread handles one element.
 *
 * The kernel uses `tanh` approximation, which was used for training GPT-2.
 * https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
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
 * Apply layer normalization to each sequence element, with weights and biases.
 * Each block processes a row (sequence element).
 *
 * The commented code is a typical reduction algorithm from the lecture slides,
 * implemented with sequential addressing (memory coalesced) and loop unrolling
 * (warp reduce).
 *
 * This kernel is implemented with the following optimizations:
 *  1. A strided loop to support arbitrary data length (over max blockDim.x,
 *     which is 1024). Each thread first aggregates elements t, t + 1024, ..
 *  2. Each warp is reduced to a single value and stored in shared memory.
 *  3. The thread 0 of each block aggregates the reduction result of each warp.
 */

__global__ void layer_norm_kernel(float *x, const float *weight,
                                  const float *bias, int hidden_size,
                                  float eps) {
  extern __shared__ float reduction[];

  int t = threadIdx.x;
  int row = blockIdx.x;
  float *data = x + row * hidden_size;

  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  /**
   * Find the mean of each row with strided loop and shared memory reduction.
   */

  // float t_sum = 0.0f;
  // for (int i = t; i < hidden_size; i += blockDim.x) {
  //   t_sum += data[i];
  // }
  // reduction[t] = t_sum;

  // for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
  //   __syncthreads();
  //   if (t < stride)
  //     reduction[t] += reduction[t + stride];
  // }
  // if (t < 32)
  //   reduction[t] = warp_reduce_sum(reduction[t]);
  // __syncthreads();

  float t_sum = 0.0f;
  for (int i = t; i < hidden_size; i += blockDim.x) {
    t_sum += data[i];
  }

  float warp_sum = warp_reduce_sum(t_sum);
  if (lane_idx == 0)
    reduction[warp_idx] = warp_sum;
  __syncthreads();
  if (t == 0) {
    for (int w = 1; w < blockDim.x / 32; ++w)
      reduction[0] += reduction[w];
  }
  __syncthreads();

  float mean = reduction[0] / hidden_size;

  /**
   * Compute the sum of squares (then variance) of the row with strided loop and
   * warp reduce.
   */

  // float t_sum_of_squares = 0.0f;
  // for (int i = t; i < hidden_size; i += blockDim.x) {
  //   t_sum_of_squares += std::pow(data[i] - mean, 2);
  // }
  // reduction[t] = t_sum_of_squares;

  // for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
  //   __syncthreads();
  //   if (t < stride)
  //     reduction[t] += reduction[t + stride];
  // }
  // if (t < 32)
  //   reduction[t] = warp_reduce_sum(reduction[t]);
  // __syncthreads();

  float t_sum_of_squares = 0.0f;
  for (int i = t; i < hidden_size; i += blockDim.x) {
    t_sum_of_squares += std::pow(data[i] - mean, 2);
  }

  float warp_sum_of_squares = warp_reduce_sum(t_sum_of_squares);
  if (lane_idx == 0)
    reduction[warp_idx] = warp_sum_of_squares;
  __syncthreads();

  if (t == 0) {
    for (int w = 1; w < blockDim.x / 32; ++w)
      reduction[0] += reduction[w];
  }
  __syncthreads();

  float variance = reduction[0] / hidden_size;
  float inv_std = rsqrtf(variance + eps);

  /**
   * Normalize and scale with weights and biases.
   */
  for (int i = t; i < hidden_size; i += blockDim.x) {
    data[i] = ((data[i] - mean) * inv_std) * weight[i] + bias[i];
  }
}

/**
 * Apply softmax over the last dimension.
 * Each block processes a flattened row. This is called in attention layers and
 * in the decoding process (calculate probabilities).
 *
 * The commented code is a typical reduction algorithm from the lecture slides,
 * implemented with sequential addressing (memory coalesced) and loop unrolling
 * (warp reduce).
 *
 * This kernel is implemented with the following optimizations:
 *  1. A strided loop to support arbitrary data length (over max blockDim.x,
 *     which is 1024). Each thread first aggregates elements t, t + 1024, ..
 *  2. Each warp is reduced to a single value and stored in shared memory.
 *  3. The thread 0 of each block aggregates the reduction result of each warp.
 */

__global__ void softmax_kernel(float *x, int dim_size, int rows) {
  extern __shared__ float reduction[];

  int t = threadIdx.x;
  int row = blockIdx.x;
  float *data = x + row * dim_size;

  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  /**
   * Find the maximum value in the row with strided loop and warp reduce.
   */

  // float t_max = -FLT_MAX;
  //   for (int i = t; i < dim_size; i += blockDim.x) {
  //     t_max = fmaxf(t_max, data[i]);
  //   }
  //   reduction[t] = t_max;

  //   for (unsigned int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
  //     __syncthreads();
  //     if (t < stride)
  //       reduction[t] = fmaxf(reduction[t], reduction[t + stride]);
  //   }
  //   if (t < 32)
  //     reduction[t] = warp_reduce_max(reduction[t]);
  //   __syncthreads();

  float t_max = -FLT_MAX;
  for (int i = t; i < dim_size; i += blockDim.x) {
    t_max = fmaxf(t_max, data[i]);
  }

  float warp_max = warp_reduce_max(t_max);
  if (lane_idx == 0)
    reduction[warp_idx] = warp_max;
  __syncthreads();
  if (t == 0) {
    for (int w = 1; w < blockDim.x / 32; ++w)
      reduction[0] = fmaxf(reduction[0], reduction[w]);
  }
  __syncthreads();

  float max = reduction[0];

  /**
   * Compute the exponential sum of the row with strided loop and warp reduce.
   */

  // float t_exp = 0.0f;
  //   for (int i = t; i < dim_size; i += blockDim.x) {
  //     t_exp += expf(data[i] - max);
  //   }
  //   reduction[t] = t_exp;

  //   for (unsigned int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
  //     __syncthreads();
  //     if (t < stride)
  //       reduction[t] += reduction[t + stride];
  //   }
  //   if (t < 32)
  //     reduction[t] = warp_reduce_sum(reduction[t]);
  //   __syncthreads();

  float t_exp_sum = 0.0f;
  for (int i = t; i < dim_size; i += blockDim.x) {
    t_exp_sum += expf(data[i] - max);
  }

  float warp_exp_sum = warp_reduce_sum(t_exp_sum);
  if (lane_idx == 0)
    reduction[warp_idx] = warp_exp_sum;
  __syncthreads();
  if (t == 0) {
    for (int w = 1; w < blockDim.x / 32; ++w)
      reduction[0] += reduction[w];
  }
  __syncthreads();

  float exp_sum = reduction[0];

  /**
   * Normalize each element with strided loop
   */
  for (int i = t; i < dim_size; i += blockDim.x) {
    data[i] = expf(data[i] - max) / exp_sum;
  }
}

/**
 * ============================================================
 * =================== Launch CUDA Kernels ====================
 * ============================================================
 */

void embed(Tensor &x, const Tensor &wte, const Tensor &wpe,
           const int *input_ids, int seq_len, int hidden_size) {
  // std::cout << "[CUDA][TRACE] Operations embed()" << std::endl;
  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(seq_len);
  embed_kernel<<<grid_dims, block_dims>>>(x.d_data, wte.d_data, wpe.d_data,
                                          input_ids, hidden_size);
}

void transpose(Tensor &out, const Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations transpose()" << std::endl;
  int64_t rows = x.shape[0];
  int64_t cols = x.shape[1];

  dim3 block_dims(TILE_SIZE, TILE_SIZE);
  dim3 grid_dims(1 + (cols - 1) / TILE_SIZE, 1 + (rows - 1) / TILE_SIZE);
  transpose_kernel<<<grid_dims, block_dims>>>(out.d_data, x.d_data, rows, cols);
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

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(1 + (n - 1) / block_size);
  add_kernel<<<grid_dims, block_dims>>>(x.d_data, y.d_data, n);
}

void add_bias(Tensor &x, const Tensor &bias) {
  // std::cout << "[CUDA][TRACE] Operations add_bias()" << std::endl;
  int64_t n = x.numel();
  int hidden_size = x.shape[1];

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(1 + (n - 1) / block_size);
  add_bias_kernel<<<grid_dims, block_dims>>>(x.d_data, bias.d_data, hidden_size,
                                             n);
}

void gelu(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations gelu()" << std::endl;
  int64_t n = x.numel();

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(1 + (n - 1) / block_size);
  gelu_kernel<<<grid_dims, block_dims>>>(x.d_data, n);
}

/**
 * The layer normalization and softmax kernels are designed so that each block
 * processes a row. For large language models, each row has the size of hidden
 * size or vocabulary size.
 *
 * To avoid block size (number of threads) limits, the kernel performs reduction
 * with strided loops. The block size is fixed to 1024 and the shared memory is
 * fixed to 1024 * sizeof(float), to fit in the limits of NVIDIA Ampere
 * architecture.
 *
 * e.g., GPT-2 has hidden size 768 and vocabulary size 50257. LLaMA 3 8B has
 * hidden size 4096 and vocabulary size 128k.
 */

void layer_norm(Tensor &x, const Tensor &weight, const Tensor &bias,
                float eps) {
  // std::cout << "[CUDA][TRACE] Operations layer_norm()" << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];

  dim3 block_dims(1024);
  dim3 grid_dims(seq_len);
  int num_warps = block_dims.x / 32;
  layer_norm_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
      x.d_data, weight.d_data, bias.d_data, hidden_size, eps);
}

void softmax(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations softmax()" << std::endl;
  int64_t dim_size = x.shape.back();       // last dimension
  int64_t num_rows = x.numel() / dim_size; // flattened rows

  dim3 block_dims(1024);
  dim3 grid_dims(num_rows);
  int num_warps = block_dims.x / 32;
  softmax_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
      x.d_data, dim_size, num_rows);
}

} // namespace operations
