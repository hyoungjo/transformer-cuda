#include "operations.cuh"
#include "tensor.cuh"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define TILE_SIZE 32

// Ampere Tensor Core Tile Configuration
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define SUB_TILE_SIZE 16 // multiple of WMMA_K

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
 * x[token, i] = embeddings[token_id, i]
 *
 * The kernel implements a strided loop to support arbitrary hidden size.
 */
__global__ void embed_kernel(float *x, const float *embeddings,
                             const int *input_ids, int hidden_size) {
  int t = threadIdx.x;
  int token = blockIdx.x;

  for (int i = t; i < hidden_size; i += blockDim.x) {
    int token_id = input_ids[token];
    x[token * hidden_size + i] = embeddings[token_id * hidden_size + i];
  }
}

__global__ void positional_encoding_kernel(float *x, const float *encodings,
                                           int hidden_size) {
  int t = threadIdx.x;
  int token = blockIdx.x;

  for (int i = t; i < hidden_size; i += blockDim.x) {
    x[token * hidden_size + i] += encodings[token * hidden_size + i];
  }
}

/**
 * Apply rotary positional embeddings (RoPE) to query and key vectors.
 * Each thread handles a pair of dimensions (i, i + head_dim / 2).
 *
 * theta_i = base ^ (-2 * (i % d) / d)
 *
 * Refer to the original paper for more details: Su et al., "RoFormer: Enhanced
 * transformer with rotary position embedding", arXiv preprint arXiv:2104.09864.
 */
__global__ void rope_kernel(float *x, int token_dim, int head_dim) {
  int t = threadIdx.x;
  int token = blockIdx.x;

  const float base = 500000.0f;

  int half_dim = head_dim / 2;
  for (int i = t; i < token_dim / 2; i += blockDim.x) {
    int head_idx = i / half_dim;
    int pair_idx = i % half_dim;

    float theta_i = 1.0f / powf(base, (float)(pair_idx * 2) / head_dim);
    float angle = token * theta_i;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int idx1 = head_idx * head_dim + pair_idx;
    int idx2 = head_idx * head_dim + pair_idx + half_dim;
    float x1 = x[token * token_dim + idx1];
    float x2 = x[token * token_dim + idx2];
    x[token * token_dim + idx1] = x1 * cos_val - x2 * sin_val;
    x[token * token_dim + idx2] = x1 * sin_val + x2 * cos_val;
  }
}

/**
 * Transpose an input matrix and save to an output matrix.
 * Each thread handles an element.
 *
 * In the naive version, the kernel directly reads and writes to the global
 * memory without considering the memory access pattern.
 *
 * The kernel is designed to use a shared memory tile, to allow coalesced memory
 * access to the input and output matrices. To avoid shared memory bank
 * conflict, padding of 1 is added to the shared memory tile.
 */

__global__ void naive_transpose_kernel(float *out, const float *in, int M,
                                       int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < N && y < M) {
    out[x * M + y] = in[y * N + x];
  }
}

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
 * In the naive version, the kernel is implemented without shared memory tiles,
 * which results in a high number of global memory accesses.
 *
 * The kernel implements a tiled matrix multiplication algorithm, using shared
 * memory to improve memory access patterns and reduce global memory access.
 */

__global__ void naive_matrix_multiplication_kernel(float *out, const float *A,
                                                   const float *B, int M, int N,
                                                   int K, bool transpose_b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      float b_val = transpose_b ? B[col * K + k] : B[k * N + col];
      sum += A[row * K + k] * b_val;
    }
    out[row * N + col] = sum;
  }
}

__global__ void matrix_multiplication_kernel(float *out, const float *A,
                                             const float *B, int M, int N,
                                             int K, bool transpose_b) {
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

    if (transpose_b) {
      /**
       * int B_row = bx * TILE_SIZE + tx;
       * int B_col = tile * TILE_SIZE + ty;
       *
       * To support coalesced global memory access, `tx` and `ty` are switched.
       * This is possible since the tiles have the same width and height.
       */
      int B_row = bx * TILE_SIZE + ty;
      int B_col = tile * TILE_SIZE + tx;
      if (B_row < N && B_col < K) {
        Bs[tx][ty] = B[B_row * K + B_col];
      } else {
        Bs[tx][ty] = 0.0f;
      }
    } else {
      int B_row = tile * TILE_SIZE + ty;
      int B_col = col;
      if (B_row < K && B_col < N) {
        Bs[ty][tx] = B[B_row * N + B_col];
      } else {
        Bs[ty][tx] = 0.0f;
      }
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
 * Perform matrix multiplication using Tensor Cores.
 *
 * A block consists of 128 threads (32 x 4 warps). Since Tensor Cores operate on
 * 16 x 16 matrices, the block's 32 x 32 tile is divided into four sections.
 */
__global__ void tensor_core_matrix_multiplication_kernel(float *out,
                                                         const float *A,
                                                         const float *B, int M,
                                                         int N, int K,
                                                         bool transpose_b) {
  int t = threadIdx.x;
  int block_row_offset = blockIdx.y * TILE_SIZE;
  int block_col_offset = blockIdx.x * TILE_SIZE;

  /**
   * Shared memory for the tiles. The tile is padded (+4) to avoid shared memory
   * bank conflicts. The padding of 4 matches the size of burst and wmma loads.
   */
  __shared__ float As[TILE_SIZE][SUB_TILE_SIZE + 4];
  __shared__ float Bs[SUB_TILE_SIZE][TILE_SIZE + 4];
  __shared__ float Cs[TILE_SIZE][TILE_SIZE + 4];

  /**
   * `acc` is a special hardware register that holds a 16 x 16 matrix of sums.
   */
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  /**
   * Let's iterate with the size of the sub-tile.
   *
   * Each loop processes 32 x 16 of As and 16 x 32 of Bs. This leaves two 16 x
   * 16 sub-tiles for As and Bs each. As a result, 2 x 2 = 4 sub-tile matrix
   * multiplications take place, handled by each warp.
   */
  int warp_idx = threadIdx.x / 32;
  int warp_row = (warp_idx / 2) * 16;
  int warp_col = (warp_idx % 2) * 16;

  for (int k = 0; k < K; k += SUB_TILE_SIZE) {
    /**
     * Load As and Bs tiles with 32 x 4 threads (128 threads).
     * Each tile has 32 x 16 = 512 elements, and therefore each thread
     * loads 512 / 128 = 4 elements.
     */
    int num_elements = TILE_SIZE * SUB_TILE_SIZE;
    int r, c, gr, gc;
    for (int i = t; i < num_elements; i += blockDim.x) {
      r = i / SUB_TILE_SIZE;
      c = i % SUB_TILE_SIZE;
      gr = block_row_offset + r;
      gc = k + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
      if (transpose_b) {
        /**
         * To fill Bs with SUB_TILE_SIZE x TILE_SIZE, the tile loaded from
         * global memory matrix B will be TILE_SIZE x SUB_TILE_SIZE.
         *
         * To support coalesced global memory access, the threads are allocated
         * column-wise for shared memory and row-wise for global memory.
         */
        c = i / SUB_TILE_SIZE;
        r = i % SUB_TILE_SIZE;
        gr = block_col_offset + c;
        gc = k + r;
        Bs[r][c] = (gr < N && gc < K) ? B[gr * K + gc] : 0.0f;
      } else {
        r = i / TILE_SIZE;
        c = i % TILE_SIZE;
        gr = k + r;
        gc = block_col_offset + c;
        Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
      }
    }
    __syncthreads();

    /**
     * Tensor core matrix multiplication. Since Tensor Cores consume WMMA_K (8)
     * at a time, this iterates over the SUB_TILE_SIZE (16).
     */
    for (int i = 0; i < SUB_TILE_SIZE; i += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          a_fragment;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          b_fragment;
      wmma::load_matrix_sync(a_fragment, &As[warp_row][i], SUB_TILE_SIZE + 4);
      wmma::load_matrix_sync(b_fragment, &Bs[i][warp_col], TILE_SIZE + 4);
      wmma::mma_sync(acc, a_fragment, b_fragment, acc);
    }
    __syncthreads();
  }

  /**
   * Each warp writes the result (16 x 16) into `Cs`.
   */
  wmma::store_matrix_sync(&Cs[warp_row][warp_col], acc, TILE_SIZE + 4,
                          wmma::mem_row_major);
  __syncthreads();

  /**
   * Write `Cs` to the global memory `out`.
   */
  int num_elements = TILE_SIZE * TILE_SIZE;
  for (int i = t; i < num_elements; i += blockDim.x) {
    int r = i / TILE_SIZE;
    int c = i % TILE_SIZE;
    int gr = block_row_offset + r;
    int gc = block_col_offset + c;
    if (gr < M && gc < N) {
      out[gr * N + gc] = Cs[r][c];
    }
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
 * Multiply y to x element-wise.
 * Each thread handles one element.
 */
__global__ void multiply_kernel(float *x, const float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= y[i];
  }
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
 * Apply SiLU (Swish) activation function to each element.
 * Each thread handles one element.
 *
 * x / (1 + exp(-x))
 */
__global__ void silu_kernel(float *x, int64_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    x[i] = val / (1.0f + expf(-val));
  }
}

/**
 * Apply layer normalization to each sequence element, with weights and biases.
 * Each block processes a row (sequence element).
 *
 * In the naive version, which is implemented without any shared memory use,
 * each block is launched with a single thread, which handles the entire row
 * sequentially.
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

__global__ void naive_layer_norm_kernel(float *x, const float *weight,
                                        const float *bias, int hidden_size,
                                        float eps) {
  int row = blockIdx.x;
  float *data = x + row * hidden_size;

  float sum = 0.0f;
  for (int i = 0; i < hidden_size; ++i)
    sum += data[i];
  float mean = sum / hidden_size;

  float sum_of_squares = 0.0f;
  for (int i = 0; i < hidden_size; ++i) {
    float diff = data[i] - mean;
    sum_of_squares += diff * diff;
  }
  float variance = sum_of_squares / hidden_size;
  float inv_std = rsqrtf(variance + eps);

  for (int i = 0; i < hidden_size; ++i) {
    data[i] = ((data[i] - mean) * inv_std) * weight[i] + bias[i];
  }
}

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
  //   float diff = data[i] - mean;
  //   t_sum_of_squares += diff * diff;
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
    float diff = data[i] - mean;
    t_sum_of_squares += diff * diff;
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
 * Apply root-mean-square (RMS) normalization to each sequence element.
 * Each block processes a row (sequence element).
 *
 * In the naive version, which is implemented without any shared memory use,
 * each block is launched with a single thread, which handles the entire row
 * sequentially.
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

__global__ void naive_rms_norm_kernel(float *x, const float *weight,
                                      int hidden_size, float eps) {
  int row = blockIdx.x;
  float *data = x + row * hidden_size;

  float sum_of_squares = 0.0f;
  for (int i = 0; i < hidden_size; ++i)
    sum_of_squares += data[i] * data[i];
  float inv_rms = rsqrtf(sum_of_squares / hidden_size + eps);

  for (int i = 0; i < hidden_size; ++i) {
    data[i] = data[i] * inv_rms * weight[i];
  }
}

__global__ void rms_norm_kernel(float *x, const float *weight, int hidden_size,
                                float eps) {
  extern __shared__ float reduction[];

  int t = threadIdx.x;
  int row = blockIdx.x;
  float *data = x + row * hidden_size;

  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  /**
   * Compute the sum of squares (then variance) of the row with strided loop and
   * warp reduce.
   */
  // float t_sum_of_squares = 0.0f;
  // for (int i = t; i < hidden_size; i += blockDim.x) {
  //   t_sum_of_squares += data[i] * data[i];
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
    t_sum_of_squares += data[i] * data[i];
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

  float inv_rms = rsqrtf(reduction[0] / hidden_size + eps);

  for (int i = t; i < hidden_size; i += blockDim.x) {
    data[i] = data[i] * inv_rms * weight[i];
  }
}

/**
 * Apply softmax over the last dimension.
 * Each block processes a flattened row. This is called in attention layers and
 * in the decoding process (calculate probabilities).
 *
 * In the naive version, which is implemented without any shared memory use,
 * each block is launched with a single thread, which handles the entire row
 * sequentially.
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

__global__ void naive_softmax_kernel(float *x, int stride, int rows) {
  int row = blockIdx.x;
  float *data = x + row * stride;

  float max = data[0];
  for (int i = 1; i < stride; ++i) {
    max = fmaxf(max, data[i]);
  }

  float exp_sum = 0.0f;
  for (int i = 0; i < stride; ++i) {
    data[i] = expf(data[i] - max);
    exp_sum += data[i];
  }

  for (int i = 0; i < stride; ++i) {
    data[i] /= exp_sum;
  }
}

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

void embed(Tensor &x, const Tensor &embeddings, const int *input_ids) {
  // std::cout << "[CUDA][TRACE] Operations embed()" << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(seq_len);
  embed_kernel<<<grid_dims, block_dims>>>(x.d_data, embeddings.d_data,
                                          input_ids, hidden_size);
}

void positional_encoding(Tensor &x, const Tensor &encodings) {
  // std::cout << "[CUDA][TRACE] Operations positional_encoding()" << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(seq_len);
  positional_encoding_kernel<<<grid_dims, block_dims>>>(
      x.d_data, encodings.d_data, hidden_size);
}

void rope(Tensor &x, int head_dim) {
  // std::cout << "[CUDA][TRACE] Operations rope()" << std::endl;
  int seq_len = x.shape[0];
  int token_dim = x.shape[1];

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(seq_len);
  rope_kernel<<<grid_dims, block_dims>>>(x.d_data, token_dim, head_dim);
}

void transpose(Tensor &out, const Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations transpose()" << std::endl;
  int64_t rows = x.shape[0];
  int64_t cols = x.shape[1];

  dim3 block_dims(TILE_SIZE, TILE_SIZE);
  dim3 grid_dims(1 + (cols - 1) / TILE_SIZE, 1 + (rows - 1) / TILE_SIZE);
  naive_transpose_kernel<<<grid_dims, block_dims>>>(out.d_data, x.d_data, rows,
                                                    cols);
  // transpose_kernel<<<grid_dims, block_dims>>>(out.d_data, x.d_data, rows,
  // cols);
}

void matmul(Tensor &out, const Tensor &A, const Tensor &B, bool transpose_b) {
  // std::cout << "[CUDA][TRACE] Operations matmul()" << std::endl;
  int64_t M = A.shape[0];
  int64_t K = A.shape[1];
  int64_t N = transpose_b ? B.shape[0] : B.shape[1];

  // dim3 block_dims(TILE_SIZE, TILE_SIZE);
  // dim3 grid_dims(1 + (N - 1) / TILE_SIZE, 1 + (M - 1) / TILE_SIZE);
  // naive_matrix_multiplication_kernel<<<grid_dims, block_dims>>>(
  //     out.d_data, A.d_data, B.d_data, M, N, K, transpose_b);
  // matrix_multiplication_kernel<<<grid_dims, block_dims>>>(
  //     out.d_data, A.d_data, B.d_data, M, N, K, transpose_b);

  dim3 block_dims(128); // 32 x 4 warps
  dim3 grid_dims(1 + (N - 1) / TILE_SIZE, 1 + (M - 1) / TILE_SIZE);
  tensor_core_matrix_multiplication_kernel<<<grid_dims, block_dims>>>(
      out.d_data, A.d_data, B.d_data, M, N, K, transpose_b);
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

void multiply(Tensor &x, const Tensor &y) {
  // std::cout << "[CUDA][TRACE] Operations multiply()" << std::endl;
  int n = x.numel();

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(1 + (n - 1) / block_size);
  multiply_kernel<<<grid_dims, block_dims>>>(x.d_data, y.d_data, n);
}

void gelu(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations gelu()" << std::endl;
  int64_t n = x.numel();

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(1 + (n - 1) / block_size);
  gelu_kernel<<<grid_dims, block_dims>>>(x.d_data, n);
}

void silu(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations silu()" << std::endl;
  int64_t n = x.numel();

  int block_size = 1024;
  dim3 block_dims(block_size);
  dim3 grid_dims(1 + (n - 1) / block_size);
  silu_kernel<<<grid_dims, block_dims>>>(x.d_data, n);
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

  // dim3 block_dims(1);
  // dim3 grid_dims(seq_len);
  // naive_layer_norm_kernel<<<grid_dims, block_dims>>>(
  //     x.d_data, weight.d_data, bias.d_data, hidden_size, eps);

  dim3 block_dims(256);
  dim3 grid_dims(seq_len);
  int num_warps = block_dims.x / 32;
  layer_norm_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
      x.d_data, weight.d_data, bias.d_data, hidden_size, eps);
}

void rms_norm(Tensor &x, const Tensor &weight, float eps) {
  // std::cout << "[CUDA][TRACE] Operations rms_norm()" << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];

  dim3 block_dims(1);
  dim3 grid_dims(seq_len);
  naive_rms_norm_kernel<<<grid_dims, block_dims>>>(x.d_data, weight.d_data,
                                                   hidden_size, eps);

  // dim3 block_dims(256);
  // dim3 grid_dims(seq_len);
  // int num_warps = block_dims.x / 32;
  // rms_norm_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
  //     x.d_data, weight.d_data, hidden_size, eps);
}

void softmax(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations softmax()" << std::endl;
  int64_t dim_size = x.shape.back();       // last dimension
  int64_t num_rows = x.numel() / dim_size; // flattened rows

  dim3 block_dims(1);
  dim3 grid_dims(num_rows);
  naive_softmax_kernel<<<grid_dims, block_dims>>>(x.d_data, dim_size, num_rows);

  // dim3 block_dims(256);
  // dim3 grid_dims(num_rows);
  // int num_warps = block_dims.x / 32;
  // softmax_kernel<<<grid_dims, block_dims, num_warps * sizeof(float)>>>(
  //     x.d_data, dim_size, num_rows);
}

} // namespace operations
