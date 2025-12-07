#include "operations.hpp"
#include "tensor.hpp"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace operations {

/**
 * ============================================================
 * =================== Helper CUDA Kernels ====================
 * ============================================================
 */

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
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (m < M && n < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[m * K + k] * B[k * N + n];
    }
    out[m * N + n] = sum;
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
 * This is a naive version where only thread 0 does the math.
 */
__global__ void layer_norm_kernel(float *x, const float *weight,
                                  const float *bias, int hidden_size,
                                  float eps) {
  int t = blockIdx.x;

  if (threadIdx.x == 0) {
    float *row = x + t * hidden_size;

    float sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i)
      sum += row[i];
    float mean = sum / hidden_size;

    float sum_of_squares = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
      float diff = row[i] - mean;
      sum_of_squares += diff * diff;
    }
    float variance = sum_of_squares / hidden_size;
    float std = sqrtf(variance + eps);

    for (int i = 0; i < hidden_size; ++i) {
      row[i] = ((row[i] - mean) / std) * weight[i] + bias[i];
    }
  }
}

/**
 * Apply softmax over the last dimension.
 * Each block processes a flattened row.
 * This is a naive version where only thread 0 does the math.
 */
__global__ void softmax_kernel(float *x, int stride, int rows) {
  int r = blockIdx.x;
  if (r < rows) {
    float *row = x + r * stride;

    if (threadIdx.x == 0) { // only thread 0 gets involved
      float max_val = row[0];
      for (int i = 1; i < stride; ++i) {
        max_val = fmaxf(max_val, row[i]);
      }

      float sum = 0.0f;
      for (int i = 0; i < stride; ++i) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
      }

      for (int i = 0; i < stride; ++i) {
        row[i] /= sum;
      }
    }
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

  dim3 block_dims(16, 16);
  dim3 grid_dims(1 + (N - 1) / 16, 1 + (M - 1) / 16);
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

  dim3 block_dims(1);
  dim3 grid_dims(seq_len);
  layer_norm_kernel<<<grid_dims, block_dims>>>(x.d_data, weight.d_data,
                                               bias.d_data, hidden_size, eps);
}

void softmax(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Operations softmax()" << std::endl;
  int64_t last_dim = x.shape.back();
  int64_t flattened_rows = x.numel() / last_dim;

  dim3 block_dims(1);
  dim3 grid_dims(flattened_rows);
  softmax_kernel<<<grid_dims, block_dims>>>(x.d_data, last_dim, flattened_rows);
}

} // namespace operations
