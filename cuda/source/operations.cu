#include "operations.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace operations {

void transpose(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Transpose " << std::endl;
  int64_t rows = x.shape[0];
  int64_t cols = x.shape[1];

  Tensor out({cols, rows});
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(j, i) = x(i, j);
    }
  }

  x = std::move(out); // move assignment
}

void matmul(Tensor &out, const Tensor &A, const Tensor &B) {
  // std::cout << "[CUDA][TRACE] Matrix Multiplication" << std::endl;
  int64_t M = A.shape[0];
  int64_t K = A.shape[1];
  if (K != B.shape[0]) {
    std::cerr << "[CUDA][ERROR] Matrix dimensions " << A.shape[0] << " x "
              << A.shape[1] << " and " << B.shape[0] << " x " << B.shape[1]
              << " do not match" << std::endl;
    exit(1);
  }
  int64_t N = B.shape[1];

  if (out.data.empty() || out.shape[0] != M || out.shape[1] != N) {
    out.resize({M, N}); // allocate space
  }

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float val = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        val += A(m, k) * B(k, n);
      }
      out(m, n) = val;
    }
  }
}

void add(Tensor &x, const Tensor &y) {
  // std::cout << "[CUDA][TRACE] Add" << std::endl;
  for (int64_t i = 0; i < x.numel(); ++i) {
    x(i) += y(i);
  }
}

void add_bias(Tensor &x, const Tensor &bias) {
  // std::cout << "[CUDA][TRACE] Add Bias " << std::endl;
  int seq_len = x.shape[0];
  int hidden_size = x.shape[1];
  for (int64_t t = 0; t < seq_len; ++t) {
    for (int64_t i = 0; i < hidden_size; ++i) {
      x(t, i) += bias(i);
    }
  }
}

void gelu(Tensor &x) {
  // std::cout << "[CUDA][TRACE] GELU " << std::endl;
  const float c1 = 0.7978845608f; // sqrt(2 / pi)
  const float c2 = 0.044715f;
  for (float &val : x.data) {
    float cube = val * val * val;
    val = 0.5f * val * (1.0f + std::tanh(c1 * (val + c2 * cube)));
  }
}

void layer_norm(Tensor &x, const Tensor &weight, const Tensor &bias,
                float eps) {
  // std::cout << "[CUDA][TRACE] Layer Normalization " << std::endl;
  int64_t seq_len = x.shape[0];
  int64_t hidden_size = x.shape[1];

  for (int64_t t = 0; t < seq_len; ++t) {
    float mean = 0.0f;
    for (int64_t i = 0; i < hidden_size; ++i) {
      mean += x(t, i);
    }
    mean /= hidden_size;

    float variance = 0.0f;
    for (int64_t i = 0; i < hidden_size; ++i) {
      variance += (x(t, i) - mean) * (x(t, i) - mean);
    }
    variance /= hidden_size;
    float std = std::sqrt(variance + eps);

    for (int64_t i = 0; i < hidden_size; ++i) {
      float val = x(t, i);
      float norm_val = (val - mean) / std;
      x(t, i) = norm_val * weight(i) + bias(i);
    }
  }
}

/**
 * Apply softmax over the last dimension
 */
void softmax(Tensor &x) {
  // std::cout << "[CUDA][TRACE] Softmax " << std::endl;
  std::vector<int64_t> shape = x.shape;
  int64_t last_dim = shape.back();
  int64_t flattened_rows = x.numel() / last_dim;

  x.resize({flattened_rows, last_dim});

  for (int64_t r = 0; r < flattened_rows; ++r) {
    // max used for numerical stability
    float max_val = x(r, 0);
    for (int64_t i = 1; i < last_dim; ++i) {
      max_val = std::max(max_val, x(r, i));
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < last_dim; ++i) {
      x(r, i) = std::exp(x(r, i) - max_val);
      sum += x(r, i);
    }

    for (int64_t i = 0; i < last_dim; ++i) {
      x(r, i) /= sum;
    }
  }

  x.resize(shape);
}

} // namespace operations
