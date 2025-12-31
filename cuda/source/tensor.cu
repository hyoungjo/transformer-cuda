#include "tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

/**
 * ============================================================
 * ==== Constructors, Assignment Operators, and Destructor ====
 * ============================================================
 */

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<int64_t> &shape, std::string device,
               bool zeros) {
  // std::cout << "[CUDA][TRACE] Tensor Constructor" << std::endl;
  this->shape = shape;
  if (device == "gpu") {
    cudaMalloc(&d_data, numel() * sizeof(float));
    if (zeros) {
      cudaMemset(d_data, 0, numel() * sizeof(float));
    }
  } else if (device == "cpu") {
    h_data.resize(numel());
    if (zeros) {
      std::fill(h_data.begin(), h_data.end(), 0);
    }
  }
}

Tensor::Tensor(const Tensor &other) {
  // std::cout << "[CUDA][TRACE] Tensor Copy Constructor" << std::endl;
  shape = other.shape;
  h_data = other.h_data;
  if (other.d_data) {
    cudaMalloc(&d_data, other.numel() * sizeof(float));
    cudaMemcpy(d_data, other.d_data, other.numel() * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
}

Tensor &Tensor::operator=(const Tensor &other) {
  // std::cout << "[CUDA][TRACE] Tensor Copy Assignment" << std::endl;
  if (this != &other) {
    shape = other.shape;
    h_data = other.h_data;
    if (d_data)
      cudaFree(d_data);
    if (other.d_data) {
      cudaMalloc(&d_data, other.numel() * sizeof(float));
      cudaMemcpy(d_data, other.d_data, other.numel() * sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }
  }
  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape(std::move(other.shape)), h_data(std::move(other.h_data)),
      d_data(other.d_data) {
  // std::cout << "[CUDA][TRACE] Tensor Move Constructor" << std::endl;
  other.d_data = nullptr;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  // std::cout << "[CUDA][TRACE] Tensor Move Assignment" << std::endl;
  if (this != &other) {
    shape = std::move(other.shape);
    h_data = std::move(other.h_data);
    if (d_data)
      cudaFree(d_data);
    d_data = other.d_data;
    other.d_data = nullptr;
  }
  return *this;
}

Tensor::~Tensor() {
  // std::cout << "[CUDA][TRACE] Tensor Destructor" << std::endl;
  if (d_data) {
    cudaFree(d_data);
  }
}

/**
 * ============================================================
 * ===================== Member Functions =====================
 * ============================================================
 */

int64_t Tensor::numel() const {
  // std::cout << "[CUDA][TRACE] Tensor numel()" << std::endl;
  if (shape.empty())
    return 0;
  return std::accumulate(shape.begin(), shape.end(), 1LL,
                         std::multiplies<int64_t>());
}

void Tensor::to(std::string device) {
  // std::cout << "[CUDA][TRACE] Tensor to()" << std::endl;
  if (device == "gpu") {
    if (d_data) {
      std::cerr << "Tensor is already on GPU" << std::endl;
      return;
    }
    size_t bytes = numel() * sizeof(float);
    cudaMalloc((void **)&d_data, bytes);
    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);
    h_data.clear();
  } else if (device == "cpu") {
    if (!h_data.empty()) {
      std::cerr << "Tensor is already on CPU" << std::endl;
      return;
    }
    int64_t num_elements = numel();
    h_data.resize(num_elements);
    cudaMemcpy(h_data.data(), d_data, num_elements * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    d_data = nullptr;
  }
}

/**
 * ============================================================
 * ================ Host-side Helper Functions ================
 * ============================================================
 */

std::vector<float>::iterator Tensor::begin() { return h_data.begin(); }
std::vector<float>::iterator Tensor::end() { return h_data.end(); }
std::vector<float>::const_iterator Tensor::begin() const {
  return h_data.begin();
}
std::vector<float>::const_iterator Tensor::end() const { return h_data.end(); }

float &Tensor::operator()(int64_t i) { return h_data[i]; }
const float &Tensor::operator()(int64_t i) const { return h_data[i]; }

float &Tensor::operator()(int64_t i, int64_t j) {
  return h_data[i * shape[1] + j];
}
const float &Tensor::operator()(int64_t i, int64_t j) const {
  return h_data[i * shape[1] + j];
}

float &Tensor::operator()(int64_t i, int64_t j, int64_t k) {
  return h_data[i * (shape[1] * shape[2]) + j * shape[2] + k];
}
const float &Tensor::operator()(int64_t i, int64_t j, int64_t k) const {
  return h_data[i * (shape[1] * shape[2]) + j * shape[2] + k];
}
