#include "tensor.hpp"
#include <numeric>

Tensor::Tensor() {}
Tensor::Tensor(const std::vector<int64_t> &shape) { resize(shape); }

int64_t Tensor::numel() const {
  if (shape.empty())
    return 0;
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<int64_t>());
}

void Tensor::resize(const std::vector<int64_t> &shape) {
  this->shape = shape;
  data.resize(numel());
}

/**
 * Iterators
 */
std::vector<float>::iterator Tensor::begin() { return data.begin(); }
std::vector<float>::iterator Tensor::end() { return data.end(); }
std::vector<float>::const_iterator Tensor::begin() const {
  return data.begin();
}
std::vector<float>::const_iterator Tensor::end() const { return data.end(); }

/**
 * Element Access
 */
float &Tensor::operator()(int64_t i) { return data[i]; }
const float &Tensor::operator()(int64_t i) const { return data[i]; }

float &Tensor::operator()(int64_t i, int64_t j) {
  return data[i * shape[1] + j];
}
const float &Tensor::operator()(int64_t i, int64_t j) const {
  return data[i * shape[1] + j];
}

float &Tensor::operator()(int64_t i, int64_t j, int64_t k) {
  return data[i * (shape[1] * shape[2]) + j * shape[2] + k];
}
const float &Tensor::operator()(int64_t i, int64_t j, int64_t k) const {
  return data[i * (shape[1] * shape[2]) + j * shape[2] + k];
}
