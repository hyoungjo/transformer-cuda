#pragma once

#include <cstdint>
#include <vector>

class Tensor {
public:
  std::vector<int64_t> shape; // d1, d2, ..
  std::vector<float> data;

  Tensor();
  Tensor(const std::vector<int64_t> &shape);

  int64_t numel() const;
  void resize(const std::vector<int64_t> &shape);

  /**
   * Iterators to support range-based for loops and algorithms.
   */
  std::vector<float>::iterator begin();
  std::vector<float>::iterator end();
  std::vector<float>::const_iterator begin() const;
  std::vector<float>::const_iterator end() const;

  /**
   * Operator for easy element accesses.
   * Using `()` since `[]` with multi-arguments is not supported before C++23.
   */
  float &operator()(int64_t i);
  const float &operator()(int64_t i) const;

  float &operator()(int64_t i, int64_t j);
  const float &operator()(int64_t i, int64_t j) const;

  float &operator()(int64_t i, int64_t j, int64_t k);
  const float &operator()(int64_t i, int64_t j, int64_t k) const;
};
