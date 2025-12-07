#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

class Tensor {
public:
  std::vector<int64_t> shape; // d1, d2, ..
  std::vector<float> h_data;
  float *d_data = nullptr;

  /**
   * Constructors, Assignment Operators, and Destructor
   * a.k.a "The rule of five"
   */
  Tensor();
  Tensor(const std::vector<int64_t> &shape, std::string device = "gpu",
         bool zeros = false);

  Tensor(const Tensor &);
  Tensor &operator=(const Tensor &);

  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  ~Tensor();

  /**
   * Member functions
   */

  int64_t numel() const;
  void to(std::string device);

  /**
   * Host-side helper functions
   */

  // Iterators to support range-based for loops and algorithms.
  std::vector<float>::iterator begin();
  std::vector<float>::iterator end();
  std::vector<float>::const_iterator begin() const;
  std::vector<float>::const_iterator end() const;

  // Operator for easy element accesses. (h_data)
  // Using `()` since `[]` with multi-arguments is not supported before C++23.
  float &operator()(int64_t i);
  const float &operator()(int64_t i) const;

  float &operator()(int64_t i, int64_t j);
  const float &operator()(int64_t i, int64_t j) const;

  float &operator()(int64_t i, int64_t j, int64_t k);
  const float &operator()(int64_t i, int64_t j, int64_t k) const;
};
