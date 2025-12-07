#pragma once

#include "tensor.hpp"

/**
 * The functions are designed not to return any values because CUDA kernel
 * launches are asynchronous. The function acts solely as a dispatcher to the
 * GPU, and the control returns immediately so that the CPU can continue.
 *
 * However, if a function returns a value, the CPU will block and wait for the
 * the kernel execution to be completed.
 */

namespace operations {

void embedding(Tensor &x, const Tensor &wte, const Tensor &wpe,
               const std::vector<int> &input_ids);

void transpose(Tensor &x);
void matmul(Tensor &out, const Tensor &A, const Tensor &B);

void add(Tensor &x, const Tensor &y);
void add_bias(Tensor &x, const Tensor &bias);

void gelu(Tensor &x);

void layer_norm(Tensor &x, const Tensor &g, const Tensor &b, float eps = 1e-5f);
void softmax(Tensor &x);

} // namespace operations
