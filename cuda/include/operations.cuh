#pragma once

#include "tensor.cuh"

/**
 * The functions are designed not to return any values because CUDA kernel
 * launches are asynchronous. The function acts solely as a dispatcher to the
 * GPU, and the control returns immediately so that the CPU can continue.
 *
 * However, if a function returns a value, the CPU will block and wait for the
 * the kernel execution to be completed.
 */

namespace operations {

void embed(Tensor &x, const Tensor &embeddings, const int *input_ids);
void positional_encoding(Tensor &x, const Tensor &positional_encoding);

void rope(Tensor &x, int head_dim);

void transpose(Tensor &out, const Tensor &x);
void matmul(Tensor &out, const Tensor &A, const Tensor &B,
            bool transpose_b = false);

void add(Tensor &x, const Tensor &y);
void add_bias(Tensor &x, const Tensor &bias);

void multiply(Tensor &x, const Tensor &y);

void gelu(Tensor &x);
void silu(Tensor &x);

void layer_norm(Tensor &x, const Tensor &g, const Tensor &b, float eps = 1e-5f);
void rms_norm(Tensor &x, const Tensor &weight, float eps = 1e-5f);
void softmax(Tensor &x);

} // namespace operations
