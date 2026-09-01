#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Top-k result from either the naive or fused kernel.
struct TopKResult {
    float* d_scores;   // device, length k, descending order
    int*   d_indices;  // device, length k, original vocab token IDs
};

// Naive reference implementation: full (1 x d) @ (d x V) projection
// followed by cub DeviceRadixSort for top-k selection.
// Temperature T=0 means greedy (no temperature scaling).
void naive_lmhead_topk(
    cublasHandle_t  cublas,
    const __half*   d_query,    // device, length d (hidden state)
    const __half*   d_W,        // device, (d x V) row-major fp16
    int             d,
    int             V,
    int             k,
    float           temperature,
    TopKResult&     result,     // pre-allocated device buffers of length k
    float*          d_workspace,    // device scratch, length V (fp32 logits)
    void*           d_sort_tmp,     // device scratch for cub sort
    size_t          sort_tmp_bytes,
    cudaStream_t    stream = nullptr
);

// Returns the number of bytes needed for the cub sort temp buffer
// for a given vocabulary size V and top-k k.
size_t naive_sort_tmp_bytes(int V);
