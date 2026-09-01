#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Reorders vocabulary columns of W by descending L2 norm.
// Produces a new weight matrix W_reordered and a permutation array
// such that perm[reordered_idx] == original_token_id.
//
// W_reordered (d x V) and perm (V) must be pre-allocated by the caller.
// col_norms is the output of compute_tile_metadata (length V, device).
//
// After this call, the fused kernel processes W_reordered column-by-column
// and uses perm to map back to original vocab indices in its output.
void reorder_vocab_by_norm(
    const __half* d_W,             // device, original (d x V) row-major fp16
    const float*  d_col_norms,     // device, length V, pre-computed norms
    __half*       d_W_reordered,   // device, output (d x V) row-major fp16
    int*          d_perm,          // device, output length V: perm[new_idx]=orig_id
    int d,
    int V,
    cudaStream_t  stream = nullptr
);
