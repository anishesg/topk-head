#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Number of vocabulary columns per tile for Cauchy-Schwarz pruning.
// Must be a compile-time constant so the tile norm precomputation and
// the fused kernel use the same tile granularity.
static constexpr int VOCAB_TILE_SIZE = 128;

// Per-tile metadata used by the Cauchy-Schwarz pruning step.
// Stored in global memory after precomputation; accessed once per tile
// in the fused kernel to decide whether to skip.
struct VocabTileMeta {
    float max_col_norm;   // max L2 norm among all columns in this tile
    int   tile_start;     // first column index (in reordered vocab space)
    int   tile_len;       // number of valid columns in this tile (<= VOCAB_TILE_SIZE)
};

// Computes the L2 norm of every column of W (shape d x V, row-major in fp16).
// col_norms[v] = ||W[:, v]||_2  for v in [0, V).
// Launch with a 1-D grid: one thread block per column, threads cooperate
// over the d dimension with warp reductions.
__global__ void compute_col_norms_kernel(
    const __half* __restrict__ W,  // (d x V) row-major
    float*        __restrict__ col_norms,
    int d,
    int V
);

// Host wrapper: allocates col_norms on device, launches kernel, fills
// tile_meta array of length ceil(V / VOCAB_TILE_SIZE).
// Caller owns the device memory for col_norms and tile_meta.
void compute_tile_metadata(
    const __half* d_W,        // device, shape (d x V)
    int d,
    int V,
    float*        d_col_norms,   // device, length V  (output)
    VocabTileMeta* d_tile_meta,  // device, length ceil(V/VOCAB_TILE_SIZE) (output)
    cudaStream_t stream = nullptr
);
