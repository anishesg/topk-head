#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "naive_lmhead.cuh"   // TopKResult
#include "tile_metadata.cuh"  // VocabTileMeta

// Sampling mode passed to the fused kernel.
enum class SamplingMode : int {
    Greedy   = 0,   // return top-1 only; temperature ignored
    TopK     = 1,   // return top-k raw scores; temperature applied
    TopP     = 2,   // nucleus sampling: top-p from temperature-scaled distribution
};

// Configuration for the fused kernel.
struct FusedTopKConfig {
    int           k;
    float         temperature;   // 0 or negative => greedy (no scaling)
    SamplingMode  mode;
    float         top_p;         // used only when mode == TopP
    float         uniform_rand;  // used only when mode == TopP for categorical sample
};

// Precomputed state for the LM head (computed once per model load or weight change).
// Holds the reordered weight matrix, permutation, tile metadata, and norms.
struct LMHeadState {
    __half*        d_W_reordered;    // (d x V) row-major fp16
    int*           d_perm;           // length V: perm[reordered_idx] = orig_token_id
    VocabTileMeta* d_tile_meta;      // length num_tiles
    float*         d_col_norms;      // length V
    int            d,  V,  num_tiles;
};

// Allocate and compute LMHeadState from raw weight matrix.
LMHeadState build_lmhead_state(
    const __half* d_W,    // device, original (d x V)
    int d,
    int V,
    cudaStream_t stream = nullptr
);

// Free all device memory in an LMHeadState.
void free_lmhead_state(LMHeadState& state);

// Single-token fused top-k decode.
// Uses Cauchy-Schwarz tile pruning and a per-thread register heap.
// All scratch memory is in registers and shared memory; no V-sized global buffer.
//
// result.d_scores / result.d_indices must be pre-allocated to length k.
// d_sampled_index: length 1, output for TopP mode (sampled token ID).
void fused_topk_decode(
    cublasHandle_t       cublas,
    const LMHeadState&   state,
    const __half*        d_query,       // device, length d
    const FusedTopKConfig& cfg,
    TopKResult&          result,        // output top-k scores + indices
    int*                 d_sampled_index,   // output for TopP mode
    cudaStream_t         stream = nullptr
);
