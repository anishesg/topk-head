#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "fused_topk_proj.cuh"

// Batched fused top-k decode.
// Each token in the batch gets its own thread block, which independently
// scans the reordered vocabulary with Cauchy-Schwarz pruning and maintains
// a register-resident heap of size k.
//
// Inputs:
//   d_queries: (batch x d) row-major fp16
//   state:     precomputed LMHeadState (shared across all tokens)
//   cfg:       sampling configuration (k, temperature, mode)
//
// Outputs:
//   d_out_scores:   (batch x k) fp32, each row descending
//   d_out_indices:  (batch x k) int32, original vocab token IDs
//   d_sampled:      (batch) int32, sampled token ID (TopP mode only)
void batched_fused_topk_decode(
    const LMHeadState&     state,
    const __half*          d_queries,     // (batch x d) row-major
    int                    batch,
    const FusedTopKConfig& cfg,
    float*                 d_out_scores,  // (batch x k)
    int*                   d_out_indices, // (batch x k)
    int*                   d_sampled,     // (batch), nullable
    cudaStream_t           stream = nullptr
);
