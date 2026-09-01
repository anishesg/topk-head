#include "fused_topk_proj_batched.cuh"
#include "register_heap.cuh"
#include "tile_metadata.cuh"
#include <float.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Batched fused kernel: one thread block per token.
// The block has 1 thread (single-token decode is naturally single-threaded
// due to the sequential heap structure), but per-block q_norm is computed
// in shared memory using all 32 threads (one warp).
// Grid dim = batch_size; each block independently scans the vocabulary.
// ---------------------------------------------------------------------------
template <int K>
static __global__ void batched_fused_topk_kernel(
    const __half*        __restrict__ W,          // (d x V) reordered
    const __half*        __restrict__ queries,    // (batch x d)
    const VocabTileMeta* __restrict__ tile_meta,
    const int*           __restrict__ perm,
    float                inv_temp,
    int                  d,
    int                  V,
    int                  num_tiles,
    float*               __restrict__ out_scores,  // (batch x K)
    int*                 __restrict__ out_indices  // (batch x K)
)
{
    const int token = blockIdx.x;
    const __half* query = queries + (long long)token * d;

    // Compute q_norm in shared memory using one warp of threads.
    // Only thread 0 runs the heap; other threads help with the norm.
    __shared__ float s_q_norm;

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(query[i]);
        local_ss += v * v;
    }
    // Warp reduction
    for (int off = 16; off > 0; off >>= 1)
        local_ss += __shfl_down_sync(0xffffffff, local_ss, off);
    if (threadIdx.x == 0) s_q_norm = sqrtf(local_ss);
    __syncthreads();

    // Only thread 0 runs the vocabulary scan and heap
    if (threadIdx.x != 0) return;

    float q_norm = s_q_norm;
    TopKHeap<K> heap;
    heap.init();

    for (int t = 0; t < num_tiles; t++) {
        const VocabTileMeta& meta = tile_meta[t];
        float effective_bound = q_norm * meta.max_col_norm;
        if (inv_temp != 1.0f) effective_bound *= inv_temp;

        if (heap.full() && effective_bound <= heap.peek_min()) continue;

        int tile_start = meta.tile_start;
        int tile_len   = meta.tile_len;

        for (int v = 0; v < tile_len; v++) {
            int col = tile_start + v;
            float acc = 0.0f;
            for (int row = 0; row < d; row++) {
                acc += __half2float(query[row]) * __half2float(W[(long long)row * V + col]);
            }
            float score = (inv_temp != 1.0f) ? acc * inv_temp : acc;
            heap.insert(score, perm[col]);
        }
    }

    float* token_scores  = out_scores  + (long long)token * K;
    int*   token_indices = out_indices + (long long)token * K;
    heap.extract_sorted(token_scores, token_indices);
}

// Nucleus sampling for batched output (one block per token, 32 threads).
static __global__ void batched_nucleus_sample_kernel(
    const float* __restrict__ scores,   // (batch x k)
    const int*   __restrict__ indices,  // (batch x k)
    int          k,
    float        top_p,
    float        uniform_rand,
    int*         __restrict__ sampled   // (batch)
)
{
    const int token = blockIdx.x;
    const float* s = scores  + (long long)token * k;
    const int*   idx = indices + (long long)token * k;
    int* out = sampled + token;

    const int lane = threadIdx.x;

    __shared__ float s_probs[256];  // max k
    __shared__ int   s_end;

    // Parallel exp-normalize
    float max_s = s[0];
    float local_sum = 0.0f;
    for (int i = lane; i < k; i += 32) {
        float p = expf(s[i] - max_s);
        s_probs[i] = p;
        local_sum += p;
    }
    __syncwarp();

    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    float total = __shfl_sync(0xffffffff, local_sum, 0);

    for (int i = lane; i < k; i += 32)
        s_probs[i] /= total;
    __syncwarp();

    if (lane == 0) {
        float cum = 0.0f;
        s_end = k;
        for (int i = 0; i < k; i++) {
            cum += s_probs[i];
            if (cum >= top_p) { s_end = i + 1; break; }
        }
        float nm = 0.0f;
        for (int i = 0; i < s_end; i++) nm += s_probs[i];
        for (int i = 0; i < s_end; i++) s_probs[i] /= nm;

        float cdf = 0.0f;
        *out = idx[s_end - 1];
        for (int i = 0; i < s_end; i++) {
            cdf += s_probs[i];
            if (cdf >= uniform_rand) { *out = idx[i]; break; }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper for batched kernel
// ---------------------------------------------------------------------------
template <int K>
static void dispatch_batched(
    const LMHeadState& state,
    const __half* d_queries,
    int batch,
    float inv_temp,
    float* d_out_scores,
    int*   d_out_indices,
    cudaStream_t stream)
{
    // One block per token, 32 threads per block (one warp: norm + heap)
    batched_fused_topk_kernel<K><<<batch, 32, 0, stream>>>(
        state.d_W_reordered, d_queries,
        state.d_tile_meta, state.d_perm,
        inv_temp,
        state.d, state.V, state.num_tiles,
        d_out_scores, d_out_indices
    );
}

void batched_fused_topk_decode(
    const LMHeadState&     state,
    const __half*          d_queries,
    int                    batch,
    const FusedTopKConfig& cfg,
    float*                 d_out_scores,
    int*                   d_out_indices,
    int*                   d_sampled,
    cudaStream_t           stream)
{
    int k = cfg.k;
    float inv_temp = (cfg.temperature > 0.0f) ? (1.0f / cfg.temperature) : 1.0f;

    switch (k) {
        case 1:   dispatch_batched<1>  (state, d_queries, batch, inv_temp, d_out_scores, d_out_indices, stream); break;
        case 10:  dispatch_batched<10> (state, d_queries, batch, inv_temp, d_out_scores, d_out_indices, stream); break;
        case 50:  dispatch_batched<50> (state, d_queries, batch, inv_temp, d_out_scores, d_out_indices, stream); break;
        case 256: dispatch_batched<256>(state, d_queries, batch, inv_temp, d_out_scores, d_out_indices, stream); break;
        default:  dispatch_batched<256>(state, d_queries, batch, inv_temp, d_out_scores, d_out_indices, stream); break;
    }

    if (cfg.mode == SamplingMode::TopP && d_sampled != nullptr) {
        batched_nucleus_sample_kernel<<<batch, 32, 0, stream>>>(
            d_out_scores, d_out_indices, k,
            cfg.top_p, cfg.uniform_rand, d_sampled
        );
    }
}
