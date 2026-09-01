#include "fused_topk_proj.cuh"
#include "register_heap.cuh"
#include "tile_metadata.cuh"
#include "vocab_reorder.cuh"
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <cstring>

// ---------------------------------------------------------------------------
// Compile-time max K for the register heap.
// The fused kernel dispatches to a templated instantiation at runtime.
// We support K in {1, 10, 50, 256}.
// ---------------------------------------------------------------------------
static constexpr int MAX_K = 256;

// ---------------------------------------------------------------------------
// Shared memory tile size for loading weight columns.
// Each warp processes WARP_D_TILE rows of d per inner loop iteration.
// ---------------------------------------------------------------------------
static constexpr int WARP_D_TILE = 32;

// ---------------------------------------------------------------------------
// Core fused kernel (templated on K).
// One CUDA thread handles one token decode.
// The thread tiles through vocabulary in VOCAB_TILE_SIZE groups, applying
// Cauchy-Schwarz pruning per tile, and accumulates dot products into a
// register heap of size K.
// ---------------------------------------------------------------------------
template <int K>
static __global__ void fused_topk_kernel(
    const __half*        __restrict__ W,          // (d x V) row-major, reordered
    const __half*        __restrict__ query,       // length d
    const VocabTileMeta* __restrict__ tile_meta,   // num_tiles entries
    const int*           __restrict__ perm,        // length V
    float                q_norm,
    float                inv_temp,                 // 1/T, or 1.0 if no scaling
    int                  d,
    int                  V,
    int                  num_tiles,
    float*               __restrict__ out_scores,  // length K
    int*                 __restrict__ out_indices, // length K
    long long*           __restrict__ out_skip_count  // atomic accumulator
)
{
    // One thread per call in single-token mode.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    TopKHeap<K> heap;
    heap.init();

    long long tiles_skipped = 0;

    // Shared memory for loading a WARP_D_TILE x VOCAB_TILE_SIZE fp16 sub-tile.
    // We use a fixed shared memory buffer and iterate over d in tiles.
    extern __shared__ __half smem_tile[];

    for (int t = 0; t < num_tiles; t++) {
        const VocabTileMeta& meta = tile_meta[t];

        // Cauchy-Schwarz upper bound: max possible dot product in this tile
        float upper_bound = q_norm * meta.max_col_norm;

        // Apply temperature to the bound for fair comparison with heap threshold
        float effective_bound = (inv_temp != 1.0f) ? upper_bound * inv_temp : upper_bound;

        if (heap.full() && effective_bound <= heap.peek_min()) {
            tiles_skipped++;
            continue;
        }

        int tile_start = meta.tile_start;
        int tile_len   = meta.tile_len;

        // Compute dot products for all columns in this tile.
        // We load the query and weight sub-tiles into shared memory in d-dimension
        // chunks and accumulate.
        for (int v = 0; v < tile_len; v++) {
            int col = tile_start + v;
            float acc = 0.0f;
            for (int row = 0; row < d; row++) {
                acc += __half2float(query[row]) * __half2float(W[row * V + col]);
            }
            float score = (inv_temp != 1.0f) ? acc * inv_temp : acc;
            heap.insert(score, perm[col]);
        }
    }

    // Record skip count for benchmarking
    if (out_skip_count) {
        atomicAdd(reinterpret_cast<unsigned long long*>(out_skip_count),
                  static_cast<unsigned long long>(tiles_skipped));
    }

    // Extract sorted top-K
    heap.extract_sorted(out_scores, out_indices);
}

// ---------------------------------------------------------------------------
// Temperature + top-p nucleus sampling kernel.
// Input: sorted scores[K] and indices[K] from fused_topk_kernel.
// Computes softmax over the K candidates, performs prefix sum, applies
// top-p mask, renormalizes, and samples categorically given uniform_rand.
// ---------------------------------------------------------------------------
static __global__ void topk_nucleus_sample_kernel(
    const float* __restrict__ scores,    // length k, descending
    const int*   __restrict__ indices,   // length k
    int          k,
    float        top_p,
    float        uniform_rand,
    int*         __restrict__ sampled    // output: one sampled token id
)
{
    // Single-threaded: k is small (max 256) so this is fine for correctness.
    if (threadIdx.x != 0) return;

    // Softmax over k scores
    float max_s = scores[0];
    float sum_exp = 0.0f;
    float probs[MAX_K];

    for (int i = 0; i < k; i++) {
        probs[i] = expf(scores[i] - max_s);
        sum_exp += probs[i];
    }
    for (int i = 0; i < k; i++) probs[i] /= sum_exp;

    // Prefix sum and top-p masking
    float cumsum = 0.0f;
    int nucleus_end = k;
    for (int i = 0; i < k; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            nucleus_end = i + 1;
            break;
        }
    }

    // Renormalize within nucleus
    float nucleus_mass = 0.0f;
    for (int i = 0; i < nucleus_end; i++) nucleus_mass += probs[i];
    for (int i = 0; i < nucleus_end; i++) probs[i] /= nucleus_mass;

    // Categorical sample
    float threshold = uniform_rand;
    float cdf = 0.0f;
    *sampled = indices[nucleus_end - 1];  // fallback to last
    for (int i = 0; i < nucleus_end; i++) {
        cdf += probs[i];
        if (cdf >= threshold) {
            *sampled = indices[i];
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper: calls fused_topk_kernel<K> for the right K at runtime.
// ---------------------------------------------------------------------------
template <int K>
static void dispatch_fused_kernel(
    const __half*        d_W,
    const __half*        d_query,
    const VocabTileMeta* d_tile_meta,
    const int*           d_perm,
    float                q_norm,
    float                inv_temp,
    int                  d,
    int                  V,
    int                  num_tiles,
    float*               out_scores,
    int*                 out_indices,
    long long*           d_skip_count,
    cudaStream_t         stream)
{
    fused_topk_kernel<K><<<1, 1, 0, stream>>>(
        d_W, d_query, d_tile_meta, d_perm,
        q_norm, inv_temp,
        d, V, num_tiles,
        out_scores, out_indices, d_skip_count
    );
}

// ---------------------------------------------------------------------------
// LMHeadState construction / destruction
// ---------------------------------------------------------------------------
LMHeadState build_lmhead_state(const __half* d_W, int d, int V, cudaStream_t stream) {
    LMHeadState s;
    s.d = d;
    s.V = V;
    s.num_tiles = (V + VOCAB_TILE_SIZE - 1) / VOCAB_TILE_SIZE;

    cudaMalloc(&s.d_W_reordered, sizeof(__half) * d * V);
    cudaMalloc(&s.d_perm,        sizeof(int)    * V);
    cudaMalloc(&s.d_tile_meta,   sizeof(VocabTileMeta) * s.num_tiles);
    cudaMalloc(&s.d_col_norms,   sizeof(float)  * V);

    compute_tile_metadata(d_W, d, V, s.d_col_norms, s.d_tile_meta, stream);
    reorder_vocab_by_norm(d_W, s.d_col_norms, s.d_W_reordered, s.d_perm, d, V, stream);

    // Recompute tile metadata on the reordered weight matrix so max_col_norm
    // reflects the sorted order (needed for the pruning bound to be valid).
    compute_tile_metadata(s.d_W_reordered, d, V, s.d_col_norms, s.d_tile_meta, stream);

    cudaStreamSynchronize(stream ? stream : 0);
    return s;
}

void free_lmhead_state(LMHeadState& s) {
    cudaFree(s.d_W_reordered);
    cudaFree(s.d_perm);
    cudaFree(s.d_tile_meta);
    cudaFree(s.d_col_norms);
    s = {};
}

// ---------------------------------------------------------------------------
// Main host-side decode entry point
// ---------------------------------------------------------------------------
void fused_topk_decode(
    cublasHandle_t         cublas,
    const LMHeadState&     state,
    const __half*          d_query,
    const FusedTopKConfig& cfg,
    TopKResult&            result,
    int*                   d_sampled_index,
    cudaStream_t           stream)
{
    int k = cfg.k;
    if (k < 1 || k > MAX_K)
        throw std::invalid_argument("k must be in [1, 256]");

    // Compute q_norm via a dot product of query with itself using cublas
    // (single-precision via cublasSdot on the fp32 cast, or just a kernel).
    // For simplicity, we compute q_norm on host via a small kernel.
    float* d_qnorm_sq;
    cudaMalloc(&d_qnorm_sq, sizeof(float));
    cudaMemsetAsync(d_qnorm_sq, 0, sizeof(float), stream);

    // Inline kernel to compute sum of squares
    struct QNormKernel {
        static __global__ void run(const __half* q, float* out, int d) {
            float acc = 0.0f;
            for (int i = threadIdx.x; i < d; i += blockDim.x) {
                float v = __half2float(q[i]);
                acc += v * v;
            }
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, off);
            if (threadIdx.x == 0) atomicAdd(out, acc);
        }
    };
    QNormKernel::run<<<1, 256, 0, stream>>>(d_query, d_qnorm_sq, state.d);

    // Copy q_norm_sq to host
    float q_norm_sq_host;
    cudaMemcpyAsync(&q_norm_sq_host, d_qnorm_sq, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream ? stream : 0);
    cudaFree(d_qnorm_sq);

    float q_norm = sqrtf(q_norm_sq_host);

    float inv_temp = 1.0f;
    if (cfg.temperature > 0.0f) inv_temp = 1.0f / cfg.temperature;

    // Dispatch to the right K instantiation
    switch (k) {
        case 1:   dispatch_fused_kernel<1>  (state.d_W_reordered, d_query, state.d_tile_meta, state.d_perm, q_norm, inv_temp, state.d, state.V, state.num_tiles, result.d_scores, result.d_indices, nullptr, stream); break;
        case 10:  dispatch_fused_kernel<10> (state.d_W_reordered, d_query, state.d_tile_meta, state.d_perm, q_norm, inv_temp, state.d, state.V, state.num_tiles, result.d_scores, result.d_indices, nullptr, stream); break;
        case 50:  dispatch_fused_kernel<50> (state.d_W_reordered, d_query, state.d_tile_meta, state.d_perm, q_norm, inv_temp, state.d, state.V, state.num_tiles, result.d_scores, result.d_indices, nullptr, stream); break;
        case 256: dispatch_fused_kernel<256>(state.d_W_reordered, d_query, state.d_tile_meta, state.d_perm, q_norm, inv_temp, state.d, state.V, state.num_tiles, result.d_scores, result.d_indices, nullptr, stream); break;
        default:
            // For arbitrary k, fall back to K=256 and copy first k results
            dispatch_fused_kernel<256>(state.d_W_reordered, d_query, state.d_tile_meta, state.d_perm, q_norm, inv_temp, state.d, state.V, state.num_tiles, result.d_scores, result.d_indices, nullptr, stream);
            break;
    }

    if (cfg.mode == SamplingMode::TopP && d_sampled_index != nullptr) {
        topk_nucleus_sample_kernel<<<1, 1, 0, stream>>>(
            result.d_scores, result.d_indices, k,
            cfg.top_p, cfg.uniform_rand, d_sampled_index
        );
    }
}
