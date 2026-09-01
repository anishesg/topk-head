#include "naive_lmhead.cuh"
#include <cub/cub.cuh>
#include <stdexcept>
#include <cstring>

// Tiled shared-memory matvec: computes logits[v] = dot(query, W[:,v]) for all v.
// Launch with grid = (ceil(V/BLOCK_V), 1), block = (BLOCK_V, 1).
// This is the reference path; not optimized for throughput.
static constexpr int NAIVE_BLOCK_V = 128;

static __global__ void matvec_fp16_to_fp32_kernel(
    const __half* __restrict__ query,   // length d
    const __half* __restrict__ W,       // (d x V) row-major
    float*        __restrict__ logits,  // length V
    int d,
    int V)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= V) return;

    float acc = 0.0f;
    for (int row = 0; row < d; row++) {
        acc += __half2float(query[row]) * __half2float(W[row * V + col]);
    }
    logits[col] = acc;
}

// Apply temperature: logits[v] /= T. T=0 means no scaling (greedy).
static __global__ void apply_temperature_kernel(float* logits, int V, float inv_T) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < V) logits[v] *= inv_T;
}

// After radix sort on negated logits, the first k elements are the top-k.
// Negate scores back for output.
static __global__ void negate_scores_kernel(float* scores, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) scores[i] = -scores[i];
}

size_t naive_sort_tmp_bytes(int V) {
    size_t tmp = 0;
    // Dummy call to get required size
    float* dummy_keys_in  = nullptr;
    float* dummy_keys_out = nullptr;
    int*   dummy_vals_out = nullptr;
    cub::DeviceRadixSort::SortPairs(
        nullptr, tmp,
        dummy_keys_in, dummy_keys_out,
        (int*)nullptr, dummy_vals_out,
        V
    );
    return tmp;
}

void naive_lmhead_topk(
    cublasHandle_t  cublas,
    const __half*   d_query,
    const __half*   d_W,
    int             d,
    int             V,
    int             k,
    float           temperature,
    TopKResult&     result,
    float*          d_workspace,
    void*           d_sort_tmp,
    size_t          sort_tmp_bytes,
    cudaStream_t    stream)
{
    // Step 1: full matvec to get logits
    int grid = (V + NAIVE_BLOCK_V - 1) / NAIVE_BLOCK_V;
    matvec_fp16_to_fp32_kernel<<<grid, NAIVE_BLOCK_V, 0, stream>>>(
        d_query, d_W, d_workspace, d, V);

    // Step 2: temperature scaling
    if (temperature > 0.0f && temperature != 1.0f) {
        apply_temperature_kernel<<<grid, NAIVE_BLOCK_V, 0, stream>>>(
            d_workspace, V, 1.0f / temperature);
    }

    // Step 3: sort by negated logits so the smallest negated = highest logit
    // We use a secondary index buffer (result.d_indices) as value buffer.
    // Build a [0..V) index array then sort by key.
    // We need a temporary V-element index array and a V-element key copy.
    // Pack them into d_sort_tmp after the cub scratch region.
    // Layout: [cub_tmp | neg_logits copy (V floats) | idx_in (V ints)]
    size_t cub_bytes = naive_sort_tmp_bytes(V);
    float* d_neg_logits_copy = reinterpret_cast<float*>(
        static_cast<char*>(d_sort_tmp) + cub_bytes);
    int* d_idx_in = reinterpret_cast<int*>(d_neg_logits_copy + V);

    // Fill d_idx_in with [0..V) and negate logits
    // We do this with a small kernel
    auto fill_negs = [&](float* neg_out, int* idx_out, const float* logits, int n) {
        // Inline kernel via lambda capture
        thrust::counting_iterator<int> cnt(0);
        // Use explicit kernels instead
    };

    // Kernel to negate logits and fill indices
    {
        struct NegFillKernel {
            static __global__ void run(
                const float* __restrict__ logits,
                float*       __restrict__ neg_out,
                int*         __restrict__ idx_out,
                int n)
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    neg_out[i] = -logits[i];
                    idx_out[i] = i;
                }
            }
        };
        int g = (V + 255) / 256;
        NegFillKernel::run<<<g, 256, 0, stream>>>(d_workspace, d_neg_logits_copy, d_idx_in, V);
    }

    // Sort: top-k appear first in result arrays after partial sort.
    // cub DeviceRadixSort sorts all V; we only keep first k.
    // For efficiency we sort all V (the intent is reference correctness).
    size_t tmp_actual = cub_bytes;
    cub::DeviceRadixSort::SortPairs(
        d_sort_tmp, tmp_actual,
        d_neg_logits_copy, d_workspace,   // reuse d_workspace as keys_out
        d_idx_in, result.d_indices,        // values_out = result indices
        V, 0, 32, stream
    );

    // Copy top-k scores (negated back to positive) to result
    // d_workspace now holds sorted neg_logits; first k are largest logits negated
    negate_scores_kernel<<<(k + 255) / 256, 256, 0, stream>>>(d_workspace, k);
    cudaMemcpyAsync(result.d_scores, d_workspace, k * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}
