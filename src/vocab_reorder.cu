#include "vocab_reorder.cuh"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// Copy columns of W into W_reordered according to a permutation array.
// Each thread handles one (row, col) element.
static __global__ void permute_columns_kernel(
    const __half* __restrict__ W,
    const int*    __restrict__ perm,
    __half*       __restrict__ W_out,
    int d,
    int V)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= V || row >= d) return;
    // perm[col] is the original column index for reordered position col
    W_out[row * V + col] = W[row * V + perm[col]];
}

void reorder_vocab_by_norm(
    const __half* d_W,
    const float*  d_col_norms,
    __half*       d_W_reordered,
    int*          d_perm,
    int           d,
    int           V,
    cudaStream_t  stream)
{
    // Build index array [0, 1, ..., V-1]
    thrust::device_ptr<int> perm_ptr(d_perm);
    thrust::sequence(thrust::cuda::par.on(stream), perm_ptr, perm_ptr + V, 0);

    // Sort indices by descending column norm using a stable sort on negated norms.
    // We sort a copy of norms (negated) alongside indices.
    thrust::device_vector<float> neg_norms(V);
    thrust::device_ptr<const float> norms_ptr(d_col_norms);
    thrust::transform(
        thrust::cuda::par.on(stream),
        norms_ptr, norms_ptr + V,
        neg_norms.begin(),
        [] __device__ (float x) { return -x; }
    );

    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        neg_norms.begin(), neg_norms.end(),
        perm_ptr
    );

    // Permute the weight matrix columns
    dim3 block(32, 8);
    dim3 grid((V + block.x - 1) / block.x, (d + block.y - 1) / block.y);
    permute_columns_kernel<<<grid, block, 0, stream>>>(d_W, d_perm, d_W_reordered, d, V);
}
