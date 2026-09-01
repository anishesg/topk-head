#include "tile_metadata.cuh"
#include <cmath>
#include <stdexcept>

// Each block handles one vocabulary column.
// Threads stride over the d dimension accumulating sum of squares,
// then perform a warp reduction to get the column's squared norm.
__global__ void compute_col_norms_kernel(
    const __half* __restrict__ W,
    float*        __restrict__ col_norms,
    int d,
    int V)
{
    const int col = blockIdx.x;
    if (col >= V) return;

    float acc = 0.0f;
    for (int row = threadIdx.x; row < d; row += blockDim.x) {
        float val = __half2float(W[row * V + col]);
        acc += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    // One warp writes; multi-warp blocks need inter-warp accumulation
    // via shared memory.
    __shared__ float smem[32];  // one slot per warp
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) smem[warp] = acc;
    __syncthreads();

    // First warp reduces across warps
    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        acc = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        if (lane == 0) col_norms[col] = sqrtf(acc);
    }
}

// Fills tile_meta from the already-computed col_norms.
// Each tile gets the max norm among its columns and the tile start/length.
static __global__ void build_tile_meta_kernel(
    const float*   __restrict__ col_norms,
    VocabTileMeta* __restrict__ tile_meta,
    int V,
    int num_tiles)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= num_tiles) return;

    int start = tile_idx * VOCAB_TILE_SIZE;
    int len   = min(VOCAB_TILE_SIZE, V - start);
    float mx  = 0.0f;
    for (int i = 0; i < len; i++) {
        float n = col_norms[start + i];
        if (n > mx) mx = n;
    }
    tile_meta[tile_idx] = {mx, start, len};
}

void compute_tile_metadata(
    const __half*  d_W,
    int d,
    int V,
    float*         d_col_norms,
    VocabTileMeta* d_tile_meta,
    cudaStream_t   stream)
{
    // One block per column for the norm kernel; use 256 threads.
    compute_col_norms_kernel<<<V, 256, 0, stream>>>(d_W, d_col_norms, d, V);

    int num_tiles = (V + VOCAB_TILE_SIZE - 1) / VOCAB_TILE_SIZE;
    int block = 256;
    int grid  = (num_tiles + block - 1) / block;
    build_tile_meta_kernel<<<grid, block, 0, stream>>>(d_col_norms, d_tile_meta, V, num_tiles);
}
