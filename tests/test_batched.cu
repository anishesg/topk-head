#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "fused_topk_proj.cuh"
#include "fused_topk_proj_batched.cuh"

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                       \
        cublasStatus_t e = (call);                                             \
        if (e != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                    __FILE__, __LINE__, (int)e);                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void fill_random_fp16(__half* d_buf, int n, float scale, unsigned seed) {
    std::vector<__half> h(n);
    srand(seed);
    for (int i = 0; i < n; i++)
        h[i] = __float2half(((float)rand()/RAND_MAX * 2.f - 1.f) * scale);
    CHECK_CUDA(cudaMemcpy(d_buf, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

// Correctness: batched output must match per-token fused decode loop
// for each token in the batch.
static bool run_batch_test(cublasHandle_t cublas, int V, int d, int k, int B) {
    __half *d_W, *d_queries;
    CHECK_CUDA(cudaMalloc(&d_W,      sizeof(__half) * (long long)d * V));
    CHECK_CUDA(cudaMalloc(&d_queries, sizeof(__half) * (long long)B * d));
    fill_random_fp16(d_W,       (long long)d * V, 0.02f, 42 + V + d + B);
    fill_random_fp16(d_queries, (long long)B * d, 1.0f,  99 + V + d + B);

    LMHeadState state = build_lmhead_state(d_W, d, V);

    // Batched decode
    float* d_batch_scores;
    int*   d_batch_indices;
    CHECK_CUDA(cudaMalloc(&d_batch_scores,  sizeof(float) * (long long)B * k));
    CHECK_CUDA(cudaMalloc(&d_batch_indices, sizeof(int)   * (long long)B * k));

    FusedTopKConfig cfg = {k, 1.0f, SamplingMode::TopK, 0.9f, 0.5f};
    batched_fused_topk_decode(state, d_queries, B, cfg,
                              d_batch_scores, d_batch_indices, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Per-token decode loop
    float* d_single_scores;
    int*   d_single_indices;
    CHECK_CUDA(cudaMalloc(&d_single_scores,  sizeof(float) * k));
    CHECK_CUDA(cudaMalloc(&d_single_indices, sizeof(int)   * k));
    TopKResult single_r = {d_single_scores, d_single_indices};

    std::vector<float> h_batch_s(B * k), h_single_s(k);
    std::vector<int>   h_batch_i(B * k), h_single_i(k);
    CHECK_CUDA(cudaMemcpy(h_batch_s.data(), d_batch_scores,  B*k*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_batch_i.data(), d_batch_indices, B*k*sizeof(int),   cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int b = 0; b < B; b++) {
        const __half* d_q_b = d_queries + (long long)b * d;
        fused_topk_decode(cublas, state, d_q_b, cfg, single_r, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_single_s.data(), d_single_scores,  k*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_single_i.data(), d_single_indices, k*sizeof(int),   cudaMemcpyDeviceToHost));

        // Compare top-1 index (greedy)
        if (h_batch_i[b * k + 0] != h_single_i[0]) mismatches++;
    }

    bool passed = (mismatches == 0);
    printf("  V=%6d d=%4d k=%3d B=%2d: %s (%d/%d top-1 mismatches)\n",
           V, d, k, B, passed ? "PASS" : "FAIL", mismatches, B);

    free_lmhead_state(state);
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_batch_scores));
    CHECK_CUDA(cudaFree(d_batch_indices));
    CHECK_CUDA(cudaFree(d_single_scores));
    CHECK_CUDA(cudaFree(d_single_indices));

    return passed;
}

int main() {
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    printf("=== Batched Correctness Tests ===\n");
    int total = 0, failed = 0;

    const int Bs[] = {1, 4, 16, 32};
    for (int B : Bs) {
        bool ok = run_batch_test(cublas, 32768, 4096, 50, B);
        total++; if (!ok) failed++;
    }
    for (int B : Bs) {
        bool ok = run_batch_test(cublas, 65536, 4096, 1, B);
        total++; if (!ok) failed++;
    }

    printf("\n=== Summary: %d/%d passed ===\n", total - failed, total);
    CHECK_CUBLAS(cublasDestroy(cublas));
    return (failed == 0) ? 0 : 1;
}
