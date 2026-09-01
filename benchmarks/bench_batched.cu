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

static void fill_random_fp16(__half* d_buf, long long n, float scale) {
    std::vector<__half> h(n);
    for (long long i = 0; i < n; i++)
        h[i] = __float2half(((float)rand()/RAND_MAX * 2.f - 1.f) * scale);
    CHECK_CUDA(cudaMemcpy(d_buf, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

static float time_ms(cudaEvent_t s, cudaEvent_t e, int ITERS, auto&& fn) {
    for (int i = 0; i < 3; i++) fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(s));
    for (int i = 0; i < ITERS; i++) fn();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    return ms / ITERS;
}

int main() {
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    cudaEvent_t ev_s, ev_e;
    CHECK_CUDA(cudaEventCreate(&ev_s));
    CHECK_CUDA(cudaEventCreate(&ev_e));

    const int V = 128256, d = 4096, k = 50;
    const int batch_sizes[] = {1, 4, 8, 16, 32, 64};
    const int ITERS = 50;

    srand(12345);

    __half* d_W;
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(__half) * (long long)d * V));
    fill_random_fp16(d_W, (long long)d * V, 0.02f);

    LMHeadState state = build_lmhead_state(d_W, d, V);

    printf("Batched throughput benchmark: V=%d d=%d k=%d\n\n", V, d, k);
    printf("%-8s %12s %14s\n", "Batch", "Latency(ms)", "Tokens/sec");
    printf("%s\n", std::string(40, '-').c_str());

    for (int B : batch_sizes) {
        __half* d_queries;
        float*  d_out_scores;
        int*    d_out_indices;
        CHECK_CUDA(cudaMalloc(&d_queries,    sizeof(__half) * (long long)B * d));
        CHECK_CUDA(cudaMalloc(&d_out_scores,  sizeof(float) * (long long)B * k));
        CHECK_CUDA(cudaMalloc(&d_out_indices, sizeof(int)   * (long long)B * k));
        fill_random_fp16(d_queries, (long long)B * d, 1.0f);

        FusedTopKConfig cfg = {k, 1.0f, SamplingMode::TopK, 0.9f, 0.5f};

        float ms = time_ms(ev_s, ev_e, ITERS, [&]() {
            batched_fused_topk_decode(
                state, d_queries, B, cfg,
                d_out_scores, d_out_indices, nullptr);
        });

        float tok_per_sec = (float)B * 1000.0f / ms;
        printf("%-8d %12.3f %14.0f\n", B, ms, tok_per_sec);

        CHECK_CUDA(cudaFree(d_queries));
        CHECK_CUDA(cudaFree(d_out_scores));
        CHECK_CUDA(cudaFree(d_out_indices));
    }

    free_lmhead_state(state);
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaEventDestroy(ev_s));
    CHECK_CUDA(cudaEventDestroy(ev_e));
    CHECK_CUBLAS(cublasDestroy(cublas));
    return 0;
}
