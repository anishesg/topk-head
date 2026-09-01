#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "fused_topk_proj.cuh"
#include "naive_lmhead.cuh"

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

struct BenchConfig {
    const char* name;
    int V, d, k;
};

// CUDA event-based timing: returns milliseconds for ITERS kernel launches.
static float time_kernel_ms(cudaEvent_t start, cudaEvent_t stop,
                             int ITERS, auto&& fn)
{
    // Warmup
    for (int i = 0; i < 3; i++) fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms / ITERS;
}

// Query tile skip rate by reading the skip counter written by fused kernel.
// Returns fraction of tiles skipped.
static float measure_skip_rate(
    cublasHandle_t cublas,
    const LMHeadState& state,
    const __half* d_query,
    int k)
{
    long long* d_skip;
    CHECK_CUDA(cudaMalloc(&d_skip, sizeof(long long)));
    CHECK_CUDA(cudaMemset(d_skip, 0, sizeof(long long)));

    float *d_fs; int *d_fi;
    CHECK_CUDA(cudaMalloc(&d_fs, sizeof(float) * k));
    CHECK_CUDA(cudaMalloc(&d_fi, sizeof(int)   * k));
    TopKResult res = {d_fs, d_fi};

    FusedTopKConfig cfg = {k, 1.0f, SamplingMode::TopK, 0.9f, 0.5f};

    // Run fused kernel manually with skip counter pointer
    // We access the internal dispatch here by re-implementing a thin wrapper.
    // For the skip counter, we patch in via a separate launch.
    // Since the public API doesn't expose skip counter, we compute it
    // analytically: iterate over tiles and count those where
    // bound < (k-th score from fused result).
    fused_topk_decode(cublas, state, d_query, cfg, res, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read top-k min score
    std::vector<float> h_scores(k);
    CHECK_CUDA(cudaMemcpy(h_scores.data(), d_fs, k*sizeof(float), cudaMemcpyDeviceToHost));
    float kth_score = h_scores[k-1];

    // Copy tile metadata to host and count skippable tiles
    std::vector<VocabTileMeta> h_meta(state.num_tiles);
    CHECK_CUDA(cudaMemcpy(h_meta.data(), state.d_tile_meta,
                          state.num_tiles * sizeof(VocabTileMeta),
                          cudaMemcpyDeviceToHost));

    // Compute q_norm on host
    std::vector<__half> h_q(state.d);
    CHECK_CUDA(cudaMemcpy(h_q.data(), d_query, state.d * sizeof(__half), cudaMemcpyDeviceToHost));
    float q_norm_sq = 0.0f;
    for (int i = 0; i < state.d; i++) {
        float v = __half2float(h_q[i]);
        q_norm_sq += v * v;
    }
    float q_norm = sqrtf(q_norm_sq);

    int skipped = 0;
    for (int t = 0; t < state.num_tiles; t++) {
        float bound = q_norm * h_meta[t].max_col_norm;
        if (bound <= kth_score) skipped++;
    }

    CHECK_CUDA(cudaFree(d_skip));
    CHECK_CUDA(cudaFree(d_fs));
    CHECK_CUDA(cudaFree(d_fi));

    return (float)skipped / state.num_tiles;
}

static void fill_random_fp16(__half* d_buf, int n, float scale) {
    std::vector<__half> h(n);
    for (int i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        h[i] = __float2half(v);
    }
    CHECK_CUDA(cudaMemcpy(d_buf, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

int main() {
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    BenchConfig configs[] = {
        {"Llama-3 (V=128256 d=4096)",  128256, 4096, 50},
        {"Llama-3 (V=128256 d=8192)",  128256, 8192, 50},
        {"GPT-4-class (V=100256 d=8192)", 100256, 8192, 50},
        {"Mistral (V=32000 d=4096)",    32000,  4096, 50},
    };

    const int ITERS = 100;
    srand(12345);

    printf("%-40s %10s %10s %10s %10s %12s %14s\n",
           "Config", "Naive(ms)", "Fused(ms)", "Speedup",
           "SkipRate", "NaiveMem(MB)", "Tokens/sec");
    printf("%s\n", std::string(110, '-').c_str());

    for (auto& cfg : configs) {
        int V = cfg.V, d = cfg.d, k = cfg.k;

        __half *d_W, *d_query;
        CHECK_CUDA(cudaMalloc(&d_W,     sizeof(__half) * (long long)d * V));
        CHECK_CUDA(cudaMalloc(&d_query, sizeof(__half) * d));
        fill_random_fp16(d_W,     (long long)d * V, 0.02f);
        fill_random_fp16(d_query, d, 1.0f);

        // Naive buffers
        float* d_ws;
        size_t sb = naive_sort_tmp_bytes(V);
        size_t ex = (size_t)V * sizeof(float) + (size_t)V * sizeof(int);
        void*  d_st;
        float *d_ns; int *d_ni;
        CHECK_CUDA(cudaMalloc(&d_ws, sizeof(float) * V));
        CHECK_CUDA(cudaMalloc(&d_st, sb + ex));
        CHECK_CUDA(cudaMalloc(&d_ns, sizeof(float) * k));
        CHECK_CUDA(cudaMalloc(&d_ni, sizeof(int)   * k));
        TopKResult naive_r = {d_ns, d_ni};

        // Fused state + buffers
        LMHeadState state = build_lmhead_state(d_W, d, V);
        float *d_fs; int *d_fi;
        CHECK_CUDA(cudaMalloc(&d_fs, sizeof(float) * k));
        CHECK_CUDA(cudaMalloc(&d_fi, sizeof(int)   * k));
        TopKResult fused_r = {d_fs, d_fi};

        FusedTopKConfig fused_cfg = {k, 1.0f, SamplingMode::TopK, 0.9f, 0.5f};

        // Time naive
        float naive_ms = time_kernel_ms(ev_start, ev_stop, ITERS, [&]() {
            naive_lmhead_topk(cublas, d_query, d_W, d, V, k, 1.0f,
                              naive_r, d_ws, d_st, sb + ex);
        });

        // Time fused (precomputation not counted, it's a one-time cost)
        float fused_ms = time_kernel_ms(ev_start, ev_stop, ITERS, [&]() {
            fused_topk_decode(cublas, state, d_query, fused_cfg, fused_r, nullptr);
        });

        float speedup = naive_ms / fused_ms;

        // Measure skip rate
        float skip_rate = measure_skip_rate(cublas, state, d_query, k);

        // Memory: naive allocates V-element fp32 workspace
        float naive_mem_mb = (float)(V * sizeof(float)) / (1024.0f * 1024.0f);

        // Throughput (fused, tokens/sec)
        float fused_tok_per_sec = 1000.0f / fused_ms;

        // Effective bandwidth: naive reads d*V fp16 weights + V fp32 for sort
        float naive_bw_gb = (float)((long long)d * V * sizeof(__half) +
                                    (long long)V * sizeof(float)) / 1e9f;
        float naive_bw_gbps = naive_bw_gb / (naive_ms * 1e-3f);

        printf("%-40s %10.3f %10.3f %10.2fx %9.1f%% %12.1f %14.0f\n",
               cfg.name, naive_ms, fused_ms, speedup,
               skip_rate * 100.0f, naive_mem_mb, fused_tok_per_sec);

        free_lmhead_state(state);
        CHECK_CUDA(cudaFree(d_W));      CHECK_CUDA(cudaFree(d_query));
        CHECK_CUDA(cudaFree(d_ws));     CHECK_CUDA(cudaFree(d_st));
        CHECK_CUDA(cudaFree(d_ns));     CHECK_CUDA(cudaFree(d_ni));
        CHECK_CUDA(cudaFree(d_fs));     CHECK_CUDA(cudaFree(d_fi));
    }

    printf("\n=== Peak memory: fused avoids %.1f MB V-element logit buffer for V=128256 ===\n",
           128256.0f * sizeof(float) / (1024.0f * 1024.0f));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUBLAS(cublasDestroy(cublas));
    return 0;
}
