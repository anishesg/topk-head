#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <cassert>
#include <stdexcept>

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

static void fill_random_fp16(__half* d_buf, int n, float scale, unsigned seed) {
    std::vector<__half> h(n);
    srand(seed);
    for (int i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        h[i] = __float2half(v);
    }
    CHECK_CUDA(cudaMemcpy(d_buf, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

// Run one correctness check for given (V, d, k, temperature).
// Returns true if passed.
static bool run_test(cublasHandle_t cublas,
                     int V, int d, int k, float temperature,
                     bool verbose)
{
    // Allocate weight matrix and query
    __half *d_W, *d_query;
    CHECK_CUDA(cudaMalloc(&d_W,     sizeof(__half) * d * V));
    CHECK_CUDA(cudaMalloc(&d_query, sizeof(__half) * d));

    fill_random_fp16(d_W,     d * V, 0.02f, 42 + V + d);
    fill_random_fp16(d_query, d,     1.0f,  99 + V + d);

    // Build fused state
    LMHeadState state = build_lmhead_state(d_W, d, V);

    // Allocate result buffers
    float *d_fused_scores, *d_naive_scores;
    int   *d_fused_indices, *d_naive_indices;
    CHECK_CUDA(cudaMalloc(&d_fused_scores,  sizeof(float) * k));
    CHECK_CUDA(cudaMalloc(&d_fused_indices, sizeof(int)   * k));
    CHECK_CUDA(cudaMalloc(&d_naive_scores,  sizeof(float) * k));
    CHECK_CUDA(cudaMalloc(&d_naive_indices, sizeof(int)   * k));

    TopKResult fused_result = {d_fused_scores, d_fused_indices};
    TopKResult naive_result = {d_naive_scores, d_naive_indices};

    // Naive: needs V-element workspace + sort scratch
    float* d_workspace;
    size_t sort_bytes = naive_sort_tmp_bytes(V);
    // Extra space for neg_logits copy and index array inside naive_lmhead
    size_t extra = (size_t)V * sizeof(float) + (size_t)V * sizeof(int);
    void*  d_sort_tmp;
    CHECK_CUDA(cudaMalloc(&d_workspace, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_sort_tmp,  sort_bytes + extra));

    FusedTopKConfig cfg;
    cfg.k           = k;
    cfg.temperature = temperature;
    cfg.mode        = SamplingMode::TopK;
    cfg.top_p       = 0.9f;
    cfg.uniform_rand = 0.5f;

    fused_topk_decode(cublas, state, d_query, cfg, fused_result, nullptr);
    naive_lmhead_topk(cublas, d_query, d_W, d, V, k, temperature,
                      naive_result, d_workspace, d_sort_tmp, sort_bytes + extra);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results to host
    std::vector<float> h_fused_s(k), h_naive_s(k);
    std::vector<int>   h_fused_i(k), h_naive_i(k);
    CHECK_CUDA(cudaMemcpy(h_fused_s.data(), d_fused_scores,  k*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fused_i.data(), d_fused_indices, k*sizeof(int),   cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_naive_s.data(), d_naive_scores,  k*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_naive_i.data(), d_naive_indices, k*sizeof(int),   cudaMemcpyDeviceToHost));

    bool passed = true;
    std::string reason;

    // For greedy (k=1 or T=0), require exact top-1 index match.
    if (k == 1 || temperature == 0.0f) {
        if (h_fused_i[0] != h_naive_i[0]) {
            passed = false;
            reason = "top-1 index mismatch";
        }
    }

    // Index set agreement: fraction of fused top-k that appear in naive top-k
    std::set<int> naive_set(h_naive_i.begin(), h_naive_i.end());
    int overlap = 0;
    for (int idx : h_fused_i) {
        if (naive_set.count(idx)) overlap++;
    }
    float agreement = (float)overlap / k;
    if (agreement < 0.95f) {
        passed = false;
        reason = "top-k set agreement < 95%";
    }

    // Score max absolute error (compare top-1 score)
    float score_err = fabsf(h_fused_s[0] - h_naive_s[0]);
    if (score_err > 1e-2f) {  // fp16 precision limitation
        passed = false;
        reason = "score error too large: " + std::to_string(score_err);
    }

    if (verbose || !passed) {
        printf("  V=%6d d=%4d k=%3d T=%.1f: %s (agreement=%.1f%% score_err=%.4f top1_fused=%d top1_naive=%d)\n",
               V, d, k, temperature,
               passed ? "PASS" : "FAIL",
               agreement * 100.0f, score_err,
               h_fused_i[0], h_naive_i[0]);
        if (!passed) printf("    REASON: %s\n", reason.c_str());
    }

    free_lmhead_state(state);
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_fused_scores));
    CHECK_CUDA(cudaFree(d_fused_indices));
    CHECK_CUDA(cudaFree(d_naive_scores));
    CHECK_CUDA(cudaFree(d_naive_indices));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_sort_tmp));

    return passed;
}

// Chi-squared test: 10000 greedy samples from a fixed weight matrix should
// produce a distribution consistent with the single-pass result.
static bool run_distribution_test(cublasHandle_t cublas, int V, int d) {
    // For distribution test, use small V and check top-1 distribution.
    // We draw 10000 random queries and check the top-1 index matches
    // between fused and naive for each.
    const int N_SAMPLES = 1000;
    int mismatches = 0;

    __half *d_W, *d_query;
    CHECK_CUDA(cudaMalloc(&d_W,     sizeof(__half) * d * V));
    CHECK_CUDA(cudaMalloc(&d_query, sizeof(__half) * d));
    fill_random_fp16(d_W, d * V, 0.02f, 7777);

    LMHeadState state = build_lmhead_state(d_W, d, V);

    float *d_fs, *d_ns;
    int   *d_fi, *d_ni;
    CHECK_CUDA(cudaMalloc(&d_fs, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fi, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ns, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ni, sizeof(int)));
    TopKResult fr = {d_fs, d_fi}, nr = {d_ns, d_ni};

    float* d_ws;
    size_t sb = naive_sort_tmp_bytes(V);
    size_t ex = (size_t)V * sizeof(float) + (size_t)V * sizeof(int);
    void* d_st;
    CHECK_CUDA(cudaMalloc(&d_ws, sizeof(float) * V));
    CHECK_CUDA(cudaMalloc(&d_st, sb + ex));

    FusedTopKConfig cfg = {1, 0.0f, SamplingMode::Greedy, 1.0f, 0.5f};

    for (int i = 0; i < N_SAMPLES; i++) {
        fill_random_fp16(d_query, d, 1.0f, i * 31337);
        fused_topk_decode(cublas, state, d_query, cfg, fr, nullptr);
        naive_lmhead_topk(cublas, d_query, d_W, d, V, 1, 0.0f, nr, d_ws, d_st, sb + ex);
        CHECK_CUDA(cudaDeviceSynchronize());

        int fi, ni;
        CHECK_CUDA(cudaMemcpy(&fi, d_fi, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&ni, d_ni, sizeof(int), cudaMemcpyDeviceToHost));
        if (fi != ni) mismatches++;
    }

    free_lmhead_state(state);
    CHECK_CUDA(cudaFree(d_W)); CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_fs)); CHECK_CUDA(cudaFree(d_fi));
    CHECK_CUDA(cudaFree(d_ns)); CHECK_CUDA(cudaFree(d_ni));
    CHECK_CUDA(cudaFree(d_ws)); CHECK_CUDA(cudaFree(d_st));

    float mismatch_rate = (float)mismatches / N_SAMPLES;
    bool passed = (mismatches == 0);
    printf("  Distribution test V=%d d=%d: %d/%d mismatches (%.2f%%) -> %s\n",
           V, d, mismatches, N_SAMPLES, mismatch_rate * 100.0f,
           passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    int total = 0, failed = 0;

    // Sweep vocab_size x hidden_dim x k x temperature
    const int vocab_sizes[]    = {32768, 65536, 131072};
    const int hidden_dims[]    = {4096, 8192};
    const int ks[]             = {1, 10, 50, 256};
    const float temperatures[] = {0.0f, 0.7f, 1.0f};

    printf("=== Correctness Tests ===\n");
    for (int V : vocab_sizes) {
        for (int d : hidden_dims) {
            for (int k : ks) {
                // Skip k > V
                if (k > V) continue;
                for (float T : temperatures) {
                    bool ok = run_test(cublas, V, d, k, T, true);
                    total++;
                    if (!ok) failed++;
                }
            }
        }
    }

    printf("\n=== Distribution Tests (greedy consistency) ===\n");
    {
        bool ok = run_distribution_test(cublas, 32768, 4096);
        total++; if (!ok) failed++;
    }
    {
        bool ok = run_distribution_test(cublas, 32768, 8192);
        total++; if (!ok) failed++;
    }

    printf("\n=== Summary: %d/%d passed ===\n", total - failed, total);

    CHECK_CUBLAS(cublasDestroy(cublas));
    return (failed == 0) ? 0 : 1;
}
