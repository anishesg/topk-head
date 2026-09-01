#pragma once
#include <cuda_fp16.h>
#include <float.h>

// A compile-time-sized min-heap stored entirely in registers.
// Tracks the running top-K (score, token_id) pairs during vocabulary scan.
// All operations are O(K) with no memory traffic beyond register access.
//
// Template parameter K must be a small compile-time constant (e.g. 1-256).
// Larger K increases register pressure and may reduce occupancy.
template <int K>
struct TopKHeap {
    float scores[K];
    int   indices[K];
    int   size;   // number of valid entries (0 <= size <= K)

    __device__ __forceinline__ void init() {
        size = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            scores[i]  = -FLT_MAX;
            indices[i] = -1;
        }
    }

    // Return the current minimum score (pruning threshold).
    // When heap is not yet full, returns -FLT_MAX so all candidates are inserted.
    __device__ __forceinline__ float peek_min() const {
        float mn = scores[0];
        #pragma unroll
        for (int i = 1; i < K; i++)
            if (scores[i] < mn) mn = scores[i];
        return mn;
    }

    // Return index of current minimum element.
    __device__ __forceinline__ int min_idx() const {
        int mi = 0;
        #pragma unroll
        for (int i = 1; i < K; i++)
            if (scores[i] < scores[mi]) mi = i;
        return mi;
    }

    // Insert (score, token_id) if score exceeds the current minimum.
    // When the heap is not yet full, always inserts.
    __device__ __forceinline__ void insert(float score, int token_id) {
        if (size < K) {
            scores[size]  = score;
            indices[size] = token_id;
            size++;
        } else {
            int mi = min_idx();
            if (score > scores[mi]) {
                scores[mi]  = score;
                indices[mi] = token_id;
            }
        }
    }

    // Extract all K entries sorted by descending score into out_scores/out_indices.
    // Non-destructive on the heap contents.
    __device__ __forceinline__ void extract_sorted(
        float* out_scores,
        int*   out_indices) const
    {
        // Copy into local arrays and do selection sort (K is small, O(K^2) is fine)
        float  tmp_s[K];
        int    tmp_i[K];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            tmp_s[i] = scores[i];
            tmp_i[i] = indices[i];
        }
        #pragma unroll
        for (int i = 0; i < K; i++) {
            int best = i;
            #pragma unroll
            for (int j = i + 1; j < K; j++)
                if (tmp_s[j] > tmp_s[best]) best = j;
            out_scores[i]  = tmp_s[best];
            out_indices[i] = tmp_i[best];
            tmp_s[best]    = -FLT_MAX;
        }
    }

    // Whether the heap is fully saturated (size == K).
    // Pruning only becomes effective after saturation.
    __device__ __forceinline__ bool full() const { return size == K; }
};
