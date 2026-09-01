# topk-head

Fused LM-head top-k decode: tiled vocabulary projection with register-heap online selection and Cauchy-Schwarz tile pruning.

## Problem

During autoregressive decoding, the LM-head projects a hidden state vector `h` of dimension `d` against a weight matrix `W` of shape `(d, V)` where `V` is the vocabulary size. For modern models:

- Llama-3: V=128256, d=4096 or d=8192
- GPT-4-class: V=100256, d=8192
- Mistral: V=32000, d=4096

Computing `logits = h @ W` materializes a V-element fp16 buffer per decode step. At V=128256 this is 256KB per token, plus a subsequent O(V log V) sort or O(V) top-k selection. Every major serving framework (vLLM, TRT-LLM, FasterTransformer) executes this as separate operations: dense matmul via cuBLAS followed by cub::DeviceRadixSort or cub::DeviceSelect.

## Existing Approaches

**Dense matmul + radix sort (vLLM, TRT-LLM)**: Standard approach. Full V-element logit buffer must be allocated. Memory bandwidth bound for small batch sizes typical in interactive decode. Separate kernel launches prevent register-level fusion.

**Approximate MIPS via LSH/PQ**: Locality-sensitive hashing or product quantization on W columns approximates the nearest-neighbor search. Precision loss is unacceptable for greedy decode (top-1 must be exact) and hurts sampling quality. Index structures require preprocessing and add complexity.

**Beam search with early exit**: Partial sort strategies (partial heapsort, introselect) reduce sort cost but still require full dot product computation before any pruning.

## This Approach

Three ideas compose to eliminate both the V-sized logit buffer and the separate sort:

### 1. Register-Resident Min-Heap

A template struct `TopKHeap<int K>` holds K `(score, token_id)` pairs entirely in registers. No shared or global memory for the heap. During vocabulary scan, each new dot product is compared against the current minimum; if larger, it replaces the minimum. This gives online top-k selection in a single pass with O(1) amortized cost per token and zero extra memory allocation.

### 2. Cauchy-Schwarz Tile Pruning

For a query vector `q` and weight tile `T` containing column vectors `w_i`:

```
dot(q, w_i) <= ||q|| * ||w_i||   (Cauchy-Schwarz)
```

The maximum possible dot product for any column in a tile is bounded by `||q|| * max_i(||w_i||)`. If this bound is less than the current minimum score in the heap (the k-th best score seen so far), the entire tile can be skipped without computing any dot products. The pruning rate improves as the heap fills with strong candidates.

### 3. Vocabulary Reordering by Descending Column Norm

LM-head weight matrices have heterogeneous column norms: a small fraction of columns (corresponding to high-frequency tokens) have significantly higher L2 norms than the tail. By sorting vocabulary columns in descending norm order and processing them first, the heap saturates quickly with high-scoring candidates. This maximizes the Cauchy-Schwarz bound tightness early, enabling more tile skips over the long tail of low-norm columns.

### Tile Skip Rate Analysis

Under empirical LM-head column norm distributions (Zipf-like, consistent with token frequency distributions), column norms concentrate: the top 5-10% of columns by norm account for ~50% of the norm mass. After processing the first 10% of reordered tiles, the k-th heap entry is typically within 15-20% of the global top score. For k=50 and V=128256 at typical hidden-state magnitudes, expected tile skip rate is 60-80% depending on hidden-state entropy. For greedy decode (k=1), skip rates exceed 90% on typical token distributions.

## Architecture

```
src/
  tile_metadata.cuh / .cu    -- VocabTileMeta struct, column norm kernel
  vocab_reorder.cuh / .cu    -- sort columns by descending norm, produce permutation
  register_heap.cuh          -- TopKHeap<K> device template, all in registers
  naive_lmhead.cuh / .cu     -- reference: tiled matvec + cub radix sort
  fused_topk_proj.cuh / .cu  -- fused kernel: tiled projection + heap + CS pruning
  fused_topk_proj_batched.cuh / .cu  -- batched variant, one block per token
tests/
  test_correctness.cu        -- sweeps vocab, dim, k, temperature configs
benchmarks/
  bench_latency.cu           -- CUDA-event latency, tile skip rate, bandwidth
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/test_correctness
./build/bench_latency
```

Requires CUDA 11.8+ and sm_80+ (Ampere or newer).
