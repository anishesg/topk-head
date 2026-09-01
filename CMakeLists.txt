cmake_minimum_required(VERSION 3.20)
project(topk_head LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Static library
add_library(topk_head STATIC
    src/tile_metadata.cu
    src/vocab_reorder.cu
    src/naive_lmhead.cu
    src/fused_topk_proj.cu
    src/fused_topk_proj_batched.cu
)

target_include_directories(topk_head PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(topk_head PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

set_target_properties(topk_head PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(topk_head PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xptxas=-v
        --extended-lambda
    >
)

# Tests
add_executable(test_correctness tests/test_correctness.cu)
target_include_directories(test_correctness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_correctness PRIVATE topk_head CUDA::cudart CUDA::cublas)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_correctness PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Benchmarks
add_executable(bench_latency benchmarks/bench_latency.cu)
target_include_directories(bench_latency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(bench_latency PRIVATE topk_head CUDA::cudart CUDA::cublas)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_latency PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

add_executable(test_batched tests/test_batched.cu)
target_include_directories(test_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_batched PRIVATE topk_head CUDA::cudart CUDA::cublas)
set_target_properties(test_batched PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_batched PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

add_executable(bench_batched benchmarks/bench_batched.cu)
target_include_directories(bench_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(bench_batched PRIVATE topk_head CUDA::cudart CUDA::cublas)
set_target_properties(bench_batched PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(bench_batched PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
add_test(NAME batched     COMMAND test_batched)
