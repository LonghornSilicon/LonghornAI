cmake_minimum_required(VERSION 3.20)

project(LonghornAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LONGHORN_BUILD_TESTS "Build the GoogleTest suite" ON)
option(LONGHORN_BUILD_BENCH "Build the benchmark harness" ON)
option(LONGHORN_BUILD_SIM "Build the Phase 9 silicon simulator" ON)
option(LONGHORN_ENABLE_OPENMP "Use OpenMP to parallelize kernels (optional)" OFF)

# --- kernel library ----------------------------------------------------------

add_library(longhorn_kernels STATIC
    src/core/profiler.cpp
    src/kernels/gemm.cpp
    src/kernels/normalization.cpp
    src/kernels/activation.cpp
    src/kernels/softmax.cpp
    src/kernels/reduction.cpp
    src/kernels/rope.cpp
    src/kernels/positional.cpp
    src/kernels/embedding.cpp
    src/kernels/transform.cpp
    src/kernels/attention.cpp
    src/kernels/kvcache.cpp
    src/kernels/kvcache_quant.cpp
    src/kernels/paged_attention.cpp
    src/kernels/quant.cpp
    src/kernels/gemm_quant.cpp
    src/kernels/moe.cpp
    src/kernels/sampling.cpp
    src/kernels/speculative.cpp
    src/kernels/ssm.cpp
    src/kernels/rwkv.cpp
    src/kernels/linear_attn.cpp
    src/kernels/conv1d.cpp
    src/runtime/cache_manager.cpp
    src/runtime/prefix_cache.cpp
    src/runtime/scheduler.cpp
)

target_include_directories(longhorn_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(longhorn_kernels PUBLIC cxx_std_17)

# Per-compiler warning flags (kept portable across MSVC / GCC / Clang).
if(MSVC)
    target_compile_options(longhorn_kernels PRIVATE /W3)
else()
    target_compile_options(longhorn_kernels PRIVATE -Wall -Wextra)
endif()

# Optional, never-required multithreading. The single-threaded path is always
# correct and builds everywhere without this.
if(LONGHORN_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(longhorn_kernels PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(longhorn_kernels PUBLIC LONGHORN_ENABLE_OPENMP)
    else()
        message(WARNING "LONGHORN_ENABLE_OPENMP=ON but OpenMP was not found; "
                        "building the single-threaded path.")
    endif()
endif()

# --- benchmark harness -------------------------------------------------------

if(LONGHORN_BUILD_BENCH)
    add_executable(longhorn_bench bench/bench_main.cpp)
    target_link_libraries(longhorn_bench PRIVATE longhorn_kernels)
    target_include_directories(longhorn_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/bench
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    if(MSVC)
        target_compile_options(longhorn_bench PRIVATE /W3)
    else()
        target_compile_options(longhorn_bench PRIVATE -Wall -Wextra)
    endif()
endif()

# --- Phase 9 silicon simulator ----------------------------------------------

if(LONGHORN_BUILD_SIM)
    add_library(longhorn_perf_model STATIC
        tools/perf_model/perf_model.cpp
    )
    target_include_directories(longhorn_perf_model PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/perf_model
    )
    target_compile_features(longhorn_perf_model PUBLIC cxx_std_17)
    if(MSVC)
        target_compile_options(longhorn_perf_model PRIVATE /W3)
    else()
        target_compile_options(longhorn_perf_model PRIVATE -Wall -Wextra)
    endif()

    add_executable(silicon_sim tools/perf_model/silicon_sim_main.cpp)
    target_link_libraries(silicon_sim PRIVATE longhorn_perf_model)
endif()

# --- tests -------------------------------------------------------------------

if(LONGHORN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
