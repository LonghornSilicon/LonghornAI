cmake_minimum_required(VERSION 3.20)

project(LonghornAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LONGHORN_BUILD_TESTS "Build the GoogleTest suite" ON)
option(LONGHORN_ENABLE_OPENMP "Use OpenMP to parallelize kernels (optional)" OFF)

# --- kernel library ----------------------------------------------------------

add_library(longhorn_kernels STATIC
    src/kernels/gemm.cpp
    src/kernels/normalization.cpp
    src/kernels/activation.cpp
    src/kernels/softmax.cpp
    src/kernels/reduction.cpp
    src/kernels/rope.cpp
    src/kernels/embedding.cpp
    src/kernels/attention.cpp
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

# --- tests -------------------------------------------------------------------

if(LONGHORN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
