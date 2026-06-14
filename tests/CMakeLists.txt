include(FetchContent)

# Prefer a system GoogleTest if one is installed (lets the suite build in
# environments without internet, e.g. CI containers). Fall back to fetching
# the upstream release otherwise.
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(longhorn_tests
    test_dtypes.cpp
    test_gemm.cpp
    test_normalization.cpp
    test_activation.cpp
    test_softmax.cpp
    test_reduction.cpp
    test_rope.cpp
    test_embedding.cpp
    test_attention.cpp
    test_topk.cpp
    test_transform.cpp
    test_attention_ext.cpp
    test_paged_attention.cpp
    test_cache_manager.cpp
    test_prefix_cache.cpp
    test_scheduler.cpp
    test_kvcache_quant.cpp
    test_quant.cpp
    test_gemm_quant.cpp
    numerical/test_ulp_sweep.cpp
)

target_link_libraries(longhorn_tests PRIVATE
    longhorn_kernels
    GTest::gtest_main
)

target_include_directories(longhorn_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(longhorn_tests)
