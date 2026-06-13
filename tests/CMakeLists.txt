include(FetchContent)

# GoogleTest via FetchContent so the suite builds on any OS without a system
# install. Pinned to a release tag for reproducibility.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

# Use the shared CRT on Windows to match the rest of the build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
