include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

function(configure_test_target target_name source_file)
    add_executable(${target_name} ${source_file})

    target_link_libraries(${target_name}
            PRIVATE
            ParexLib
            ParexCuda
            GTest::gtest_main
    )

    if (MSVC)
        target_compile_definitions(${target_name} PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)
    endif()

    include(GoogleTest)
    gtest_discover_tests(${target_name})
endfunction()

configure_test_target(CudaTest cudaTest.cpp)
configure_test_target(clusterTest clusterTest.cpp)
