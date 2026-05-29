set(CMAKE_CUDA_ARCHITECTURES 80 86)

file(GLOB ALL_CUDA_FILES CONFIGURE_DEPENDS *.cu *.cuh *.h)

add_library(ParexCuda STATIC)
target_sources(ParexCuda PRIVATE ${ALL_CUDA_FILES})

target_link_libraries(ParexCuda PUBLIC ParexLib)

target_compile_options(ParexCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(ParexCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_options(ParexCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        --use_fast_math                # Fast math intrinsic functions
        -Xptxas=-O3,-v                 # Maximize PTX optimization and show register usage
        -lineinfo                      # Adds line info for Nsight Compute profiling without slowing down code
        >
        $<$<COMPILE_LANGUAGE:CXX>:
        /O2                            # MSVC Maximize Speed
        /Oi                            # Enable Intrinsic Functions
        /Ot                            # Favor fast code
        >
)