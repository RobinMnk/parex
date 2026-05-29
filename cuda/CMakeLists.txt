file(GLOB ALL_CUDA_FILES CONFIGURE_DEPENDS *.cu *.cuh *.h)

add_library(ParexCuda STATIC)
target_sources(ParexCuda PRIVATE ${ALL_CUDA_FILES})
set_target_properties(ParexCuda PROPERTIES CUDA_ARCHITECTURES "80;86")

target_link_libraries(ParexCuda PUBLIC ParexLib)

target_compile_options(ParexCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        --use_fast_math
        -Xptxas=-O3,-v
        -lineinfo
        >
        $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        >
)
