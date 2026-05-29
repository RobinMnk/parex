add_library(ParexLib INTERFACE)
target_include_directories(ParexLib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_include_directories(ParexLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

#target_compile_options(ParexLib INTERFACE
#        $<$<COMPILE_LANGUAGE:CXX>:
#            $<$<CONFIG:Debug>:/W4>
#            $<$<CONFIG:Release>:
#                /O2
#                /Ob2
#                /GL
#                /fp:fast
#                /arch:AVX2
#                /permissive-
#            >
#        >
#)

target_compile_options(ParexLib INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Release>:
        -O3
        --use_fast_math
        -Xptxas=-O3,-v
        -lineinfo
        --extra-device-vectorization
        >
        >
)

# Enable Device Link Time Optimization if using CUDA 11.0+
set_target_properties(ParexLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ParexLib PROPERTIES INTER_PROCEEDURAL_OPTIMIZATION_RELEASE TRUE)
