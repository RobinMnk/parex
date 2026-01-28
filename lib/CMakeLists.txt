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