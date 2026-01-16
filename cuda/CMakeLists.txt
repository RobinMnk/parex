set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB ALL_CUDA_FILES CONFIGURE_DEPENDS *.cu *.cuh *.h)

add_library(ParexCuda STATIC)
target_sources(ParexCuda PRIVATE ${ALL_CUDA_FILES})

target_link_libraries(ParexCuda PUBLIC ParexLib)