cmake_minimum_required(VERSION 3.18)
project(parex LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(parex main.cu timer.h randomWalk.cu devGraph.cu)

# Set CUDA path manually if needed
find_package(CUDAToolkit REQUIRED)

# Add CUDA include directories for host compiler
target_include_directories(parex PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(parex PRIVATE CUDA::cudart)

#target_compile_options(parex PRIVATE -O3)