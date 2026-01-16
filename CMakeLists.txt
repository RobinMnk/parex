cmake_minimum_required(VERSION 3.18)
project(Parex LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(lib)
add_subdirectory(cuda)

# Main app
add_executable(Parex main.cpp)
target_link_libraries(Parex PRIVATE ParexLib ParexCuda)