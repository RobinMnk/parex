add_library(ParexLib INTERFACE)
target_include_directories(ParexLib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_include_directories(ParexLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})