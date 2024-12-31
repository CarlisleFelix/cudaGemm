cmake_minimum_required(VERSION 3.30)
project(learn1 CXX C CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR}/header)
include_directories(${PROJECT_SOURCE_DIR}/kernel)

add_executable(main main.cu utils.cu)
target_link_libraries(main cublas)
set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(test test.cu utils.cu)
target_link_libraries(test cublas)
set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
