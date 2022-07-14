cmake_minimum_required(VERSION 3.16)
project(cuda_ws CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(/usr/local/cuda/samples/common/inc)
include_directories(/usr/local/cuda-10.2/include)
include_directories(/usr/local/cuda-10.2/samples/common/inc)

add_executable(cuda_ws
        # thread_idx_1d.cu
        thread_idx_2d.cu
        # thread_idx.cu
        )

set_target_properties(
        cuda_ws
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
