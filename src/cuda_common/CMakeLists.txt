add_library(cuda_common STATIC
    cuda-utils.cu
    blocking_kernel.cu
)
set_target_properties(cuda_common PROPERTIES CUDA_ARCHITECTURES 70)
target_link_libraries(cuda_common PRIVATE argparse)
target_include_directories(cuda_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cuda_common PRIVATE ${CMAKE_SOURCE_DIR}/include)
