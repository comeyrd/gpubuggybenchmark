add_library(hip_common STATIC
    hip-utils.hip
)
target_link_libraries(hip_common PRIVATE argparse)
target_include_directories(hip_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hip_common PRIVATE ${CMAKE_SOURCE_DIR}/include)
