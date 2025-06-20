set(FPC_DIR ${CMAKE_SOURCE_DIR}/fpc)

set(FPC_COMMON_SRC
    ${FPC_DIR}/fpc-cpu.cpp
    ${FPC_DIR}/fpc.cpp
)

if(BUILD_WITH_CUDA)
    file(GLOB FLAWED_CUDA "${CMAKE_SOURCE_DIR}/fpc/*.cu")
    add_library(fpc STATIC ${FPC_COMMON_SRC} ${FLAWED_CUDA})    
    set_target_properties(fpc PROPERTIES CUDA_ARCHITECTURES 70)

elseif(BUILD_WITH_HIP)
    file(GLOB FLAWED_CUDA "${CMAKE_SOURCE_DIR}/fpc/*.hip")
    add_library(fpc STATIC ${FPC_COMMON_SRC} ${FLAWED_CUDA})

endif()

target_include_directories(fpc PUBLIC ${CMAKE_SOURCE_DIR}/fpc)
