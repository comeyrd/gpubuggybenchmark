set(FPC_DIR ${CMAKE_SOURCE_DIR}/fpc)

set(FPC_COMMON_SRC
    ${FPC_DIR}/fpc-cpu.cpp
    ${FPC_DIR}/fpc.cpp
)

if(BUILD_WITH_CUDA)
    file(GLOB FLAWED_CUDA "${FPC_DIR}/*.cu" "${FPC_DIR}/flawed/*.cu")
    add_library(fpc STATIC ${FPC_COMMON_SRC} ${FLAWED_CUDA})    
    set_target_properties(fpc PROPERTIES CUDA_ARCHITECTURES 70)

elseif(BUILD_WITH_HIP)
    file(GLOB FLAWED_CUDA "${FPC_DIR}/*.hip" "${FPC_DIR}/flawed/*.hip")
    add_library(fpc STATIC ${FPC_COMMON_SRC} ${FLAWED_CUDA})

endif()

target_include_directories(fpc PUBLIC ${FPC_DIR} ${FPC_DIR}/flawed)
