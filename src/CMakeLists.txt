cmake_minimum_required(VERSION 3.15)

project(gpubuggy LANGUAGES CXX)
##TODO setup Debug & Release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -g")
include(CheckLanguage)
add_compile_options(-Wall -O3 -fopenmp -funroll-loops)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Using CUDA toolkit: ${CMAKE_CUDA_COMPILER}")
    set(BUILD_WITH_CUDA TRUE)
    set(BUILD_WITH_HIP FALSE)
    enable_language(CUDA)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 ")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
    ##TODO setup Debug & Release
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep --keep-dir ${CMAKE_BINARY_DIR}/ptx")
    add_subdirectory(cuda_common)
else()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        set(BUILD_WITH_HIP TRUE)
        set(BUILD_WITH_CUDA FALSE)
        enable_language(HIP)
        message(STATUS "Building with HIP")
        add_subdirectory(hip_common)
    else()
        if(APPLE)
            set(CMAKE_CXX_STANDARD 17)
            set(CMAKE_CXX_STANDARD_REQUIRED ON)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -objective-c++ -fobjc-arc -fmodules ")
            execute_process(
                COMMAND xcrun --find metal
                OUTPUT_VARIABLE METAL_COMPILER_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )

            if(METAL_COMPILER_PATH)
                message(STATUS "Found Metal compiler at ${METAL_COMPILER_PATH}")
                set(BUILD_WITH_METAL TRUE)
                set(METAL_COMPILER ${METAL_COMPILER_PATH})
            else()
                message(FATAL_ERROR "Metal compiler not found via xcrun, cannot compile")
            endif()
            if(NOT DEFINED ENV{METAL_CPP_PATH})
            message(FATAL_ERROR "Metal compiler found but metal-cpp header not in METAL_CPP_PATH env variable")
            else()
                set(METAL_CPP_PATH $ENV{METAL_CPP_PATH})
            endif()
        else()
            message(FATAL_ERROR "CUDA and HIP compilers not found, and not on Apple platform, can't compile")
        endif()
    endif()
endif()
#add_subdirectory(fpc)
#add_subdirectory(acc)
add_subdirectory(bilateral)
add_executable(gpubuggy main.cpp)

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        v3.2
)
FetchContent_MakeAvailable(argparse)

set(KERNELS_TO_LINK bilateral)
#set(KERNELS_TO_LINK accuracy bilateral fpc)

find_package(OpenMP REQUIRED)

if(APPLE)
    add_dependencies(gpubuggy fpc)
    target_link_libraries(gpubuggy PRIVATE argparse "-Wl,-all_load" ${KERNELS_TO_LINK})
elseif(UNIX)
    target_link_libraries(gpubuggy PRIVATE argparse "-Wl,--whole-archive" ${KERNELS_TO_LINK} "-Wl,--no-whole-archive" OpenMP::OpenMP_CXX)
endif()

target_include_directories(gpubuggy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)