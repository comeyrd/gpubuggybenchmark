set(ACCURACY_DIR ${CMAKE_SOURCE_DIR}/acc)

set(ACCURACY_COMMON_SRC
    ${ACCURACY_DIR}/accuracy.cpp
)

set(ACCURACY_LIB_TO_LINK argparse)
set(ACCURACY_DIR_TO_INCLUDE ${ACCURACY_DIR} ${ACCURACY_DIR}/flawed/include ${CMAKE_SOURCE_DIR}/include)

if(BUILD_WITH_CUDA)
    file(GLOB FLAWED_CUDA "${ACCURACY_DIR}/*.cu" "${ACCURACY_DIR}/flawed/*.cu" "${ACCURACY_DIR}/cuda/*.cu" )
    add_library(accuracy STATIC ${ACCURACY_COMMON_SRC} ${FLAWED_CUDA})    
    set_target_properties(accuracy PROPERTIES CUDA_ARCHITECTURES 70)
    target_link_libraries(accuracy ${ACCURACY_LIB_TO_LINK} cuda_common)
    target_include_directories(accuracy PUBLIC ${ACCURACY_DIR_TO_INCLUDE} ${ACCURACY_DIR}/cuda)

elseif(BUILD_WITH_HIP)
    file(GLOB FLAWED_HIP "${ACCURACY_DIR}/*.hip" "${ACCURACY_DIR}/flawed/*.hip")
    add_library(accuracy STATIC ${ACCURACY_COMMON_SRC} ${FLAWED_HIP})
    target_link_libraries(accuracy ${ACCURACY_LIB_TO_LINK})
    target_include_directories(accuracy PUBLIC ${ACCURACY_DIR_TO_INCLUDE} ${ACCURACY_DIR}/hip)

elseif(BUILD_WITH_METAL)
    # Find all .metal shader files
    file(GLOB FLAWED_METAL "${ACCURACY_DIR}/*.metal" "${ACCURACY_DIR}/flawed/*.metal")

    # Find all .metalcpp files (C++ source using Metal C++ API)
    file(GLOB METALCPP_SOURCES "${ACCURACY_DIR}/*-metal.cpp" "${ACCURACY_DIR}/flawed/*-metal.cpp")

    # Compile each .metal file to .air and then .metallib
    set(METAL_LIBS "")
    foreach(METAL_FILE IN LISTS FLAWED_METAL)
        get_filename_component(METAL_NAME ${METAL_FILE} NAME_WE)
        set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/${METAL_NAME}.air")
        set(METAL_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/${METAL_NAME}.metallib")

        add_custom_command(
            OUTPUT ${METAL_AIR}
            COMMAND ${METAL_COMPILER} -c ${METAL_FILE} -o ${METAL_AIR} -std=macos-metal2.4
            DEPENDS ${METAL_FILE}
            COMMENT "Compiling Metal shader ${METAL_FILE} to AIR"
        )
        add_custom_command(
            OUTPUT ${METAL_METALLIB}
            COMMAND xcrun metallib ${METAL_AIR} -o ${METAL_METALLIB}
            DEPENDS ${METAL_AIR}
            COMMENT "Creating Metal library ${METAL_METALLIB}"
        )

        list(APPEND METAL_LIBS ${METAL_METALLIB})
    endforeach()

    # Add your CPU sources plus .metalcpp files to a library
    add_library(accuracy STATIC ${ACCURACY_COMMON_SRC} ${METALCPP_SOURCES})
    
    set_target_properties(accuracy PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    target_include_directories(accuracy PUBLIC ${ACCURACY_DIR} ${ACCURACY_DIR}/flawed ${METAL_CPP_PATH})
    
    # Ensure Metal kernel libs are built before accuracy target
    add_custom_target(metal_kernels ALL DEPENDS ${METAL_LIBS})
    add_dependencies(accuracy metal_kernels)
    target_link_libraries(accuracy PUBLIC "-framework Metal" ${ACCURACY_LIB_TO_LINK})

endif()

