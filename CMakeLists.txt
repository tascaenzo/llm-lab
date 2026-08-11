cmake_minimum_required(VERSION 3.24)

project(
    llm_lab
    VERSION 0.1.0
    DESCRIPTION "A from-scratch LLM learning project written in C"
    LANGUAGES C)

include(CTest)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(LLM_LAB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(LLM_LAB_BUILD_BENCHMARKS "Build runtime benchmark executables" OFF)

include(cmake/CompilerWarnings.cmake)
find_package(Threads REQUIRED)

add_library(runtime STATIC
    src/runtime/backend.c
    src/runtime/backends/cpu/cpu_backend.c
    src/runtime/backends/cpu/cpu_executor.c
    src/runtime/backends/cpu/cpu_features.c
    src/runtime/backends/cpu/cpu_kernels.c
    src/runtime/backends/cpu/cpu_operations.c
    src/runtime/backends/cpu/cpu_simd.c
    src/runtime/backends/cpu/cpu_threads.c
    src/runtime/memory.c
    src/runtime/operations.c
    src/runtime/tensor.c)

if(APPLE)
    enable_language(OBJC)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)

    set(LLM_METAL_KERNEL_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/metal/kernels/runtime.metal)
    set(LLM_METAL_SIMDGROUP_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/metal/kernels/matmul_simdgroup.metal)
    set(LLM_METAL_GENERATED_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/metal)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                 ${LLM_METAL_KERNEL_SOURCE} ${LLM_METAL_SIMDGROUP_SOURCE})
    file(READ ${LLM_METAL_KERNEL_SOURCE} LLM_METAL_KERNEL_HEX HEX)
    string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1," LLM_METAL_KERNEL_BYTES
                         "${LLM_METAL_KERNEL_HEX}")
    file(READ ${LLM_METAL_SIMDGROUP_SOURCE} LLM_METAL_SIMDGROUP_HEX HEX)
    string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1," LLM_METAL_SIMDGROUP_BYTES
                         "${LLM_METAL_SIMDGROUP_HEX}")
    file(MAKE_DIRECTORY ${LLM_METAL_GENERATED_DIRECTORY})
    set(LLM_METAL_HAS_PRECOMPILED_LIBRARY 0)
    set(LLM_METAL_LIBRARY_BYTES "")
    execute_process(
        COMMAND xcrun -f metal
        RESULT_VARIABLE LLM_METAL_COMPILER_RESULT
        OUTPUT_VARIABLE LLM_METAL_COMPILER
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND xcrun -f metallib
        RESULT_VARIABLE LLM_METALLIB_RESULT
        OUTPUT_VARIABLE LLM_METALLIB_LINKER
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(LLM_METAL_COMPILER_RESULT EQUAL 0 AND LLM_METALLIB_RESULT EQUAL 0)
        set(LLM_METAL_AIR ${LLM_METAL_GENERATED_DIRECTORY}/runtime.air)
        set(LLM_METAL_SIMDGROUP_AIR ${LLM_METAL_GENERATED_DIRECTORY}/matmul_simdgroup.air)
        set(LLM_METAL_LIBRARY ${LLM_METAL_GENERATED_DIRECTORY}/runtime.metallib)
        execute_process(
            COMMAND ${LLM_METAL_COMPILER} -c ${LLM_METAL_KERNEL_SOURCE} -o ${LLM_METAL_AIR}
            RESULT_VARIABLE LLM_METAL_AIR_RESULT
            ERROR_VARIABLE LLM_METAL_AIR_ERROR)
        if(LLM_METAL_AIR_RESULT EQUAL 0)
            execute_process(
                COMMAND ${LLM_METAL_COMPILER} -c ${LLM_METAL_SIMDGROUP_SOURCE}
                        -o ${LLM_METAL_SIMDGROUP_AIR}
                RESULT_VARIABLE LLM_METAL_SIMDGROUP_AIR_RESULT
                ERROR_VARIABLE LLM_METAL_SIMDGROUP_AIR_ERROR)
        endif()
        if(LLM_METAL_AIR_RESULT EQUAL 0 AND LLM_METAL_SIMDGROUP_AIR_RESULT EQUAL 0)
            execute_process(
                COMMAND ${LLM_METALLIB_LINKER} ${LLM_METAL_AIR} ${LLM_METAL_SIMDGROUP_AIR}
                        -o ${LLM_METAL_LIBRARY}
                RESULT_VARIABLE LLM_METALLIB_BUILD_RESULT
                ERROR_VARIABLE LLM_METALLIB_BUILD_ERROR)
        endif()
        if(LLM_METAL_AIR_RESULT EQUAL 0 AND LLM_METAL_SIMDGROUP_AIR_RESULT EQUAL 0
           AND LLM_METALLIB_BUILD_RESULT EQUAL 0)
            file(READ ${LLM_METAL_LIBRARY} LLM_METAL_LIBRARY_HEX HEX)
            string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1,"
                                 LLM_METAL_LIBRARY_BYTES "${LLM_METAL_LIBRARY_HEX}")
            set(LLM_METAL_HAS_PRECOMPILED_LIBRARY 1)
            message(STATUS "Metal kernels: embedded precompiled metallib")
        else()
            message(WARNING "Metal offline compilation failed; using source fallback")
        endif()
    else()
        message(STATUS "Metal kernels: offline compiler unavailable; using source fallback")
    endif()
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MetalKernels.h.in
        ${LLM_METAL_GENERATED_DIRECTORY}/metal_kernels_embedded.h
        @ONLY)
    target_sources(runtime PRIVATE
        src/runtime/backends/metal/metal_backend.m
        src/runtime/backends/metal/metal_memory.m
        src/runtime/backends/metal/metal_operations.m)
    target_include_directories(runtime PRIVATE ${LLM_METAL_GENERATED_DIRECTORY})
    target_link_libraries(runtime PUBLIC ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})
else()
    target_sources(runtime PRIVATE src/runtime/backends/metal/metal_backend_stub.c)
endif()

target_include_directories(runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/metal)
target_compile_features(runtime PUBLIC c_std_23)
target_link_libraries(runtime PUBLIC Threads::Threads)
llm_lab_set_warnings(runtime)
if(UNIX AND NOT APPLE)
    target_link_libraries(runtime PUBLIC m)
endif()

add_library(tokenizer STATIC
    src/tokenizer/model.c
    src/tokenizer/model_io.c
    src/tokenizer/pretokenizer.c
    src/tokenizer/sha256.c
    src/tokenizer/tokenizer.c
    src/tokenizer/train.c
    src/tokenizer/train_data.c)
target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tokenizer PUBLIC c_std_23)
llm_lab_set_warnings(tokenizer)

add_library(dataset STATIC
    src/dataset/dataset.c
    src/dataset/dataset_io.c
    src/dataset/jsonl.c)
target_include_directories(dataset
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer)
target_compile_features(dataset PUBLIC c_std_23)
target_link_libraries(dataset PUBLIC tokenizer)
llm_lab_set_warnings(dataset)

add_executable(llm_lab_cli apps/llm-lab/main.c)
target_compile_features(llm_lab_cli PRIVATE c_std_23)
target_link_libraries(llm_lab_cli PRIVATE dataset tokenizer)
llm_lab_set_warnings(llm_lab_cli)
set_target_properties(llm_lab_cli PROPERTIES OUTPUT_NAME llm-lab)

add_executable(tokenizer_experiment apps/tokenizer-experiment/main.c)
target_compile_features(tokenizer_experiment PRIVATE c_std_23)
target_link_libraries(tokenizer_experiment PRIVATE tokenizer)
llm_lab_set_warnings(tokenizer_experiment)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(LLM_LAB_BUILD_BENCHMARKS)
    add_subdirectory(utils/benchmarks)
endif()
