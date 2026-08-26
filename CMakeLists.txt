cmake_minimum_required(VERSION 3.22)

if(WIN32)
    message(FATAL_ERROR
        "Windows is not supported by llm-lab at this stage. Use macOS or Linux.")
endif()

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
    src/runtime/backends/cpu/cpu_linear.c
    src/runtime/backends/cpu/cpu_operations.c
    src/runtime/backends/cpu/cpu_optimizer.c
    src/runtime/backends/cpu/cpu_simd.c
    src/runtime/backends/cpu/cpu_transformer.c
    src/runtime/backends/cpu/cpu_threads.c
    src/runtime/memory.c
    src/runtime/operations.c
    src/runtime/tensor.c)

if(APPLE)
    enable_language(OBJC)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METAL_PERFORMANCE_SHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)

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
    target_link_libraries(runtime PUBLIC ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK}
                                         ${METAL_PERFORMANCE_SHADERS_FRAMEWORK})
else()
    target_sources(runtime PRIVATE src/runtime/backends/metal/metal_backend_stub.c)
endif()

# The CUDA backend is built whenever a CUDA toolchain is present, which is the
# case on the cloud GPU hosts and never on the Apple Silicon workstation. Set
# LLM_LAB_ENABLE_CUDA=OFF to skip it, or ON to fail loudly when it is missing.
include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED LLM_LAB_ENABLE_CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(LLM_LAB_ENABLE_CUDA ON)
    else()
        set(LLM_LAB_ENABLE_CUDA OFF)
    endif()
endif()
set(LLM_LAB_ENABLE_CUDA ${LLM_LAB_ENABLE_CUDA} CACHE BOOL "Build the CUDA backend")

if(LLM_LAB_ENABLE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "LLM_LAB_ENABLE_CUDA is ON but no CUDA compiler was found. "
            "Install the CUDA toolkit or configure with -DLLM_LAB_ENABLE_CUDA=OFF.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Not "native": detecting the local GPU fails on a build machine that has
    # none, which is exactly the case for a compile-only CI runner and for a
    # build host separate from the training host. The default covers the cloud
    # GPUs this project targets; override it to cut compile time to one.
    #   70 = V100, 80 = A100, 89 = L4 / RTX 4090, 90 = H100
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;89;90")
    endif()
    target_sources(runtime PRIVATE
        src/runtime/backends/cuda/cuda_backend.cu
        src/runtime/backends/cuda/cuda_kernels.cu
        src/runtime/backends/cuda/cuda_memory.cu
        src/runtime/backends/cuda/cuda_operations.cu)
    set_target_properties(runtime PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(runtime PUBLIC CUDA::cudart CUDA::cublas)
    message(STATUS "CUDA backend: enabled for architectures ${CMAKE_CUDA_ARCHITECTURES}")
else()
    target_sources(runtime PRIVATE src/runtime/backends/cuda/cuda_backend_stub.c)
    message(STATUS "CUDA backend: disabled")
endif()

target_include_directories(runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/cuda
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
    src/dataset/jsonl.c
    src/dataset/json_tokens.c
    src/dataset/sft_dataset.c)
target_include_directories(dataset
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer)
target_compile_features(dataset PUBLIC c_std_23)
target_link_libraries(dataset PUBLIC tokenizer)
llm_lab_set_warnings(dataset)

add_library(model STATIC
    src/model/model.c
    src/model/checkpoint.c
    src/model/parameter.c
    src/model/trainer.c
    src/model/layers/embedding.c
    src/model/layers/transformer.c
    src/model/layers/output_head.c)
target_include_directories(model
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/model)
target_compile_features(model PUBLIC c_std_23)
target_link_libraries(model PUBLIC dataset runtime)
llm_lab_set_warnings(model)

add_library(project_environment STATIC apps/common/project_environment.c)
target_include_directories(project_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps/common)
target_compile_features(project_environment PUBLIC c_std_23)
llm_lab_set_warnings(project_environment)

add_executable(llm_lab_cli
    apps/llm-lab/main.c
    apps/llm-lab/cli_commands.c)
target_compile_features(llm_lab_cli PRIVATE c_std_23)
target_link_libraries(llm_lab_cli PRIVATE model dataset tokenizer project_environment)
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
