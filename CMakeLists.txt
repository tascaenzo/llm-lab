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
target_include_directories(runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/backends/cpu)
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
