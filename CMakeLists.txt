cmake_minimum_required(VERSION 3.24)

project(
    llm_lab
    VERSION 0.1.0
    DESCRIPTION "A from-scratch LLM learning project written in C"
    LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(LLM_LAB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

include(cmake/CompilerWarnings.cmake)

add_library(tokenizer STATIC
    src/tokenizer/model.c
    src/tokenizer/model_io.c
    src/tokenizer/pretokenizer.c
    src/tokenizer/tokenizer.c
    src/tokenizer/train.c)
target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tokenizer PUBLIC c_std_23)
llm_lab_set_warnings(tokenizer)

add_executable(llm_lab_cli apps/llm-lab/main.c)
target_compile_features(llm_lab_cli PRIVATE c_std_23)
target_link_libraries(llm_lab_cli PRIVATE tokenizer)
llm_lab_set_warnings(llm_lab_cli)
set_target_properties(llm_lab_cli PROPERTIES OUTPUT_NAME llm-lab)

add_executable(tokenizer_experiment apps/tokenizer-experiment/main.c)
target_compile_features(tokenizer_experiment PRIVATE c_std_23)
target_link_libraries(tokenizer_experiment PRIVATE tokenizer)
llm_lab_set_warnings(tokenizer_experiment)
