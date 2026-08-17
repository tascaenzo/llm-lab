find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(runtime_test runtime/test_runtime.c)
target_compile_features(runtime_test PRIVATE c_std_23)
target_link_libraries(runtime_test PRIVATE runtime)
target_include_directories(runtime_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(runtime_test)
add_test(NAME runtime.unit COMMAND runtime_test)

add_executable(runtime_operations_test runtime/test_operations.c)
target_compile_features(runtime_operations_test PRIVATE c_std_23)
target_link_libraries(runtime_operations_test PRIVATE runtime)
target_include_directories(runtime_operations_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(runtime_operations_test)
add_test(NAME runtime.operations COMMAND runtime_operations_test)

add_executable(runtime_language_operations_test runtime/test_language_operations.c)
target_compile_features(runtime_language_operations_test PRIVATE c_std_23)
target_link_libraries(runtime_language_operations_test PRIVATE runtime)
target_include_directories(runtime_language_operations_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(runtime_language_operations_test)
add_test(NAME runtime.language_operations COMMAND runtime_language_operations_test)

add_executable(runtime_cpu_backend_test runtime/test_cpu_backend.c)
target_compile_features(runtime_cpu_backend_test PRIVATE c_std_23)
target_link_libraries(runtime_cpu_backend_test PRIVATE runtime)
target_include_directories(runtime_cpu_backend_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
    ${PROJECT_SOURCE_DIR}/src/runtime/backends/cpu)
llm_lab_set_warnings(runtime_cpu_backend_test)
add_test(NAME runtime.cpu_backend COMMAND runtime_cpu_backend_test)

add_executable(runtime_cpu_transformer_test
    runtime/backend_contract_suite.c
    runtime/test_cpu_backend_contract.c)
target_compile_features(runtime_cpu_transformer_test PRIVATE c_std_23)
target_link_libraries(runtime_cpu_transformer_test PRIVATE runtime)
target_include_directories(runtime_cpu_transformer_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(runtime_cpu_transformer_test)
add_test(NAME runtime.backend_contract.cpu COMMAND runtime_cpu_transformer_test)

add_executable(runtime_metal_backend_test
    runtime/backend_contract_suite.c
    runtime/test_metal_backend.c)
target_compile_features(runtime_metal_backend_test PRIVATE c_std_23)
target_link_libraries(runtime_metal_backend_test PRIVATE model runtime)
target_include_directories(runtime_metal_backend_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(runtime_metal_backend_test)
add_test(NAME runtime.metal_backend COMMAND runtime_metal_backend_test)

add_executable(runtime_cuda_backend_test
    runtime/backend_contract_suite.c
    runtime/test_cuda_backend.c)
target_compile_features(runtime_cuda_backend_test PRIVATE c_std_23)
target_link_libraries(runtime_cuda_backend_test PRIVATE model runtime)
target_include_directories(runtime_cuda_backend_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(runtime_cuda_backend_test)
add_test(NAME runtime.cuda_backend COMMAND runtime_cuda_backend_test)

function(llm_lab_add_tokenizer_test test_name source)
    add_executable(${test_name} ${source})
    target_compile_features(${test_name} PRIVATE c_std_23)
    target_link_libraries(${test_name} PRIVATE tokenizer)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
        ${PROJECT_SOURCE_DIR}/src/tokenizer)
    target_compile_definitions(${test_name} PRIVATE
        LLM_LAB_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
        LLM_LAB_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
    llm_lab_set_warnings(${test_name})
    add_test(NAME tokenizer.${test_name} COMMAND ${test_name})
endfunction()

llm_lab_add_tokenizer_test(encode_decode tokenizer/test_encode_decode.c)
llm_lab_add_tokenizer_test(model_io tokenizer/test_model_io.c)
llm_lab_add_tokenizer_test(pretokenizer tokenizer/test_pretokenizer.c)
llm_lab_add_tokenizer_test(training tokenizer/test_training.c)

add_executable(dataset_test dataset/test_dataset.c)
target_compile_features(dataset_test PRIVATE c_std_23)
target_link_libraries(dataset_test PRIVATE dataset)
target_include_directories(dataset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
target_compile_definitions(dataset_test PRIVATE
    LLM_LAB_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LLM_LAB_TEST_DOCUMENTS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/documents.jsonl")
llm_lab_set_warnings(dataset_test)
add_test(NAME dataset.unit COMMAND dataset_test)

add_executable(model_test model/test_model.c)
target_compile_features(model_test PRIVATE c_std_23)
target_link_libraries(model_test PRIVATE model)
target_include_directories(model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer)
llm_lab_set_warnings(model_test)
add_test(NAME model.unit COMMAND model_test)

add_test(
    NAME python.unit
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${CMAKE_CURRENT_SOURCE_DIR}/python
        -p test_*.py)
set_tests_properties(python.unit PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME integration.cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli.py
        $<TARGET_FILE:llm_lab_cli>
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-a.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-b.txt)

add_test(
    NAME integration.artifact_pipeline
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_artifact_pipeline.py
        ${PROJECT_SOURCE_DIR}/utils/corpus/train_tokenizer.py
        $<TARGET_FILE:llm_lab_cli>
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-a.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-b.txt)

add_test(
    NAME integration.tokenizer_experiment
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_experiment.py
        $<TARGET_FILE:llm_lab_cli>
        $<TARGET_FILE:tokenizer_experiment>
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-a.txt)

add_test(
    NAME integration.dataset_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_dataset_cli.py
        $<TARGET_FILE:llm_lab_cli>
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/documents.jsonl
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-a.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny-corpus-b.txt)

set_tests_properties(
    python.unit integration.cli integration.artifact_pipeline integration.tokenizer_experiment
    integration.dataset_cli
    PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
