find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

set_tests_properties(
    python.unit integration.cli integration.artifact_pipeline integration.tokenizer_experiment
    PROPERTIES
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
