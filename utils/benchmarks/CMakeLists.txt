add_executable(runtime_benchmark
    runtime/benchmark_cpu.c
    runtime/benchmark_suite.c)
target_compile_features(runtime_benchmark PRIVATE c_std_23)
target_link_libraries(runtime_benchmark PRIVATE runtime)
llm_lab_set_warnings(runtime_benchmark)

add_executable(runtime_benchmark_report
    runtime/benchmark_report.c
    runtime/benchmark_suite.c)
target_compile_features(runtime_benchmark_report PRIVATE c_std_23)
target_link_libraries(runtime_benchmark_report PRIVATE runtime)
llm_lab_set_warnings(runtime_benchmark_report)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(runtime_performance_suite
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/performance_suite.py
        --benchmark $<TARGET_FILE:runtime_benchmark>
    DEPENDS runtime_benchmark
    USES_TERMINAL)

if(BUILD_TESTING)
    add_test(
        NAME benchmark.cpu.smoke
        COMMAND runtime_benchmark
            --backend cpu
            --operations all
            --threads 1,2
            --elements 4096
            --rows 16
            --columns 16
            --inner 16
            --warmup 1
            --iterations 2
            --sample-ms 1)
    set_tests_properties(benchmark.cpu.smoke PROPERTIES TIMEOUT 60)

    add_test(
        NAME benchmark.cpu.jsonl_smoke
        COMMAND runtime_benchmark
            --backend cpu
            --format jsonl
            --operations add
            --threads 1
            --elements 4096
            --warmup 1
            --iterations 2
            --sample-ms 1)
    set_tests_properties(benchmark.cpu.jsonl_smoke PROPERTIES TIMEOUT 60)

    add_test(
        NAME benchmark.runtime.metal_smoke
        COMMAND runtime_benchmark
            --backend metal
            --operations matmul
            --precision f16
            --rows 16
            --columns 16
            --inner 16
            --warmup 1
            --iterations 2
            --sample-ms 1)
    set_tests_properties(benchmark.runtime.metal_smoke PROPERTIES TIMEOUT 60)

    add_test(
        NAME benchmark.runtime.report_smoke
        COMMAND runtime_benchmark_report --quick)
    set_tests_properties(benchmark.runtime.report_smoke PROPERTIES TIMEOUT 120)

    add_test(
        NAME benchmark.runtime.performance_suite_smoke
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance_suite.py
            --smoke
            --benchmark $<TARGET_FILE:runtime_benchmark>)
    set_tests_properties(benchmark.runtime.performance_suite_smoke PROPERTIES TIMEOUT 120)
endif()
