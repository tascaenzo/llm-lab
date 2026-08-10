add_executable(runtime_cpu_benchmark
    runtime/benchmark_cpu.c
    runtime/benchmark_suite.c)
target_compile_features(runtime_cpu_benchmark PRIVATE c_std_23)
target_link_libraries(runtime_cpu_benchmark PRIVATE runtime)
llm_lab_set_warnings(runtime_cpu_benchmark)

if(BUILD_TESTING)
    add_test(
        NAME benchmark.cpu.smoke
        COMMAND runtime_cpu_benchmark
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
        COMMAND runtime_cpu_benchmark
            --format jsonl
            --operations add
            --threads 1
            --elements 4096
            --warmup 1
            --iterations 2
            --sample-ms 1)
    set_tests_properties(benchmark.cpu.jsonl_smoke PROPERTIES TIMEOUT 60)
endif()
