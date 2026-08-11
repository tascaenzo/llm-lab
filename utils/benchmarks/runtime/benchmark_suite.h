#ifndef LLM_LAB_BENCHMARK_SUITE_H
#define LLM_LAB_BENCHMARK_SUITE_H

#include <stddef.h>

#include "runtime/runtime.h"

typedef enum cpu_benchmark_operation {
    CPU_BENCHMARK_COPY = 0,
    CPU_BENCHMARK_ADD,
    CPU_BENCHMARK_REDUCE_SUM,
    CPU_BENCHMARK_MATMUL,
    CPU_BENCHMARK_GATHER,
    CPU_BENCHMARK_SCATTER_ADD,
    CPU_BENCHMARK_SOFTMAX,
    CPU_BENCHMARK_CROSS_ENTROPY_FORWARD,
    CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD,
    CPU_BENCHMARK_OPERATION_COUNT
} cpu_benchmark_operation;

typedef enum runtime_benchmark_backend {
    RUNTIME_BENCHMARK_CPU = 0,
    RUNTIME_BENCHMARK_METAL,
} runtime_benchmark_backend;

typedef struct cpu_benchmark_config {
    size_t elements;
    size_t rows;
    size_t columns;
    size_t inner_size;
    size_t warmup_iterations;
    size_t measured_iterations;
    double minimum_sample_seconds;
    int deterministic;
    llm_dtype matmul_dtype;
} cpu_benchmark_config;

typedef struct cpu_benchmark_result {
    cpu_benchmark_operation operation;
    runtime_benchmark_backend backend;
    llm_dtype dtype;
    size_t requested_threads;
    size_t actual_threads;
    size_t repetitions_per_sample;
    double minimum_seconds;
    double median_seconds;
    double p95_seconds;
    double mean_seconds;
    double standard_deviation_seconds;
    double throughput;
    const char *throughput_unit;
    double guard_value;
    double gpu_median_seconds;
    double gpu_p95_seconds;
    double pipeline_compilation_seconds;
    char device_name[128];
} cpu_benchmark_result;

const char *cpu_benchmark_operation_name(cpu_benchmark_operation operation);
int cpu_benchmark_operation_parse(const char *name, cpu_benchmark_operation *out_operation);

const char *runtime_benchmark_backend_name(runtime_benchmark_backend backend);
const char *runtime_benchmark_dtype_name(llm_dtype dtype);

int runtime_benchmark_run(runtime_benchmark_backend backend, cpu_benchmark_operation operation,
                          size_t requested_threads, const cpu_benchmark_config *config,
                          cpu_benchmark_result *out_result);

#endif
