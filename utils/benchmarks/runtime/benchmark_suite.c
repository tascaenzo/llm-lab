#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "benchmark_suite.h"
#include "runtime/runtime.h"

typedef struct cpu_benchmark_workload {
    cpu_benchmark_operation operation;
    llm_backend *backend;
    llm_tensor first;
    llm_tensor second;
    llm_tensor indices;
    llm_tensor output;
    size_t elements;
    size_t rows;
    size_t columns;
    size_t inner_size;
} cpu_benchmark_workload;

static const char *const operation_names[CPU_BENCHMARK_OPERATION_COUNT] = {
    [CPU_BENCHMARK_COPY] = "copy",
    [CPU_BENCHMARK_ADD] = "add",
    [CPU_BENCHMARK_REDUCE_SUM] = "reduce_sum",
    [CPU_BENCHMARK_MATMUL] = "matmul",
    [CPU_BENCHMARK_GATHER] = "gather",
    [CPU_BENCHMARK_SOFTMAX] = "softmax",
    [CPU_BENCHMARK_CROSS_ENTROPY_FORWARD] = "cross_entropy_forward",
    [CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD] = "cross_entropy_backward",
};

const char *cpu_benchmark_operation_name(cpu_benchmark_operation operation) {
    if (operation < CPU_BENCHMARK_COPY || operation >= CPU_BENCHMARK_OPERATION_COUNT) {
        return "unknown";
    }
    return operation_names[operation];
}

int cpu_benchmark_operation_parse(const char *name, cpu_benchmark_operation *out_operation) {
    if (name == NULL || out_operation == NULL) {
        return 0;
    }
    for (size_t index = 0U; index < CPU_BENCHMARK_OPERATION_COUNT; ++index) {
        if (strcmp(name, operation_names[index]) == 0) {
            *out_operation = (cpu_benchmark_operation)index;
            return 1;
        }
    }
    return 0;
}

static int checked_multiply(size_t left, size_t right, size_t *out_product) {
    if (left != 0U && right > SIZE_MAX / left) {
        return 0;
    }
    *out_product = left * right;
    return 1;
}

static double current_seconds(void) {
    struct timespec time = {0};
    if (timespec_get(&time, TIME_UTC) != TIME_UTC) {
        return 0.0;
    }
    return (double)time.tv_sec + (double)time.tv_nsec / 1000000000.0;
}

static int compare_double(const void *left, const void *right) {
    const double left_value = *(const double *)left;
    const double right_value = *(const double *)right;
    return (left_value > right_value) - (left_value < right_value);
}

static void workload_destroy(cpu_benchmark_workload *workload) {
    llm_tensor_destroy(&workload->output);
    llm_tensor_destroy(&workload->indices);
    llm_tensor_destroy(&workload->second);
    llm_tensor_destroy(&workload->first);
    llm_backend_destroy(workload->backend);
    *workload = (cpu_benchmark_workload){0};
}

static llm_status create_f32_tensor(llm_backend *backend, size_t rank, const size_t *shape,
                                    llm_tensor *out_tensor, float fill_value) {
    llm_status status = llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, out_tensor);
    if (status == LLM_OK) {
        status = llm_tensor_fill_f32(backend, out_tensor, fill_value);
    }
    return status;
}

static llm_status create_indices(llm_backend *backend, size_t count, size_t upper_bound,
                                 llm_tensor *out_tensor) {
    if (upper_bound == 0U || upper_bound > UINT32_MAX) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t shape[] = {count};
    llm_status status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, shape, out_tensor);
    if (status != LLM_OK) {
        return status;
    }
    if (count > SIZE_MAX / sizeof(uint32_t)) {
        return LLM_OVERFLOW;
    }
    uint32_t *values = malloc(count * sizeof(*values));
    if (values == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    for (size_t index = 0U; index < count; ++index) {
        values[index] = (uint32_t)(index % upper_bound);
    }
    status = llm_tensor_write(backend, out_tensor, values, count * sizeof(*values));
    free(values);
    return status;
}

static llm_status setup_vector_workload(cpu_benchmark_workload *workload) {
    const size_t shape[] = {workload->elements};
    llm_status status = create_f32_tensor(workload->backend, 1U, shape, &workload->first, 1.25F);
    if (status == LLM_OK && workload->operation == CPU_BENCHMARK_ADD) {
        status = create_f32_tensor(workload->backend, 1U, shape, &workload->second, 2.5F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 1U, shape, &workload->output, 0.0F);
    }
    return status;
}

static llm_status setup_matrix_workload(cpu_benchmark_workload *workload) {
    const size_t matrix_shape[] = {workload->rows, workload->columns};
    const size_t row_shape[] = {workload->rows};
    llm_status status = LLM_OK;
    switch (workload->operation) {
    case CPU_BENCHMARK_REDUCE_SUM:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->first, 0.25F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, row_shape, &workload->output, 0.0F);
        }
        break;
    case CPU_BENCHMARK_GATHER:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->first, 0.5F);
        if (status == LLM_OK) {
            status = create_indices(workload->backend, workload->rows, workload->rows,
                                    &workload->indices);
        }
        if (status == LLM_OK) {
            status =
                create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->output, 0.0F);
        }
        break;
    case CPU_BENCHMARK_SOFTMAX:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->first, 0.0F);
        if (status == LLM_OK) {
            status =
                create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->output, 0.0F);
        }
        break;
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->first, 0.0F);
        if (status == LLM_OK) {
            status = create_indices(workload->backend, workload->rows, workload->columns,
                                    &workload->indices);
        }
        if (status == LLM_OK && workload->operation == CPU_BENCHMARK_CROSS_ENTROPY_FORWARD) {
            status = create_f32_tensor(workload->backend, 0U, NULL, &workload->output, 0.0F);
        } else if (status == LLM_OK) {
            status =
                create_f32_tensor(workload->backend, 2U, matrix_shape, &workload->output, 0.0F);
        }
        break;
    default:
        status = LLM_INVALID_ARGUMENT;
        break;
    }
    return status;
}

static llm_status setup_matmul_workload(cpu_benchmark_workload *workload) {
    const size_t left_shape[] = {workload->rows, workload->inner_size};
    const size_t right_shape[] = {workload->inner_size, workload->columns};
    const size_t output_shape[] = {workload->rows, workload->columns};
    llm_status status =
        create_f32_tensor(workload->backend, 2U, left_shape, &workload->first, 0.25F);
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 2U, right_shape, &workload->second, -0.125F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 2U, output_shape, &workload->output, 0.0F);
    }
    return status;
}

static llm_status workload_setup(cpu_benchmark_operation operation, size_t requested_threads,
                                 const cpu_benchmark_config *config,
                                 cpu_benchmark_workload *out_workload) {
    *out_workload = (cpu_benchmark_workload){
        .operation = operation,
        .elements = config->elements,
        .rows = config->rows,
        .columns = config->columns,
        .inner_size = config->inner_size,
    };
    const llm_cpu_backend_config backend_config = {
        .thread_count = requested_threads,
        .deterministic = config->deterministic,
    };
    llm_status status = llm_backend_cpu_create_with_config(&backend_config, &out_workload->backend);
    if (status != LLM_OK) {
        return status;
    }
    switch (operation) {
    case CPU_BENCHMARK_COPY:
    case CPU_BENCHMARK_ADD:
        status = setup_vector_workload(out_workload);
        break;
    case CPU_BENCHMARK_REDUCE_SUM:
    case CPU_BENCHMARK_GATHER:
    case CPU_BENCHMARK_SOFTMAX:
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        status = setup_matrix_workload(out_workload);
        break;
    case CPU_BENCHMARK_MATMUL:
        status = setup_matmul_workload(out_workload);
        break;
    case CPU_BENCHMARK_OPERATION_COUNT:
        status = LLM_INVALID_ARGUMENT;
        break;
    }
    if (status != LLM_OK) {
        workload_destroy(out_workload);
    }
    return status;
}

static llm_status workload_execute(cpu_benchmark_workload *workload) {
    switch (workload->operation) {
    case CPU_BENCHMARK_COPY:
        return llm_tensor_copy(workload->backend, &workload->first, &workload->output);
    case CPU_BENCHMARK_ADD:
        return llm_add(workload->backend, &workload->first, &workload->second, &workload->output);
    case CPU_BENCHMARK_REDUCE_SUM:
        return llm_reduce_sum_last(workload->backend, &workload->first, &workload->output);
    case CPU_BENCHMARK_MATMUL:
        return llm_matmul(workload->backend, &workload->first, &workload->second,
                          &workload->output);
    case CPU_BENCHMARK_GATHER:
        return llm_gather_rows(workload->backend, &workload->first, &workload->indices,
                               &workload->output);
    case CPU_BENCHMARK_SOFTMAX:
        return llm_softmax_last(workload->backend, &workload->first, &workload->output);
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
        return llm_cross_entropy_forward(workload->backend, &workload->first, &workload->indices,
                                         &workload->output);
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        return llm_cross_entropy_backward(workload->backend, &workload->first, &workload->indices,
                                          &workload->output);
    case CPU_BENCHMARK_OPERATION_COUNT:
        return LLM_INVALID_ARGUMENT;
    }
    return LLM_INVALID_ARGUMENT;
}

static int close_enough(float actual, float expected) {
    const float scale = fabsf(expected) > 1.0F ? fabsf(expected) : 1.0F;
    return fabsf(actual - expected) <= 1.0e-4F * scale;
}

static int workload_verify(cpu_benchmark_workload *workload, double *out_guard_value) {
    if (workload->output.element_count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    const size_t byte_count = workload->output.element_count * sizeof(float);
    float *values = malloc(byte_count);
    if (values == NULL) {
        return 0;
    }
    if (llm_tensor_read(workload->backend, &workload->output, values, byte_count) != LLM_OK) {
        free(values);
        return 0;
    }

    float expected = 0.0F;
    switch (workload->operation) {
    case CPU_BENCHMARK_COPY:
        expected = 1.25F;
        break;
    case CPU_BENCHMARK_ADD:
        expected = 3.75F;
        break;
    case CPU_BENCHMARK_REDUCE_SUM:
        expected = (float)workload->columns * 0.25F;
        break;
    case CPU_BENCHMARK_MATMUL:
        expected = (float)workload->inner_size * -0.03125F;
        break;
    case CPU_BENCHMARK_GATHER:
        expected = 0.5F;
        break;
    case CPU_BENCHMARK_SOFTMAX:
        expected = 1.0F / (float)workload->columns;
        break;
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
        expected = logf((float)workload->columns);
        break;
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        expected = 1.0F / ((float)workload->rows * (float)workload->columns) -
                   1.0F / (float)workload->rows;
        break;
    case CPU_BENCHMARK_OPERATION_COUNT:
        free(values);
        return 0;
    }
    const int valid = close_enough(values[0], expected);
    *out_guard_value = (double)values[0];
    free(values);
    return valid;
}

static double workload_throughput(const cpu_benchmark_workload *workload, double seconds,
                                  const char **out_unit) {
    double work = 0.0;
    switch (workload->operation) {
    case CPU_BENCHMARK_COPY:
        work = 2.0 * (double)workload->elements * sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_ADD:
        work = 3.0 * (double)workload->elements * sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_REDUCE_SUM:
        work = ((double)workload->rows * (double)workload->columns + (double)workload->rows) *
               sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_MATMUL:
        work =
            2.0 * (double)workload->rows * (double)workload->inner_size * (double)workload->columns;
        *out_unit = "GFLOP/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_GATHER:
        work = (2.0 * (double)workload->rows * (double)workload->columns * sizeof(float)) +
               (double)workload->rows * sizeof(uint32_t);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_SOFTMAX:
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        work = (double)workload->rows * (double)workload->columns;
        *out_unit = "M elements/s";
        return work / seconds / 1.0e6;
    case CPU_BENCHMARK_OPERATION_COUNT:
        *out_unit = "unknown";
        return 0.0;
    }
    *out_unit = "unknown";
    return 0.0;
}

int cpu_benchmark_run(cpu_benchmark_operation operation, size_t requested_threads,
                      const cpu_benchmark_config *config, cpu_benchmark_result *out_result) {
    if (operation < CPU_BENCHMARK_COPY || operation >= CPU_BENCHMARK_OPERATION_COUNT ||
        config == NULL || out_result == NULL || config->elements == 0U || config->rows == 0U ||
        config->columns == 0U || config->inner_size == 0U || config->warmup_iterations == 0U ||
        config->measured_iterations == 0U || config->minimum_sample_seconds <= 0.0 ||
        config->measured_iterations > SIZE_MAX / sizeof(double)) {
        return 0;
    }
    size_t matrix_elements = 0U;
    if (checked_multiply(config->rows, config->columns, &matrix_elements) == 0) {
        return 0;
    }
    (void)matrix_elements;

    cpu_benchmark_workload workload = {0};
    const llm_status setup_status = workload_setup(operation, requested_threads, config, &workload);
    if (setup_status != LLM_OK) {
        fprintf(stderr, "%s setup failed: %s\n", cpu_benchmark_operation_name(operation),
                llm_status_string(setup_status));
        return 0;
    }
    for (size_t iteration = 0U; iteration < config->warmup_iterations; ++iteration) {
        const llm_status status = workload_execute(&workload);
        if (status != LLM_OK) {
            fprintf(stderr, "%s warmup failed: %s\n", cpu_benchmark_operation_name(operation),
                    llm_status_string(status));
            workload_destroy(&workload);
            return 0;
        }
    }

    size_t probe_repetitions = 1U;
    double probe_total_seconds = 0.0;
    llm_status probe_status = LLM_OK;
    while (probe_total_seconds <= 0.0 && probe_repetitions <= 1048576U) {
        const double probe_start = current_seconds();
        for (size_t repetition = 0U; repetition < probe_repetitions; ++repetition) {
            probe_status = workload_execute(&workload);
            if (probe_status != LLM_OK) {
                break;
            }
        }
        probe_total_seconds = current_seconds() - probe_start;
        if (probe_total_seconds <= 0.0 && probe_status == LLM_OK) {
            probe_repetitions *= 2U;
        }
    }
    if (probe_status != LLM_OK || probe_total_seconds <= 0.0) {
        fprintf(stderr, "%s calibration failed: %s\n", cpu_benchmark_operation_name(operation),
                llm_status_string(probe_status));
        workload_destroy(&workload);
        return 0;
    }
    const double probe_seconds = probe_total_seconds / (double)probe_repetitions;
    size_t repetitions_per_sample = 1U;
    if (probe_seconds < config->minimum_sample_seconds) {
        const double requested_repetitions = ceil(config->minimum_sample_seconds / probe_seconds);
        const double maximum_repetitions = 1000000.0;
        repetitions_per_sample =
            (size_t)(requested_repetitions > maximum_repetitions ? maximum_repetitions
                                                                 : requested_repetitions);
    }

    double *durations = malloc(config->measured_iterations * sizeof(*durations));
    if (durations == NULL) {
        workload_destroy(&workload);
        return 0;
    }
    for (size_t iteration = 0U; iteration < config->measured_iterations; ++iteration) {
        const double start = current_seconds();
        llm_status status = LLM_OK;
        for (size_t repetition = 0U; repetition < repetitions_per_sample; ++repetition) {
            status = workload_execute(&workload);
            if (status != LLM_OK) {
                break;
            }
        }
        durations[iteration] = (current_seconds() - start) / (double)repetitions_per_sample;
        if (status != LLM_OK || durations[iteration] <= 0.0) {
            fprintf(stderr, "%s measurement failed: %s\n", cpu_benchmark_operation_name(operation),
                    llm_status_string(status));
            free(durations);
            workload_destroy(&workload);
            return 0;
        }
    }

    double guard_value = 0.0;
    if (workload_verify(&workload, &guard_value) == 0) {
        fprintf(stderr, "%s produced an invalid result\n", cpu_benchmark_operation_name(operation));
        free(durations);
        workload_destroy(&workload);
        return 0;
    }

    double mean = 0.0;
    for (size_t index = 0U; index < config->measured_iterations; ++index) {
        mean += durations[index];
    }
    mean /= (double)config->measured_iterations;
    double variance = 0.0;
    for (size_t index = 0U; index < config->measured_iterations; ++index) {
        const double difference = durations[index] - mean;
        variance += difference * difference;
    }
    variance /= (double)config->measured_iterations;
    qsort(durations, config->measured_iterations, sizeof(*durations), compare_double);
    const size_t median_index = config->measured_iterations / 2U;
    const double median = config->measured_iterations % 2U == 0U
                              ? (durations[median_index - 1U] + durations[median_index]) * 0.5
                              : durations[median_index];
    size_t p95_rank = (size_t)ceil((double)config->measured_iterations * 0.95);
    if (p95_rank == 0U) {
        p95_rank = 1U;
    }
    const size_t p95_index = p95_rank - 1U;

    *out_result = (cpu_benchmark_result){
        .operation = operation,
        .requested_threads = requested_threads,
        .actual_threads = llm_backend_cpu_thread_count(workload.backend),
        .repetitions_per_sample = repetitions_per_sample,
        .minimum_seconds = durations[0],
        .median_seconds = median,
        .p95_seconds = durations[p95_index],
        .mean_seconds = mean,
        .standard_deviation_seconds = sqrt(variance),
        .guard_value = guard_value,
    };
    out_result->throughput =
        workload_throughput(&workload, out_result->median_seconds, &out_result->throughput_unit);

    free(durations);
    workload_destroy(&workload);
    return 1;
}
