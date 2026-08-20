#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "benchmark_suite.h"
#include "runtime/runtime.h"

#define BENCHMARK_TENSOR_COUNT 7U

enum benchmark_tensor_slot {
    BENCHMARK_FIRST = 0,
    BENCHMARK_SECOND,
    BENCHMARK_THIRD,
    BENCHMARK_FOURTH,
    BENCHMARK_FIFTH,
    BENCHMARK_SIXTH,
    BENCHMARK_OUTPUT,
};

typedef struct cpu_benchmark_workload {
    cpu_benchmark_operation operation;
    runtime_benchmark_backend backend_kind;
    llm_backend *backend;
    llm_tensor tensors[BENCHMARK_TENSOR_COUNT];
    size_t elements;
    size_t rows;
    size_t columns;
    size_t inner_size;
    size_t batch_size;
    size_t sequence_length;
    size_t query_head_count;
    size_t key_value_head_count;
    size_t head_dimension;
} cpu_benchmark_workload;

static const char *const operation_names[CPU_BENCHMARK_OPERATION_COUNT] = {
    [CPU_BENCHMARK_ZERO] = "zero",
    [CPU_BENCHMARK_FILL] = "fill",
    [CPU_BENCHMARK_COPY] = "copy",
    [CPU_BENCHMARK_ADD] = "add",
    [CPU_BENCHMARK_MULTIPLY] = "multiply",
    [CPU_BENCHMARK_SCALE] = "scale",
    [CPU_BENCHMARK_ACCUMULATE] = "accumulate",
    [CPU_BENCHMARK_REDUCE_SUM] = "reduce_sum",
    [CPU_BENCHMARK_REDUCE_MAX] = "reduce_max",
    [CPU_BENCHMARK_REDUCE_MEAN_SQUARE] = "reduce_mean_square",
    [CPU_BENCHMARK_MATMUL] = "matmul",
    [CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT] = "matmul_transpose_left",
    [CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT] = "matmul_transpose_right",
    [CPU_BENCHMARK_GATHER] = "gather",
    [CPU_BENCHMARK_SCATTER_ADD] = "scatter_add",
    [CPU_BENCHMARK_SILU] = "silu",
    [CPU_BENCHMARK_SILU_BACKWARD] = "silu_backward",
    [CPU_BENCHMARK_RMS_NORM] = "rms_norm",
    [CPU_BENCHMARK_RMS_NORM_BACKWARD] = "rms_norm_backward",
    [CPU_BENCHMARK_ROPE] = "rope",
    [CPU_BENCHMARK_ROPE_BACKWARD] = "rope_backward",
    [CPU_BENCHMARK_ATTENTION] = "attention",
    [CPU_BENCHMARK_ATTENTION_BACKWARD] = "attention_backward",
    [CPU_BENCHMARK_SOFTMAX] = "softmax",
    [CPU_BENCHMARK_CROSS_ENTROPY_FORWARD] = "cross_entropy_forward",
    [CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD] = "cross_entropy_backward",
    [CPU_BENCHMARK_ADAMW] = "adamw",
};

const char *runtime_benchmark_backend_name(runtime_benchmark_backend backend) {
    return backend == RUNTIME_BENCHMARK_CPU     ? "cpu"
           : backend == RUNTIME_BENCHMARK_METAL ? "metal"
           : backend == RUNTIME_BENCHMARK_CUDA  ? "cuda"
                                                : "unknown";
}

const char *runtime_benchmark_dtype_name(llm_dtype dtype) {
    return dtype == LLM_DTYPE_F32 ? "f32" : dtype == LLM_DTYPE_U32 ? "u32" : "unsupported";
}

const char *cpu_benchmark_operation_name(cpu_benchmark_operation operation) {
    if (operation < CPU_BENCHMARK_ZERO || operation >= CPU_BENCHMARK_OPERATION_COUNT) {
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

int runtime_benchmark_operation_supported(runtime_benchmark_backend backend,
                                          cpu_benchmark_operation operation) {
    if (operation < CPU_BENCHMARK_ZERO || operation >= CPU_BENCHMARK_OPERATION_COUNT) {
        return 0;
    }
    if (backend == RUNTIME_BENCHMARK_CPU) {
        return 1;
    }
    if (backend != RUNTIME_BENCHMARK_METAL && backend != RUNTIME_BENCHMARK_CUDA) {
        return 0;
    }
    switch (operation) {
    case CPU_BENCHMARK_ZERO:
    case CPU_BENCHMARK_FILL:
    case CPU_BENCHMARK_COPY:
    case CPU_BENCHMARK_ADD:
    case CPU_BENCHMARK_MULTIPLY:
    case CPU_BENCHMARK_SCALE:
    case CPU_BENCHMARK_REDUCE_SUM:
    case CPU_BENCHMARK_REDUCE_MAX:
    case CPU_BENCHMARK_REDUCE_MEAN_SQUARE:
    case CPU_BENCHMARK_MATMUL:
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT:
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT:
    case CPU_BENCHMARK_ACCUMULATE:
    case CPU_BENCHMARK_GATHER:
    case CPU_BENCHMARK_SCATTER_ADD:
    case CPU_BENCHMARK_SILU:
    case CPU_BENCHMARK_SILU_BACKWARD:
    case CPU_BENCHMARK_RMS_NORM:
    case CPU_BENCHMARK_RMS_NORM_BACKWARD:
    case CPU_BENCHMARK_ROPE:
    case CPU_BENCHMARK_ROPE_BACKWARD:
    case CPU_BENCHMARK_ATTENTION:
    case CPU_BENCHMARK_ATTENTION_BACKWARD:
    case CPU_BENCHMARK_SOFTMAX:
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
    case CPU_BENCHMARK_ADAMW:
        return 1;
    case CPU_BENCHMARK_OPERATION_COUNT:
        return 0;
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
#if defined(CLOCK_MONOTONIC)
    struct timespec time = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        return 0.0;
    }
    return (double)time.tv_sec + (double)time.tv_nsec / 1000000000.0;
#else
    struct timespec time = {0};
    if (timespec_get(&time, TIME_UTC) != TIME_UTC) {
        return 0.0;
    }
    return (double)time.tv_sec + (double)time.tv_nsec / 1000000000.0;
#endif
}

static int benchmark_uses_batch(runtime_benchmark_backend backend,
                                const cpu_benchmark_config *config) {
    return config->batch_accelerator_operations != 0 && backend != RUNTIME_BENCHMARK_CPU;
}

static llm_status benchmark_begin_batch(cpu_benchmark_workload *workload,
                                        const cpu_benchmark_config *config) {
    return benchmark_uses_batch(workload->backend_kind, config) != 0
               ? llm_backend_begin_batch(workload->backend)
               : LLM_OK;
}

static llm_status benchmark_end_batch(cpu_benchmark_workload *workload, int batch_started,
                                      llm_status status) {
    if (batch_started == 0) {
        return status;
    }
    const llm_status batch_status = llm_backend_end_batch(workload->backend);
    return status == LLM_OK ? batch_status : status;
}

static int compare_double(const void *left, const void *right) {
    const double left_value = *(const double *)left;
    const double right_value = *(const double *)right;
    return (left_value > right_value) - (left_value < right_value);
}

static llm_tensor *workload_tensor(cpu_benchmark_workload *workload,
                                   enum benchmark_tensor_slot slot) {
    return &workload->tensors[(size_t)slot];
}

static void workload_destroy(cpu_benchmark_workload *workload) {
    for (size_t remaining = BENCHMARK_TENSOR_COUNT; remaining > 0U; --remaining) {
        llm_tensor_destroy(&workload->tensors[remaining - 1U]);
    }
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
    if (upper_bound == 0U || upper_bound > UINT32_MAX || count > SIZE_MAX / sizeof(uint32_t)) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t shape[] = {count};
    llm_status status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, shape, out_tensor);
    if (status != LLM_OK) {
        return status;
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
    llm_status status = LLM_OK;
    switch (workload->operation) {
    case CPU_BENCHMARK_ZERO:
    case CPU_BENCHMARK_FILL:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_OUTPUT), 0.75F);
        break;
    case CPU_BENCHMARK_COPY:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 1.25F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_ADD:
    case CPU_BENCHMARK_MULTIPLY:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 1.25F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_SECOND), 2.5F);
        }
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_SCALE:
    case CPU_BENCHMARK_SILU:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.5F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_ACCUMULATE:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.0001F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.5F);
        }
        break;
    case CPU_BENCHMARK_SILU_BACKWARD:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.5F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_SECOND), 0.25F);
        }
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_ADAMW:
        status = create_f32_tensor(workload->backend, 1U, shape,
                                   workload_tensor(workload, BENCHMARK_OUTPUT), 1.0F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_FIRST), 0.01F);
        }
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_SECOND), 0.0F);
        }
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, shape,
                                       workload_tensor(workload, BENCHMARK_THIRD), 0.0F);
        }
        break;
    default:
        status = LLM_INVALID_ARGUMENT;
        break;
    }
    return status;
}

static llm_status setup_matrix_workload(cpu_benchmark_workload *workload) {
    const size_t matrix_shape[] = {workload->rows, workload->columns};
    const size_t row_shape[] = {workload->rows};
    const size_t weight_shape[] = {workload->columns};
    llm_status status = LLM_OK;
    switch (workload->operation) {
    case CPU_BENCHMARK_REDUCE_SUM:
    case CPU_BENCHMARK_REDUCE_MAX:
    case CPU_BENCHMARK_REDUCE_MEAN_SQUARE:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.25F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, row_shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_GATHER:
    case CPU_BENCHMARK_SCATTER_ADD:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.0001F);
        if (status == LLM_OK) {
            status = create_indices(workload->backend, workload->rows, workload->rows,
                                    workload_tensor(workload, BENCHMARK_SECOND));
        }
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_SOFTMAX:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.0F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.0F);
        if (status == LLM_OK) {
            status = create_indices(workload->backend, workload->rows, workload->columns,
                                    workload_tensor(workload, BENCHMARK_SECOND));
        }
        if (status == LLM_OK && workload->operation == CPU_BENCHMARK_CROSS_ENTROPY_FORWARD) {
            status = create_f32_tensor(workload->backend, 0U, NULL,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        } else if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        break;
    case CPU_BENCHMARK_RMS_NORM:
    case CPU_BENCHMARK_RMS_NORM_BACKWARD:
        status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                   workload_tensor(workload, BENCHMARK_FIRST), 0.5F);
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 1U, weight_shape,
                                       workload_tensor(workload, BENCHMARK_SECOND), 1.0F);
        }
        if (status == LLM_OK && workload->operation == CPU_BENCHMARK_RMS_NORM_BACKWARD) {
            status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                       workload_tensor(workload, BENCHMARK_THIRD), 0.25F);
        }
        if (status == LLM_OK) {
            status = create_f32_tensor(workload->backend, 2U, matrix_shape,
                                       workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
        }
        if (status == LLM_OK && workload->operation == CPU_BENCHMARK_RMS_NORM_BACKWARD) {
            status = create_f32_tensor(workload->backend, 1U, weight_shape,
                                       workload_tensor(workload, BENCHMARK_FOURTH), 0.0F);
        }
        break;
    default:
        status = LLM_INVALID_ARGUMENT;
        break;
    }
    return status;
}

static llm_status setup_matmul_workload(cpu_benchmark_workload *workload) {
    size_t left_shape[] = {workload->rows, workload->inner_size};
    size_t right_shape[] = {workload->inner_size, workload->columns};
    const size_t output_shape[] = {workload->rows, workload->columns};
    if (workload->operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT) {
        left_shape[0] = workload->inner_size;
        left_shape[1] = workload->rows;
    } else if (workload->operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT) {
        right_shape[0] = workload->columns;
        right_shape[1] = workload->inner_size;
    }
    llm_status status = LLM_OK;
    status = create_f32_tensor(workload->backend, 2U, left_shape,
                               workload_tensor(workload, BENCHMARK_FIRST), 0.25F);
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 2U, right_shape,
                                   workload_tensor(workload, BENCHMARK_SECOND), -0.125F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 2U, output_shape,
                                   workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
    }
    return status;
}

static llm_status setup_rope_workload(cpu_benchmark_workload *workload) {
    const size_t input_shape[] = {workload->batch_size, workload->sequence_length,
                                  workload->query_head_count, workload->head_dimension};
    const size_t table_shape[] = {workload->sequence_length, workload->head_dimension / 2U};
    llm_status status = create_f32_tensor(workload->backend, 4U, input_shape,
                                          workload_tensor(workload, BENCHMARK_FIRST), 0.5F);
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 2U, table_shape,
                                   workload_tensor(workload, BENCHMARK_SECOND), 1.0F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 2U, table_shape,
                                   workload_tensor(workload, BENCHMARK_THIRD), 0.0F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 4U, input_shape,
                                   workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
    }
    return status;
}

static llm_status setup_attention_workload(cpu_benchmark_workload *workload) {
    const size_t query_shape[] = {workload->batch_size, workload->sequence_length,
                                  workload->query_head_count, workload->head_dimension};
    const size_t key_value_shape[] = {workload->batch_size, workload->sequence_length,
                                      workload->key_value_head_count, workload->head_dimension};
    llm_status status = create_f32_tensor(workload->backend, 4U, query_shape,
                                          workload_tensor(workload, BENCHMARK_FIRST), 0.125F);
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 4U, key_value_shape,
                                   workload_tensor(workload, BENCHMARK_SECOND), 0.125F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 4U, key_value_shape,
                                   workload_tensor(workload, BENCHMARK_THIRD), 0.25F);
    }
    if (status == LLM_OK && workload->operation == CPU_BENCHMARK_ATTENTION_BACKWARD) {
        status = create_f32_tensor(workload->backend, 4U, query_shape,
                                   workload_tensor(workload, BENCHMARK_FOURTH), 0.125F);
    }
    if (status == LLM_OK) {
        status = create_f32_tensor(workload->backend, 4U, query_shape,
                                   workload_tensor(workload, BENCHMARK_OUTPUT), 0.0F);
    }
    if (status == LLM_OK && workload->operation == CPU_BENCHMARK_ATTENTION_BACKWARD) {
        status = create_f32_tensor(workload->backend, 4U, key_value_shape,
                                   workload_tensor(workload, BENCHMARK_FIFTH), 0.0F);
    }
    if (status == LLM_OK && workload->operation == CPU_BENCHMARK_ATTENTION_BACKWARD) {
        status = create_f32_tensor(workload->backend, 4U, key_value_shape,
                                   workload_tensor(workload, BENCHMARK_SIXTH), 0.0F);
    }
    return status;
}

static int operation_is_vector(cpu_benchmark_operation operation) {
    switch (operation) {
    case CPU_BENCHMARK_ZERO:
    case CPU_BENCHMARK_FILL:
    case CPU_BENCHMARK_COPY:
    case CPU_BENCHMARK_ADD:
    case CPU_BENCHMARK_MULTIPLY:
    case CPU_BENCHMARK_SCALE:
    case CPU_BENCHMARK_ACCUMULATE:
    case CPU_BENCHMARK_SILU:
    case CPU_BENCHMARK_SILU_BACKWARD:
    case CPU_BENCHMARK_ADAMW:
        return 1;
    default:
        return 0;
    }
}

static int operation_is_matrix(cpu_benchmark_operation operation) {
    switch (operation) {
    case CPU_BENCHMARK_REDUCE_SUM:
    case CPU_BENCHMARK_REDUCE_MAX:
    case CPU_BENCHMARK_REDUCE_MEAN_SQUARE:
    case CPU_BENCHMARK_GATHER:
    case CPU_BENCHMARK_SCATTER_ADD:
    case CPU_BENCHMARK_RMS_NORM:
    case CPU_BENCHMARK_RMS_NORM_BACKWARD:
    case CPU_BENCHMARK_SOFTMAX:
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        return 1;
    default:
        return 0;
    }
}

static llm_status workload_setup(runtime_benchmark_backend backend_kind,
                                 cpu_benchmark_operation operation, size_t requested_threads,
                                 const cpu_benchmark_config *config,
                                 cpu_benchmark_workload *out_workload) {
    *out_workload = (cpu_benchmark_workload){
        .operation = operation,
        .backend_kind = backend_kind,
        .elements = config->elements,
        .rows = config->rows,
        .columns = config->columns,
        .inner_size = config->inner_size,
        .batch_size = config->batch_size,
        .sequence_length = config->sequence_length,
        .query_head_count = config->query_head_count,
        .key_value_head_count = config->key_value_head_count,
        .head_dimension = config->head_dimension,
    };
    llm_status status = LLM_INVALID_ARGUMENT;
    if (backend_kind == RUNTIME_BENCHMARK_CPU) {
        const llm_cpu_backend_config backend_config = {
            .thread_count = requested_threads,
        };
        status = llm_backend_cpu_create_with_config(&backend_config, &out_workload->backend);
    } else if (backend_kind == RUNTIME_BENCHMARK_METAL) {
        status = llm_backend_metal_create(&out_workload->backend);
    } else if (backend_kind == RUNTIME_BENCHMARK_CUDA) {
        status = llm_backend_cuda_create(&out_workload->backend);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (operation_is_vector(operation) != 0) {
        status = setup_vector_workload(out_workload);
    } else if (operation_is_matrix(operation) != 0) {
        status = setup_matrix_workload(out_workload);
    } else if (operation == CPU_BENCHMARK_MATMUL ||
               operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT ||
               operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT) {
        status = setup_matmul_workload(out_workload);
    } else if (operation == CPU_BENCHMARK_ROPE || operation == CPU_BENCHMARK_ROPE_BACKWARD) {
        status = setup_rope_workload(out_workload);
    } else if (operation == CPU_BENCHMARK_ATTENTION ||
               operation == CPU_BENCHMARK_ATTENTION_BACKWARD) {
        status = setup_attention_workload(out_workload);
    }
    if (status != LLM_OK) {
        workload_destroy(out_workload);
    }
    return status;
}

static llm_status workload_execute(cpu_benchmark_workload *workload) {
    llm_tensor *first = workload_tensor(workload, BENCHMARK_FIRST);
    llm_tensor *second = workload_tensor(workload, BENCHMARK_SECOND);
    llm_tensor *third = workload_tensor(workload, BENCHMARK_THIRD);
    llm_tensor *fourth = workload_tensor(workload, BENCHMARK_FOURTH);
    llm_tensor *fifth = workload_tensor(workload, BENCHMARK_FIFTH);
    llm_tensor *sixth = workload_tensor(workload, BENCHMARK_SIXTH);
    llm_tensor *output = workload_tensor(workload, BENCHMARK_OUTPUT);
    switch (workload->operation) {
    case CPU_BENCHMARK_ZERO:
        return llm_tensor_zero(workload->backend, output);
    case CPU_BENCHMARK_FILL:
        return llm_tensor_fill_f32(workload->backend, output, 0.75F);
    case CPU_BENCHMARK_COPY:
        return llm_tensor_copy(workload->backend, first, output);
    case CPU_BENCHMARK_ADD:
        return llm_add(workload->backend, first, second, output);
    case CPU_BENCHMARK_MULTIPLY:
        return llm_multiply(workload->backend, first, second, output);
    case CPU_BENCHMARK_SCALE:
        return llm_scale(workload->backend, first, 0.5F, output);
    case CPU_BENCHMARK_ACCUMULATE:
        return llm_accumulate(workload->backend, first, output);
    case CPU_BENCHMARK_REDUCE_SUM:
        return llm_reduce_sum_last(workload->backend, first, output);
    case CPU_BENCHMARK_REDUCE_MAX:
        return llm_reduce_max_last(workload->backend, first, output);
    case CPU_BENCHMARK_REDUCE_MEAN_SQUARE:
        return llm_reduce_mean_square_last(workload->backend, first, output);
    case CPU_BENCHMARK_MATMUL:
        return llm_matmul(workload->backend, first, second, output);
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT: {
        const llm_matmul_options options = {.transpose_left = 1, .transpose_right = 0};
        return llm_matmul_ex(workload->backend, first, second, &options, output);
    }
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT: {
        const llm_matmul_options options = {.transpose_left = 0, .transpose_right = 1};
        return llm_matmul_ex(workload->backend, first, second, &options, output);
    }
    case CPU_BENCHMARK_GATHER:
        return llm_gather_rows(workload->backend, first, second, output);
    case CPU_BENCHMARK_SCATTER_ADD:
        return llm_scatter_add_rows(workload->backend, first, second, output);
    case CPU_BENCHMARK_SILU:
        return llm_silu(workload->backend, first, output);
    case CPU_BENCHMARK_SILU_BACKWARD:
        return llm_silu_backward(workload->backend, first, second, output);
    case CPU_BENCHMARK_RMS_NORM:
        return llm_rms_norm(workload->backend, first, second, 1.0e-5F, output);
    case CPU_BENCHMARK_RMS_NORM_BACKWARD:
        return llm_rms_norm_backward(workload->backend, first, second, third, 1.0e-5F, output,
                                     fourth);
    case CPU_BENCHMARK_ROPE:
        return llm_rope(workload->backend, first, second, third, output);
    case CPU_BENCHMARK_ROPE_BACKWARD:
        return llm_rope_backward(workload->backend, first, second, third, output);
    case CPU_BENCHMARK_ATTENTION: {
        const llm_attention_options options = {
            .scale = 1.0F / sqrtf((float)workload->head_dimension),
        };
        return llm_attention_forward(workload->backend, first, second, third, &options, output);
    }
    case CPU_BENCHMARK_ATTENTION_BACKWARD: {
        const llm_attention_options options = {
            .scale = 1.0F / sqrtf((float)workload->head_dimension),
        };
        return llm_attention_backward(workload->backend, first, second, third, fourth, &options,
                                      output, fifth, sixth);
    }
    case CPU_BENCHMARK_SOFTMAX:
        return llm_softmax_last(workload->backend, first, output);
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
        return llm_cross_entropy_forward(workload->backend, first, second, output);
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        return llm_cross_entropy_backward(workload->backend, first, second, output);
    case CPU_BENCHMARK_ADAMW: {
        const llm_adamw_options options = {
            .learning_rate = 1.0e-5F,
            .beta1 = 0.9F,
            .beta2 = 0.999F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.01F,
            .gradient_scale = 1.0F,
            .step = 1ULL,
        };
        return llm_adamw_update(workload->backend, output, first, second, third, &options);
    }
    case CPU_BENCHMARK_OPERATION_COUNT:
        return LLM_INVALID_ARGUMENT;
    }
    return LLM_INVALID_ARGUMENT;
}

static int close_enough(float actual, float expected) {
    const float scale = fabsf(expected) > 1.0F ? fabsf(expected) : 1.0F;
    return fabsf(actual - expected) <= 1.0e-4F * scale;
}

static int workload_read_first_value(cpu_benchmark_workload *workload, float *out_value) {
    llm_tensor *result = workload_tensor(workload, BENCHMARK_OUTPUT);
    if (result->element_count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    const size_t byte_count = result->element_count * sizeof(float);
    float *values = malloc(byte_count);
    if (values == NULL) {
        return 0;
    }
    const int valid = llm_tensor_read(workload->backend, result, values, byte_count) == LLM_OK;
    if (valid != 0) {
        *out_value = values[0];
    }
    free(values);
    return valid;
}

static int workload_verify(cpu_benchmark_workload *workload, double *out_guard_value) {
    float actual = 0.0F;
    if (workload_read_first_value(workload, &actual) == 0 || !isfinite(actual)) {
        return 0;
    }
    float expected = 0.0F;
    int compare_expected = 1;
    switch (workload->operation) {
    case CPU_BENCHMARK_ZERO:
        expected = 0.0F;
        break;
    case CPU_BENCHMARK_FILL:
        expected = 0.75F;
        break;
    case CPU_BENCHMARK_COPY:
        expected = 1.25F;
        break;
    case CPU_BENCHMARK_ADD:
        expected = 3.75F;
        break;
    case CPU_BENCHMARK_MULTIPLY:
        expected = 3.125F;
        break;
    case CPU_BENCHMARK_SCALE:
        expected = 0.25F;
        break;
    case CPU_BENCHMARK_ACCUMULATE:
    case CPU_BENCHMARK_SCATTER_ADD:
    case CPU_BENCHMARK_ADAMW:
        compare_expected = 0;
        break;
    case CPU_BENCHMARK_REDUCE_SUM:
        expected = (float)workload->columns * 0.25F;
        break;
    case CPU_BENCHMARK_REDUCE_MAX:
        expected = 0.25F;
        break;
    case CPU_BENCHMARK_REDUCE_MEAN_SQUARE:
        expected = 0.0625F;
        break;
    case CPU_BENCHMARK_MATMUL:
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT:
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT:
        expected = (float)workload->inner_size * -0.03125F;
        break;
    case CPU_BENCHMARK_GATHER:
        expected = 0.0001F;
        break;
    case CPU_BENCHMARK_SILU:
        expected = 0.5F / (1.0F + expf(-0.5F));
        break;
    case CPU_BENCHMARK_SILU_BACKWARD: {
        const float sigmoid = 1.0F / (1.0F + expf(-0.5F));
        expected = 0.25F * sigmoid * (1.0F + 0.5F * (1.0F - sigmoid));
        break;
    }
    case CPU_BENCHMARK_RMS_NORM:
        expected = 0.5F / sqrtf(0.25F + 1.0e-5F);
        break;
    case CPU_BENCHMARK_RMS_NORM_BACKWARD: {
        const float inverse_rms = 1.0F / sqrtf(0.25F + 1.0e-5F);
        expected = 0.25F * inverse_rms - 0.0625F * inverse_rms * inverse_rms * inverse_rms;
        break;
    }
    case CPU_BENCHMARK_ROPE:
    case CPU_BENCHMARK_ROPE_BACKWARD:
        expected = 0.5F;
        break;
    case CPU_BENCHMARK_ATTENTION:
        expected = 0.25F;
        break;
    case CPU_BENCHMARK_ATTENTION_BACKWARD:
        expected = 0.0F;
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
        return 0;
    }
    *out_guard_value = (double)actual;
    return compare_expected == 0 || close_enough(actual, expected);
}

static double attention_pair_count(const cpu_benchmark_workload *workload) {
    return (double)workload->batch_size * (double)workload->query_head_count *
           (double)workload->sequence_length * (double)(workload->sequence_length + 1U) * 0.5;
}

static double workload_throughput(const cpu_benchmark_workload *workload, double seconds,
                                  const char **out_unit) {
    double work = 0.0;
    switch (workload->operation) {
    case CPU_BENCHMARK_ZERO:
    case CPU_BENCHMARK_FILL:
        work = (double)workload->elements * sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_COPY:
    case CPU_BENCHMARK_SCALE:
    case CPU_BENCHMARK_SILU:
    case CPU_BENCHMARK_SILU_BACKWARD:
        work = 2.0 * (double)workload->elements * sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_ADD:
    case CPU_BENCHMARK_MULTIPLY:
    case CPU_BENCHMARK_ACCUMULATE:
        work = 3.0 * (double)workload->elements * sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_REDUCE_SUM:
    case CPU_BENCHMARK_REDUCE_MAX:
    case CPU_BENCHMARK_REDUCE_MEAN_SQUARE:
        work = ((double)workload->rows * (double)workload->columns + (double)workload->rows) *
               sizeof(float);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_MATMUL:
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT:
    case CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT:
        work =
            2.0 * (double)workload->rows * (double)workload->inner_size * (double)workload->columns;
        *out_unit = "GFLOP/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_GATHER:
        work = (2.0 * (double)workload->rows * (double)workload->columns * sizeof(float)) +
               (double)workload->rows * sizeof(uint32_t);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_SCATTER_ADD:
        work = (3.0 * (double)workload->rows * (double)workload->columns * sizeof(float)) +
               (double)workload->rows * sizeof(uint32_t);
        *out_unit = "GB/s";
        return work / seconds / 1.0e9;
    case CPU_BENCHMARK_RMS_NORM:
    case CPU_BENCHMARK_RMS_NORM_BACKWARD:
    case CPU_BENCHMARK_SOFTMAX:
    case CPU_BENCHMARK_CROSS_ENTROPY_FORWARD:
    case CPU_BENCHMARK_CROSS_ENTROPY_BACKWARD:
        work = (double)workload->rows * (double)workload->columns;
        *out_unit = "M elements/s";
        return work / seconds / 1.0e6;
    case CPU_BENCHMARK_ROPE:
    case CPU_BENCHMARK_ROPE_BACKWARD:
        work = (double)workload->batch_size * (double)workload->sequence_length *
               (double)workload->query_head_count * (double)workload->head_dimension;
        *out_unit = "M elements/s";
        return work / seconds / 1.0e6;
    case CPU_BENCHMARK_ATTENTION:
    case CPU_BENCHMARK_ATTENTION_BACKWARD:
        work = attention_pair_count(workload);
        *out_unit = "M token-pairs/s";
        return work / seconds / 1.0e6;
    case CPU_BENCHMARK_ADAMW:
        work = (double)workload->elements;
        *out_unit = "M parameters/s";
        return work / seconds / 1.0e6;
    case CPU_BENCHMARK_OPERATION_COUNT:
        *out_unit = "unknown";
        return 0.0;
    }
    *out_unit = "unknown";
    return 0.0;
}

static int config_is_valid(const cpu_benchmark_config *config) {
    if (config == NULL || config->elements == 0U || config->rows == 0U || config->columns == 0U ||
        config->inner_size == 0U || config->batch_size == 0U || config->sequence_length == 0U ||
        config->query_head_count == 0U || config->key_value_head_count == 0U ||
        config->head_dimension == 0U || config->head_dimension % 2U != 0U ||
        config->query_head_count % config->key_value_head_count != 0U ||
        config->warmup_iterations == 0U || config->measured_iterations == 0U ||
        config->minimum_sample_seconds <= 0.0 ||
        config->measured_iterations > SIZE_MAX / sizeof(double)) {
        return 0;
    }
    size_t product = 0U;
    return checked_multiply(config->rows, config->columns, &product) != 0 &&
           checked_multiply(config->batch_size, config->sequence_length, &product) != 0;
}

int runtime_benchmark_run(runtime_benchmark_backend backend_kind, cpu_benchmark_operation operation,
                          size_t requested_threads, const cpu_benchmark_config *config,
                          cpu_benchmark_result *out_result) {
    if ((backend_kind != RUNTIME_BENCHMARK_CPU && backend_kind != RUNTIME_BENCHMARK_METAL &&
         backend_kind != RUNTIME_BENCHMARK_CUDA) ||
        operation < CPU_BENCHMARK_ZERO || operation >= CPU_BENCHMARK_OPERATION_COUNT ||
        out_result == NULL || config_is_valid(config) == 0 ||
        runtime_benchmark_operation_supported(backend_kind, operation) == 0) {
        return 0;
    }

    cpu_benchmark_workload workload = {0};
    const llm_status setup_status =
        workload_setup(backend_kind, operation, requested_threads, config, &workload);
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
        probe_status = benchmark_begin_batch(&workload, config);
        const int probe_batch_started =
            probe_status == LLM_OK && benchmark_uses_batch(workload.backend_kind, config) != 0;
        const double probe_start = current_seconds();
        for (size_t repetition = 0U; probe_status == LLM_OK && repetition < probe_repetitions;
             ++repetition) {
            probe_status = workload_execute(&workload);
        }
        probe_status = benchmark_end_batch(&workload, probe_batch_started, probe_status);
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
    double *gpu_durations = calloc(config->measured_iterations, sizeof(*gpu_durations));
    if (durations == NULL || gpu_durations == NULL) {
        free(gpu_durations);
        free(durations);
        workload_destroy(&workload);
        return 0;
    }
    double pipeline_compilation_seconds = 0.0;
    if (backend_kind == RUNTIME_BENCHMARK_METAL) {
        llm_metal_backend_metrics initial_metrics = {0};
        (void)llm_backend_metal_get_metrics(workload.backend, &initial_metrics);
        pipeline_compilation_seconds = initial_metrics.pipeline_compilation_seconds;
    }
    for (size_t iteration = 0U; iteration < config->measured_iterations; ++iteration) {
        if (backend_kind == RUNTIME_BENCHMARK_METAL) {
            (void)llm_backend_metal_reset_metrics(workload.backend);
        } else if (backend_kind == RUNTIME_BENCHMARK_CUDA) {
            (void)llm_backend_cuda_reset_metrics(workload.backend);
        }
        const double start = current_seconds();
        llm_status status = benchmark_begin_batch(&workload, config);
        const int batch_started =
            status == LLM_OK && benchmark_uses_batch(workload.backend_kind, config) != 0;
        for (size_t repetition = 0U; status == LLM_OK && repetition < repetitions_per_sample;
             ++repetition) {
            status = workload_execute(&workload);
        }
        status = benchmark_end_batch(&workload, batch_started, status);
        durations[iteration] = (current_seconds() - start) / (double)repetitions_per_sample;
        if (backend_kind == RUNTIME_BENCHMARK_METAL) {
            llm_metal_backend_metrics metrics = {0};
            if (llm_backend_metal_get_metrics(workload.backend, &metrics) != LLM_OK) {
                status = LLM_BACKEND_ERROR;
            } else {
                gpu_durations[iteration] =
                    metrics.total_gpu_seconds / (double)repetitions_per_sample;
            }
        } else if (backend_kind == RUNTIME_BENCHMARK_CUDA) {
            llm_cuda_backend_metrics metrics = {0};
            if (llm_backend_cuda_get_metrics(workload.backend, &metrics) != LLM_OK) {
                status = LLM_BACKEND_ERROR;
            } else {
                gpu_durations[iteration] =
                    metrics.total_gpu_seconds / (double)repetitions_per_sample;
            }
        }
        if (status != LLM_OK || durations[iteration] <= 0.0) {
            fprintf(stderr, "%s measurement failed: %s\n", cpu_benchmark_operation_name(operation),
                    llm_status_string(status));
            free(durations);
            free(gpu_durations);
            workload_destroy(&workload);
            return 0;
        }
    }

    double guard_value = 0.0;
    if (workload_verify(&workload, &guard_value) == 0) {
        fprintf(stderr, "%s produced an invalid result\n", cpu_benchmark_operation_name(operation));
        free(durations);
        free(gpu_durations);
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
    qsort(gpu_durations, config->measured_iterations, sizeof(*gpu_durations), compare_double);
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
        .backend = backend_kind,
        .dtype = LLM_DTYPE_F32,
        .requested_threads = requested_threads,
        .actual_threads = backend_kind == RUNTIME_BENCHMARK_CPU
                              ? llm_backend_cpu_thread_count(workload.backend)
                              : 0U,
        .repetitions_per_sample = repetitions_per_sample,
        .minimum_seconds = durations[0],
        .median_seconds = median,
        .p95_seconds = durations[p95_index],
        .mean_seconds = mean,
        .standard_deviation_seconds = sqrt(variance),
        .nanoseconds_per_call = median * 1.0e9,
        .calls_per_second = 1.0 / median,
        .guard_value = guard_value,
        .gpu_median_seconds =
            (backend_kind == RUNTIME_BENCHMARK_METAL || backend_kind == RUNTIME_BENCHMARK_CUDA)
                ? (config->measured_iterations % 2U == 0U
                       ? (gpu_durations[median_index - 1U] + gpu_durations[median_index]) * 0.5
                       : gpu_durations[median_index])
                : 0.0,
        .gpu_p95_seconds =
            (backend_kind == RUNTIME_BENCHMARK_METAL || backend_kind == RUNTIME_BENCHMARK_CUDA)
                ? gpu_durations[p95_index]
                : 0.0,
        .pipeline_compilation_seconds = pipeline_compilation_seconds,
    };
    const char *device_name =
        backend_kind == RUNTIME_BENCHMARK_METAL  ? llm_backend_metal_device_name(workload.backend)
        : backend_kind == RUNTIME_BENCHMARK_CUDA ? llm_backend_cuda_device_name(workload.backend)
                                                 : "CPU";
    if (device_name != NULL) {
        (void)snprintf(out_result->device_name, sizeof(out_result->device_name), "%s", device_name);
    }
    out_result->throughput =
        workload_throughput(&workload, out_result->median_seconds, &out_result->throughput_unit);

    free(durations);
    free(gpu_durations);
    workload_destroy(&workload);
    return 1;
}
