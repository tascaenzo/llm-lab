#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cpu_internal.h"
#include "cpu_kernels.h"

#define LLM_CPU_MIN_MEMORY_BYTES_PER_TASK (1024U * 1024U)
#define LLM_CPU_MIN_ELEMENTWISE_VALUES_PER_TASK 16384U
#define LLM_CPU_TARGET_ROW_VALUES_PER_TASK 16384U
#define LLM_CPU_TARGET_MATMUL_OPERATIONS_PER_TASK 262144U
#define LLM_CPU_SCATTER_COLUMNS_PER_TASK 256U

typedef struct cpu_zero_job {
    unsigned char *memory;
} cpu_zero_job;

typedef struct cpu_copy_job {
    const unsigned char *source;
    unsigned char *destination;
} cpu_copy_job;

typedef struct cpu_fill_job {
    float *values;
    float value;
} cpu_fill_job;

typedef struct cpu_binary_job {
    const float *left;
    const float *right;
    float *output;
} cpu_binary_job;

typedef struct cpu_scale_job {
    const float *input;
    float *output;
    float scale;
} cpu_scale_job;

typedef struct cpu_reduction_job {
    const float *input;
    float *output;
    size_t reduction_size;
} cpu_reduction_job;

typedef struct cpu_matmul_job {
    const float *left;
    const float *right;
    float *output;
    size_t inner_size;
    size_t columns;
} cpu_matmul_job;

typedef struct cpu_gather_job {
    const float *table;
    const uint32_t *indices;
    float *output;
    size_t row_count;
    size_t row_width;
} cpu_gather_job;

typedef struct cpu_scatter_job {
    const float *source;
    const uint32_t *indices;
    float *table;
    size_t index_count;
    size_t row_width;
} cpu_scatter_job;

typedef struct cpu_softmax_job {
    const float *input;
    float *output;
    size_t row_width;
} cpu_softmax_job;

typedef struct cpu_cross_entropy_backward_job {
    const float *logits;
    const uint32_t *targets;
    float *gradient;
    size_t vocabulary_size;
    size_t normalization_row_count;
} cpu_cross_entropy_backward_job;

static size_t divide_round_up(size_t value, size_t divisor) {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

static size_t rows_per_task(size_t row_width, size_t target_values) {
    if (row_width >= target_values) {
        return 1U;
    }
    return divide_round_up(target_values, row_width);
}

static llm_status zero_range(void *context, size_t begin, size_t end) {
    cpu_zero_job *job = context;
    (void)memset(job->memory + begin, 0, end - begin);
    return LLM_OK;
}

static llm_status copy_range(void *context, size_t begin, size_t end) {
    cpu_copy_job *job = context;
    (void)memcpy(job->destination + begin, job->source + begin, end - begin);
    return LLM_OK;
}

static llm_status fill_range(void *context, size_t begin, size_t end) {
    cpu_fill_job *job = context;
    for (size_t index = begin; index < end; ++index) {
        job->values[index] = job->value;
    }
    return LLM_OK;
}

static llm_status add_range(void *context, size_t begin, size_t end) {
    cpu_binary_job *job = context;
    return llm_cpu_add_f32(job->left + begin, job->right + begin, job->output + begin, end - begin);
}

static llm_status multiply_range(void *context, size_t begin, size_t end) {
    cpu_binary_job *job = context;
    return llm_cpu_multiply_f32(job->left + begin, job->right + begin, job->output + begin,
                                end - begin);
}

static llm_status scale_range(void *context, size_t begin, size_t end) {
    cpu_scale_job *job = context;
    return llm_cpu_scale_f32(job->input + begin, job->scale, job->output + begin, end - begin);
}

static llm_status reduce_sum_range(void *context, size_t begin, size_t end) {
    cpu_reduction_job *job = context;
    return llm_cpu_reduce_sum_last_f32(job->input + begin * job->reduction_size,
                                       job->output + begin, end - begin, job->reduction_size);
}

static llm_status reduce_max_range(void *context, size_t begin, size_t end) {
    cpu_reduction_job *job = context;
    return llm_cpu_reduce_max_last_f32(job->input + begin * job->reduction_size,
                                       job->output + begin, end - begin, job->reduction_size);
}

static llm_status reduce_mean_square_range(void *context, size_t begin, size_t end) {
    cpu_reduction_job *job = context;
    return llm_cpu_reduce_mean_square_last_f32(job->input + begin * job->reduction_size,
                                               job->output + begin, end - begin,
                                               job->reduction_size);
}

static llm_status matmul_range(void *context, size_t begin, size_t end) {
    cpu_matmul_job *job = context;
    return llm_cpu_matmul_f32(job->left + begin * job->inner_size, job->right,
                              job->output + begin * job->columns, end - begin, job->inner_size,
                              job->columns);
}

static llm_status gather_range(void *context, size_t begin, size_t end) {
    cpu_gather_job *job = context;
    return llm_cpu_gather_rows_f32(job->table, job->row_count, job->row_width, job->indices + begin,
                                   end - begin, job->output + begin * job->row_width);
}

static llm_status scatter_column_range(void *context, size_t begin, size_t end) {
    cpu_scatter_job *job = context;
    for (size_t index = 0U; index < job->index_count; ++index) {
        const size_t destination_row = (size_t)job->indices[index];
        const float *source_row = job->source + index * job->row_width;
        float *destination = job->table + destination_row * job->row_width;
        for (size_t column = begin; column < end; ++column) {
            destination[column] += source_row[column];
        }
    }
    return LLM_OK;
}

static llm_status softmax_range(void *context, size_t begin, size_t end) {
    cpu_softmax_job *job = context;
    return llm_cpu_softmax_last_f32(job->input + begin * job->row_width,
                                    job->output + begin * job->row_width, end - begin,
                                    job->row_width);
}

static llm_status cross_entropy_backward_range(void *context, size_t begin, size_t end) {
    cpu_cross_entropy_backward_job *job = context;
    return llm_cpu_cross_entropy_backward_f32(job->logits + begin * job->vocabulary_size,
                                              job->targets + begin, end - begin,
                                              job->vocabulary_size, job->normalization_row_count,
                                              job->gradient + begin * job->vocabulary_size);
}

llm_status llm_cpu_execute_zero(void *context, void *memory, size_t byte_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || memory == NULL || byte_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_zero_job job = {.memory = memory};
    return llm_cpu_parallel_for(cpu->executor, byte_count, LLM_CPU_MIN_MEMORY_BYTES_PER_TASK,
                                zero_range, &job);
}

llm_status llm_cpu_execute_copy(void *context, const void *source, void *destination,
                                size_t byte_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || source == NULL || destination == NULL || byte_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_copy_job job = {.source = source, .destination = destination};
    return llm_cpu_parallel_for(cpu->executor, byte_count, LLM_CPU_MIN_MEMORY_BYTES_PER_TASK,
                                copy_range, &job);
}

llm_status llm_cpu_execute_fill_f32(void *context, float *values, size_t value_count, float value) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || values == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_fill_job job = {.values = values, .value = value};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_MIN_ELEMENTWISE_VALUES_PER_TASK,
                                fill_range, &job);
}

llm_status llm_cpu_execute_add_f32(void *context, const float *left, const float *right,
                                   float *output, size_t value_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || left == NULL || right == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_binary_job job = {.left = left, .right = right, .output = output};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_MIN_ELEMENTWISE_VALUES_PER_TASK,
                                add_range, &job);
}

llm_status llm_cpu_execute_multiply_f32(void *context, const float *left, const float *right,
                                        float *output, size_t value_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || left == NULL || right == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_binary_job job = {.left = left, .right = right, .output = output};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_MIN_ELEMENTWISE_VALUES_PER_TASK,
                                multiply_range, &job);
}

llm_status llm_cpu_execute_scale_f32(void *context, const float *input, float scale, float *output,
                                     size_t value_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || input == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_scale_job job = {.input = input, .output = output, .scale = scale};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_MIN_ELEMENTWISE_VALUES_PER_TASK,
                                scale_range, &job);
}

static llm_status execute_reduction(llm_cpu_context *cpu, const float *input, float *output,
                                    size_t outer_count, size_t reduction_size,
                                    llm_cpu_range_fn function) {
    if (cpu == NULL || input == NULL || output == NULL || outer_count == 0U ||
        reduction_size == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_reduction_job job = {
        .input = input,
        .output = output,
        .reduction_size = reduction_size,
    };
    return llm_cpu_parallel_for(cpu->executor, outer_count,
                                rows_per_task(reduction_size, LLM_CPU_TARGET_ROW_VALUES_PER_TASK),
                                function, &job);
}

llm_status llm_cpu_execute_reduce_sum_last_f32(void *context, const float *input, float *output,
                                               size_t outer_count, size_t reduction_size) {
    return execute_reduction(context, input, output, outer_count, reduction_size, reduce_sum_range);
}

llm_status llm_cpu_execute_reduce_max_last_f32(void *context, const float *input, float *output,
                                               size_t outer_count, size_t reduction_size) {
    return execute_reduction(context, input, output, outer_count, reduction_size, reduce_max_range);
}

llm_status llm_cpu_execute_reduce_mean_square_last_f32(void *context, const float *input,
                                                       float *output, size_t outer_count,
                                                       size_t reduction_size) {
    return execute_reduction(context, input, output, outer_count, reduction_size,
                             reduce_mean_square_range);
}

llm_status llm_cpu_execute_matmul_f32(void *context, const float *left, const float *right,
                                      float *output, size_t rows, size_t inner_size,
                                      size_t columns) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || left == NULL || right == NULL || output == NULL || rows == 0U ||
        inner_size == 0U || columns == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_matmul_job job = {
        .left = left,
        .right = right,
        .output = output,
        .inner_size = inner_size,
        .columns = columns,
    };
    const size_t operations_per_row =
        inner_size > SIZE_MAX / columns ? SIZE_MAX : inner_size * columns;
    return llm_cpu_parallel_for(
        cpu->executor, rows,
        rows_per_task(operations_per_row, LLM_CPU_TARGET_MATMUL_OPERATIONS_PER_TASK), matmul_range,
        &job);
}

llm_status llm_cpu_execute_gather_rows_f32(void *context, const float *table, size_t row_count,
                                           size_t row_width, const uint32_t *indices,
                                           size_t index_count, float *output) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || table == NULL || indices == NULL || output == NULL || row_count == 0U ||
        row_width == 0U || index_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t index = 0U; index < index_count; ++index) {
        if ((size_t)indices[index] >= row_count) {
            return LLM_INVALID_INDEX;
        }
    }
    cpu_gather_job job = {
        .table = table,
        .indices = indices,
        .output = output,
        .row_count = row_count,
        .row_width = row_width,
    };
    return llm_cpu_parallel_for(cpu->executor, index_count,
                                rows_per_task(row_width, LLM_CPU_TARGET_ROW_VALUES_PER_TASK),
                                gather_range, &job);
}

llm_status llm_cpu_execute_scatter_add_rows_f32(void *context, const float *source,
                                                const uint32_t *indices, size_t index_count,
                                                size_t row_width, size_t row_count, float *table) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || source == NULL || indices == NULL || table == NULL || index_count == 0U ||
        row_width == 0U || row_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t index = 0U; index < index_count; ++index) {
        if ((size_t)indices[index] >= row_count) {
            return LLM_INVALID_INDEX;
        }
    }
    cpu_scatter_job job = {
        .source = source,
        .indices = indices,
        .table = table,
        .index_count = index_count,
        .row_width = row_width,
    };
    return llm_cpu_parallel_for(cpu->executor, row_width, LLM_CPU_SCATTER_COLUMNS_PER_TASK,
                                scatter_column_range, &job);
}

llm_status llm_cpu_execute_softmax_last_f32(void *context, const float *input, float *output,
                                            size_t outer_count, size_t row_width) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || input == NULL || output == NULL || outer_count == 0U || row_width == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_softmax_job job = {.input = input, .output = output, .row_width = row_width};
    return llm_cpu_parallel_for(cpu->executor, outer_count,
                                rows_per_task(row_width, LLM_CPU_TARGET_ROW_VALUES_PER_TASK),
                                softmax_range, &job);
}

llm_status llm_cpu_execute_cross_entropy_forward_f32(void *context, const float *logits,
                                                     const uint32_t *targets, size_t row_count,
                                                     size_t vocabulary_size, float *loss) {
    if (context == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    return llm_cpu_cross_entropy_forward_f32(logits, targets, row_count, vocabulary_size, loss);
}

llm_status llm_cpu_execute_cross_entropy_backward_f32(void *context, const float *logits,
                                                      const uint32_t *targets, size_t row_count,
                                                      size_t vocabulary_size, float *gradient) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || logits == NULL || targets == NULL || gradient == NULL || row_count == 0U ||
        vocabulary_size == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row = 0U; row < row_count; ++row) {
        if ((size_t)targets[row] >= vocabulary_size) {
            return LLM_INVALID_INDEX;
        }
    }
    cpu_cross_entropy_backward_job job = {
        .logits = logits,
        .targets = targets,
        .gradient = gradient,
        .vocabulary_size = vocabulary_size,
        .normalization_row_count = row_count,
    };
    return llm_cpu_parallel_for(cpu->executor, row_count,
                                rows_per_task(vocabulary_size, LLM_CPU_TARGET_ROW_VALUES_PER_TASK),
                                cross_entropy_backward_range, &job);
}

static uint32_t cpu_float_bits(float value) {
    uint32_t bits = 0U;
    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float cpu_float_from_bits(uint32_t bits) {
    float value = 0.0F;
    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

static float cpu_bf16_to_float(uint16_t value) {
    return cpu_float_from_bits((uint32_t)value << 16U);
}

static uint16_t cpu_float_to_bf16(float value) {
    uint32_t bits = cpu_float_bits(value);
    const uint32_t absolute = bits & 0x7FFFFFFFU;
    if (absolute >= 0x7F800000U) {
        return (uint16_t)((bits >> 16U) | (absolute > 0x7F800000U ? 0x0040U : 0U));
    }
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return (uint16_t)(bits >> 16U);
}

static float cpu_f16_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000U) << 16U;
    uint32_t exponent = ((uint32_t)value >> 10U) & 0x1FU;
    uint32_t mantissa = (uint32_t)value & 0x03FFU;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            return cpu_float_from_bits(sign);
        }
        exponent = 113U;
        while ((mantissa & 0x0400U) == 0U) {
            mantissa <<= 1U;
            --exponent;
        }
        mantissa &= 0x03FFU;
    } else if (exponent == 31U) {
        exponent = 255U;
    } else {
        exponent += 112U;
    }
    return cpu_float_from_bits(sign | (exponent << 23U) | (mantissa << 13U));
}

static uint16_t cpu_float_to_f16(float value) {
    const uint32_t bits = cpu_float_bits(value);
    const uint16_t sign = (uint16_t)((bits >> 16U) & 0x8000U);
    const uint32_t exponent_bits = (bits >> 23U) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent_bits == 255U) {
        return (uint16_t)(sign | (mantissa == 0U ? 0x7C00U : 0x7E00U));
    }
    const int exponent = (int)exponent_bits - 127 + 15;
    if (exponent >= 31) {
        return (uint16_t)(sign | 0x7C00U);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        uint32_t subnormal = (mantissa | 0x800000U) >> (uint32_t)(1 - exponent);
        subnormal = (subnormal + 0x0FFFU + ((subnormal >> 13U) & 1U)) >> 13U;
        return (uint16_t)(sign | (uint16_t)subnormal);
    }
    const uint32_t rounded = mantissa + 0x0FFFU + ((mantissa >> 13U) & 1U);
    if ((rounded & 0x800000U) != 0U) {
        return (uint16_t)(sign | (uint16_t)((exponent + 1) << 10U));
    }
    return (uint16_t)(sign | (uint16_t)((uint32_t)exponent << 10U) | (uint16_t)(rounded >> 13U));
}

llm_status llm_cpu_execute_cast(void *context, const void *input, llm_dtype input_dtype,
                                void *output, llm_dtype output_dtype, size_t value_count) {
    if (context == NULL || input == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    if (input_dtype == LLM_DTYPE_F32 && output_dtype == LLM_DTYPE_F16) {
        const float *source = input;
        uint16_t *destination = output;
        for (size_t index = 0U; index < value_count; ++index) {
            destination[index] = cpu_float_to_f16(source[index]);
        }
        return LLM_OK;
    }
    if (input_dtype == LLM_DTYPE_F16 && output_dtype == LLM_DTYPE_F32) {
        const uint16_t *source = input;
        float *destination = output;
        for (size_t index = 0U; index < value_count; ++index) {
            destination[index] = cpu_f16_to_float(source[index]);
        }
        return LLM_OK;
    }
    if (input_dtype == LLM_DTYPE_F32 && output_dtype == LLM_DTYPE_BF16) {
        const float *source = input;
        uint16_t *destination = output;
        for (size_t index = 0U; index < value_count; ++index) {
            destination[index] = cpu_float_to_bf16(source[index]);
        }
        return LLM_OK;
    }
    if (input_dtype == LLM_DTYPE_BF16 && output_dtype == LLM_DTYPE_F32) {
        const uint16_t *source = input;
        float *destination = output;
        for (size_t index = 0U; index < value_count; ++index) {
            destination[index] = cpu_bf16_to_float(source[index]);
        }
        return LLM_OK;
    }
    return LLM_UNSUPPORTED_DTYPE;
}

llm_status llm_cpu_execute_matmul_mixed_f32(void *context, const void *left, const void *right,
                                            llm_dtype input_dtype, float *output, size_t rows,
                                            size_t inner_size, size_t columns) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || left == NULL || right == NULL || output == NULL || rows == 0U ||
        inner_size == 0U || columns == 0U ||
        (input_dtype != LLM_DTYPE_F16 && input_dtype != LLM_DTYPE_BF16)) {
        return LLM_INVALID_ARGUMENT;
    }
    if (rows > SIZE_MAX / inner_size || inner_size > SIZE_MAX / columns) {
        return LLM_OVERFLOW;
    }
    const size_t left_count = rows * inner_size;
    const size_t right_count = inner_size * columns;
    if (left_count > SIZE_MAX - right_count ||
        left_count + right_count > SIZE_MAX / sizeof(float)) {
        return LLM_OVERFLOW;
    }
    float *packed = malloc((left_count + right_count) * sizeof(*packed));
    if (packed == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    float *packed_left = packed;
    float *packed_right = packed + left_count;
    const uint16_t *left_reduced = left;
    const uint16_t *right_reduced = right;
    for (size_t index = 0U; index < left_count; ++index) {
        packed_left[index] = input_dtype == LLM_DTYPE_F16 ? cpu_f16_to_float(left_reduced[index])
                                                          : cpu_bf16_to_float(left_reduced[index]);
    }
    for (size_t index = 0U; index < right_count; ++index) {
        packed_right[index] = input_dtype == LLM_DTYPE_F16
                                  ? cpu_f16_to_float(right_reduced[index])
                                  : cpu_bf16_to_float(right_reduced[index]);
    }
    const llm_status status = llm_cpu_execute_matmul_f32(cpu, packed_left, packed_right, output,
                                                         rows, inner_size, columns);
    free(packed);
    return status;
}
