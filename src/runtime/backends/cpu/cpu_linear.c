#include <math.h>

#include "cpu_internal.h"

#define LLM_CPU_LINEAR_VALUES_PER_TASK 16384U
#define LLM_CPU_MATMUL_ROWS_PER_TASK 4U

typedef struct cpu_accumulate_job {
    const float *source;
    float *destination;
} cpu_accumulate_job;

typedef struct cpu_matmul_ex_job {
    const float *left;
    const float *right;
    float *output;
    size_t left_columns;
    size_t right_columns;
    size_t inner_size;
    size_t output_columns;
    int transpose_left;
    int transpose_right;
} cpu_matmul_ex_job;

static llm_status accumulate_range(void *context, size_t begin, size_t end) {
    cpu_accumulate_job *job = context;
    for (size_t index = begin; index < end; ++index) {
        const float value = job->destination[index] + job->source[index];
        if (!isfinite(value)) {
            return LLM_NUMERICAL_ERROR;
        }
        job->destination[index] = value;
    }
    return LLM_OK;
}

static llm_status matmul_ex_range(void *context, size_t begin, size_t end) {
    cpu_matmul_ex_job *job = context;
    for (size_t row = begin; row < end; ++row) {
        for (size_t column = 0U; column < job->output_columns; ++column) {
            float sum = 0.0F;
            for (size_t inner = 0U; inner < job->inner_size; ++inner) {
                const size_t left_index = job->transpose_left != 0
                                              ? inner * job->left_columns + row
                                              : row * job->left_columns + inner;
                const size_t right_index = job->transpose_right != 0
                                               ? column * job->right_columns + inner
                                               : inner * job->right_columns + column;
                const float left_value = job->left[left_index];
                const float right_value = job->right[right_index];
                if (!isfinite(left_value) || !isfinite(right_value)) {
                    return LLM_NUMERICAL_ERROR;
                }
                sum += left_value * right_value;
            }
            if (!isfinite(sum)) {
                return LLM_NUMERICAL_ERROR;
            }
            job->output[row * job->output_columns + column] = sum;
        }
    }
    return LLM_OK;
}

llm_status llm_cpu_execute_accumulate_f32(void *context, const float *source, float *destination,
                                          size_t value_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || source == NULL || destination == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_accumulate_job job = {.source = source, .destination = destination};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_LINEAR_VALUES_PER_TASK,
                                accumulate_range, &job);
}

llm_status llm_cpu_execute_matmul_ex_f32(void *context, const float *left, const float *right,
                                         float *output, size_t left_rows, size_t left_columns,
                                         size_t right_rows, size_t right_columns,
                                         int transpose_left, int transpose_right) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || left == NULL || right == NULL || output == NULL || left_rows == 0U ||
        left_columns == 0U || right_rows == 0U || right_columns == 0U ||
        (transpose_left != 0 && transpose_left != 1) ||
        (transpose_right != 0 && transpose_right != 1)) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t output_rows = transpose_left != 0 ? left_columns : left_rows;
    const size_t inner_size = transpose_left != 0 ? left_rows : left_columns;
    const size_t right_inner = transpose_right != 0 ? right_columns : right_rows;
    const size_t output_columns = transpose_right != 0 ? right_rows : right_columns;
    if (inner_size != right_inner) {
        return LLM_INVALID_SHAPE;
    }
    cpu_matmul_ex_job job = {
        .left = left,
        .right = right,
        .output = output,
        .left_columns = left_columns,
        .right_columns = right_columns,
        .inner_size = inner_size,
        .output_columns = output_columns,
        .transpose_left = transpose_left,
        .transpose_right = transpose_right,
    };
    return llm_cpu_parallel_for(cpu->executor, output_rows, LLM_CPU_MATMUL_ROWS_PER_TASK,
                                matmul_ex_range, &job);
}
