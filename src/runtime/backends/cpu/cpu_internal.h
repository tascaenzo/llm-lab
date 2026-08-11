#ifndef LLM_LAB_CPU_INTERNAL_H
#define LLM_LAB_CPU_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "cpu_executor.h"
#include "runtime_internal.h"

#define LLM_CPU_MAX_THREADS 256U

typedef struct llm_cpu_context {
    llm_cpu_executor *executor;
    size_t thread_count;
    int deterministic;
} llm_cpu_context;

size_t llm_cpu_detect_thread_count(void);

llm_status llm_cpu_execute_zero(void *context, void *memory, size_t byte_count);
llm_status llm_cpu_execute_copy(void *context, const void *source, void *destination,
                                size_t byte_count);
llm_status llm_cpu_execute_cast(void *context, const void *input, llm_dtype input_dtype,
                                void *output, llm_dtype output_dtype, size_t value_count);
llm_status llm_cpu_execute_fill_f32(void *context, float *values, size_t value_count, float value);
llm_status llm_cpu_execute_add_f32(void *context, const float *left, const float *right,
                                   float *output, size_t value_count);
llm_status llm_cpu_execute_multiply_f32(void *context, const float *left, const float *right,
                                        float *output, size_t value_count);
llm_status llm_cpu_execute_scale_f32(void *context, const float *input, float scale, float *output,
                                     size_t value_count);
llm_status llm_cpu_execute_reduce_sum_last_f32(void *context, const float *input, float *output,
                                               size_t outer_count, size_t reduction_size);
llm_status llm_cpu_execute_reduce_max_last_f32(void *context, const float *input, float *output,
                                               size_t outer_count, size_t reduction_size);
llm_status llm_cpu_execute_reduce_mean_square_last_f32(void *context, const float *input,
                                                       float *output, size_t outer_count,
                                                       size_t reduction_size);
llm_status llm_cpu_execute_matmul_f32(void *context, const float *left, const float *right,
                                      float *output, size_t rows, size_t inner_size,
                                      size_t columns);
llm_status llm_cpu_execute_matmul_mixed_f32(void *context, const void *left, const void *right,
                                            llm_dtype input_dtype, float *output, size_t rows,
                                            size_t inner_size, size_t columns);
llm_status llm_cpu_execute_gather_rows_f32(void *context, const float *table, size_t row_count,
                                           size_t row_width, const uint32_t *indices,
                                           size_t index_count, float *output);
llm_status llm_cpu_execute_scatter_add_rows_f32(void *context, const float *source,
                                                const uint32_t *indices, size_t index_count,
                                                size_t row_width, size_t row_count, float *table);
llm_status llm_cpu_execute_softmax_last_f32(void *context, const float *input, float *output,
                                            size_t outer_count, size_t row_width);
llm_status llm_cpu_execute_cross_entropy_forward_f32(void *context, const float *logits,
                                                     const uint32_t *targets, size_t row_count,
                                                     size_t vocabulary_size, float *loss);
llm_status llm_cpu_execute_cross_entropy_backward_f32(void *context, const float *logits,
                                                      const uint32_t *targets, size_t row_count,
                                                      size_t vocabulary_size, float *gradient);

#endif
