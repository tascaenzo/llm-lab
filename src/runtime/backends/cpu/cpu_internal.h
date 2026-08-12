#ifndef LLM_LAB_CPU_INTERNAL_H
#define LLM_LAB_CPU_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "backend_internal.h"
#include "cpu_executor.h"

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
llm_status llm_cpu_execute_matmul_ex_f32(void *context, const float *left, const float *right,
                                         float *output, size_t left_rows, size_t left_columns,
                                         size_t right_rows, size_t right_columns,
                                         int transpose_left, int transpose_right);
llm_status llm_cpu_execute_matmul_mixed_f32(void *context, const void *left, const void *right,
                                            llm_dtype input_dtype, float *output, size_t rows,
                                            size_t inner_size, size_t columns);
llm_status llm_cpu_execute_gather_rows_f32(void *context, const float *table, size_t row_count,
                                           size_t row_width, const uint32_t *indices,
                                           size_t index_count, float *output);
llm_status llm_cpu_execute_scatter_add_rows_f32(void *context, const float *source,
                                                const uint32_t *indices, size_t index_count,
                                                size_t row_width, size_t row_count, float *table);
llm_status llm_cpu_execute_accumulate_f32(void *context, const float *source, float *destination,
                                          size_t value_count);
llm_status llm_cpu_execute_silu_f32(void *context, const float *input, float *output,
                                    size_t value_count);
llm_status llm_cpu_execute_silu_backward_f32(void *context, const float *input,
                                             const float *output_gradient, float *input_gradient,
                                             size_t value_count);
llm_status llm_cpu_execute_rms_norm_f32(void *context, const float *input, const float *weight,
                                        float epsilon, float *output, size_t outer_count,
                                        size_t row_width);
llm_status llm_cpu_execute_rms_norm_backward_f32(void *context, const float *input,
                                                 const float *weight, const float *output_gradient,
                                                 float epsilon, float *input_gradient,
                                                 float *weight_gradient, size_t outer_count,
                                                 size_t row_width);
llm_status llm_cpu_execute_rope_f32(void *context, const float *input, const float *cos_table,
                                    const float *sin_table, size_t batch_count,
                                    size_t sequence_length, size_t head_count,
                                    size_t head_dimension, size_t table_position_count,
                                    size_t position_offset, float *output);
llm_status llm_cpu_execute_rope_backward_f32(void *context, const float *output_gradient,
                                             const float *cos_table, const float *sin_table,
                                             size_t batch_count, size_t sequence_length,
                                             size_t head_count, size_t head_dimension,
                                             size_t table_position_count, size_t position_offset,
                                             float *input_gradient);
llm_status llm_cpu_execute_attention_forward_f32(
    void *context, const float *query, const float *key, const float *value, float scale,
    size_t query_position_offset, size_t batch_count, size_t query_length, size_t key_length,
    size_t query_head_count, size_t key_value_head_count, size_t head_dimension, float *output);
llm_status llm_cpu_execute_attention_backward_f32(
    void *context, const float *query, const float *key, const float *value,
    const float *output_gradient, float scale, size_t query_position_offset, size_t batch_count,
    size_t query_length, size_t key_length, size_t query_head_count, size_t key_value_head_count,
    size_t head_dimension, float *query_gradient, float *key_gradient, float *value_gradient);
llm_status llm_cpu_execute_softmax_last_f32(void *context, const float *input, float *output,
                                            size_t outer_count, size_t row_width);
llm_status llm_cpu_execute_cross_entropy_forward_f32(void *context, const float *logits,
                                                     const uint32_t *targets, size_t row_count,
                                                     size_t vocabulary_size, float *loss);
llm_status llm_cpu_execute_cross_entropy_backward_f32(void *context, const float *logits,
                                                      const uint32_t *targets, size_t row_count,
                                                      size_t vocabulary_size, float *gradient);
llm_status llm_cpu_execute_adamw_update_f32(void *context, float *parameter, const float *gradient,
                                            float *first_moment, float *second_moment,
                                            size_t value_count, float learning_rate, float beta1,
                                            float beta2, float epsilon, float weight_decay,
                                            float gradient_scale, unsigned long long step);

#endif
