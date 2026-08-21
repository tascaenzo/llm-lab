#ifndef LLM_LAB_BACKEND_INTERNAL_H
#define LLM_LAB_BACKEND_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "runtime/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Hardware contract used by the validated runtime facade. */
typedef struct llm_backend_ops {
    void (*destroy)(void *context);
    int (*supports_dtype)(const void *context, llm_dtype dtype);
    llm_status (*allocate)(void *context, size_t byte_count, void **out_memory);
    void (*deallocate)(void *context, void *memory);
    llm_status (*zero)(void *context, void *memory, size_t byte_count);
    llm_status (*copy)(void *context, const void *source, void *destination, size_t byte_count);
    llm_status (*fill_f32)(void *context, float *values, size_t value_count, float value);
    llm_status (*add_f32)(void *context, const float *left, const float *right, float *output,
                          size_t value_count);
    llm_status (*multiply_f32)(void *context, const float *left, const float *right, float *output,
                               size_t value_count);
    llm_status (*scale_f32)(void *context, const float *input, float scale, float *output,
                            size_t value_count);
    llm_status (*reduce_sum_last_f32)(void *context, const float *input, float *output,
                                      size_t outer_count, size_t reduction_size);
    llm_status (*reduce_max_last_f32)(void *context, const float *input, float *output,
                                      size_t outer_count, size_t reduction_size);
    llm_status (*reduce_mean_square_last_f32)(void *context, const float *input, float *output,
                                              size_t outer_count, size_t reduction_size);
    llm_status (*accumulate_sum_squares_f32)(void *context, const float *input, float *accumulator,
                                             size_t value_count);
    llm_status (*matmul_f32)(void *context, const float *left, const float *right, float *output,
                             size_t rows, size_t inner_size, size_t columns);
    llm_status (*matmul_ex_f32)(void *context, const float *left, const float *right, float *output,
                                size_t left_rows, size_t left_columns, size_t right_rows,
                                size_t right_columns, int transpose_left, int transpose_right);
    llm_status (*gather_rows_f32)(void *context, const float *table, size_t row_count,
                                  size_t row_width, const uint32_t *indices, size_t index_count,
                                  float *output);
    llm_status (*scatter_add_rows_f32)(void *context, const float *source, const uint32_t *indices,
                                       size_t index_count, size_t row_width, size_t row_count,
                                       float *table);
    llm_status (*accumulate_f32)(void *context, const float *source, float *destination,
                                 size_t value_count);
    llm_status (*silu_f32)(void *context, const float *input, float *output, size_t value_count);
    llm_status (*silu_backward_f32)(void *context, const float *input, const float *output_gradient,
                                    float *input_gradient, size_t value_count);
    llm_status (*rms_norm_f32)(void *context, const float *input, const float *weight,
                               float epsilon, float *output, size_t outer_count, size_t row_width);
    llm_status (*rms_norm_backward_f32)(void *context, const float *input, const float *weight,
                                        const float *output_gradient, float epsilon,
                                        float *input_gradient, float *weight_gradient,
                                        size_t outer_count, size_t row_width);
    llm_status (*rope_f32)(void *context, const float *input, const float *cos_table,
                           const float *sin_table, size_t batch_count, size_t sequence_length,
                           size_t head_count, size_t head_dimension, float *output);
    llm_status (*rope_backward_f32)(void *context, const float *output_gradient,
                                    const float *cos_table, const float *sin_table,
                                    size_t batch_count, size_t sequence_length, size_t head_count,
                                    size_t head_dimension, float *input_gradient);
    llm_status (*attention_forward_f32)(void *context, const float *query, const float *key,
                                        const float *value, float scale, size_t batch_count,
                                        size_t sequence_length, size_t query_head_count,
                                        size_t key_value_head_count, size_t head_dimension,
                                        float *output);
    llm_status (*attention_backward_f32)(void *context, const float *query, const float *key,
                                         const float *value, const float *output_gradient,
                                         float scale, size_t batch_count, size_t sequence_length,
                                         size_t query_head_count, size_t key_value_head_count,
                                         size_t head_dimension, float *query_gradient,
                                         float *key_gradient, float *value_gradient);
    llm_status (*softmax_last_f32)(void *context, const float *input, float *output,
                                   size_t outer_count, size_t row_width);
    llm_status (*cross_entropy_forward_f32)(void *context, const float *logits,
                                            const uint32_t *targets, size_t row_count,
                                            size_t vocabulary_size, float *loss);
    llm_status (*cross_entropy_backward_f32)(void *context, const float *logits,
                                             const uint32_t *targets, size_t row_count,
                                             size_t vocabulary_size, float *gradient);
    llm_status (*adamw_update_f32)(void *context, float *parameter, float *gradient,
                                   float *first_moment, float *second_moment, size_t value_count,
                                   float learning_rate, float beta1, float beta2, float epsilon,
                                   float weight_decay, float gradient_scale,
                                   unsigned long long step, int zero_gradient);
    llm_status (*synchronize)(void *context);
} llm_backend_ops;

struct llm_backend {
    llm_device_type device;
    const llm_backend_ops *ops;
    void *context;
};

size_t llm_dtype_size(llm_dtype dtype);
int llm_backend_supports_dtype(const llm_backend *backend, llm_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif
