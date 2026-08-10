#ifndef LLM_LAB_CPU_KERNELS_H
#define LLM_LAB_CPU_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#include "runtime/runtime.h"

llm_status llm_cpu_add_f32(const float *left, const float *right, float *output,
                           size_t value_count);
llm_status llm_cpu_multiply_f32(const float *left, const float *right, float *output,
                                size_t value_count);
llm_status llm_cpu_scale_f32(const float *input, float scale, float *output, size_t value_count);

llm_status llm_cpu_reduce_sum_last_f32(const float *input, float *output, size_t outer_count,
                                       size_t reduction_size);
llm_status llm_cpu_reduce_max_last_f32(const float *input, float *output, size_t outer_count,
                                       size_t reduction_size);
llm_status llm_cpu_reduce_mean_square_last_f32(const float *input, float *output,
                                               size_t outer_count, size_t reduction_size);

llm_status llm_cpu_matmul_f32(const float *left, const float *right, float *output, size_t rows,
                              size_t inner_size, size_t columns);

llm_status llm_cpu_gather_rows_f32(const float *table, size_t row_count, size_t row_width,
                                   const uint32_t *indices, size_t index_count, float *output);
llm_status llm_cpu_scatter_add_rows_f32(const float *source, const uint32_t *indices,
                                        size_t index_count, size_t row_width, size_t row_count,
                                        float *table);

llm_status llm_cpu_softmax_last_f32(const float *input, float *output, size_t outer_count,
                                    size_t row_width);
llm_status llm_cpu_cross_entropy_forward_f32(const float *logits, const uint32_t *targets,
                                             size_t row_count, size_t vocabulary_size, float *loss);
llm_status llm_cpu_cross_entropy_backward_f32(const float *logits, const uint32_t *targets,
                                              size_t row_count, size_t vocabulary_size,
                                              size_t normalization_row_count, float *gradient);

#endif
