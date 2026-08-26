#ifndef LLM_LAB_METAL_INTERNAL_H
#define LLM_LAB_METAL_INTERNAL_H

#include <pthread.h>
#include <stdint.h>

#include "backend_internal.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef enum llm_metal_pipeline {
    LLM_METAL_PIPELINE_FILL = 0,
    LLM_METAL_PIPELINE_ADD,
    LLM_METAL_PIPELINE_MULTIPLY,
    LLM_METAL_PIPELINE_SCALE,
    LLM_METAL_PIPELINE_REDUCE_SUM,
    LLM_METAL_PIPELINE_REDUCE_MAX,
    LLM_METAL_PIPELINE_REDUCE_MEAN_SQUARE,
    LLM_METAL_PIPELINE_ACCUMULATE_SUM_SQUARES,
    LLM_METAL_PIPELINE_MATMUL,
    LLM_METAL_PIPELINE_MATMUL_LARGE,
    LLM_METAL_PIPELINE_MATMUL_SIMDGROUP,
    LLM_METAL_PIPELINE_ACCUMULATE,
    LLM_METAL_PIPELINE_GATHER,
    LLM_METAL_PIPELINE_SCATTER_ADD,
    LLM_METAL_PIPELINE_SOFTMAX,
    LLM_METAL_PIPELINE_CROSS_ENTROPY_FORWARD,
    LLM_METAL_PIPELINE_CROSS_ENTROPY_BACKWARD,
    LLM_METAL_PIPELINE_SILU,
    LLM_METAL_PIPELINE_SILU_BACKWARD,
    LLM_METAL_PIPELINE_RMS_NORM,
    LLM_METAL_PIPELINE_RMS_NORM_BACKWARD,
    LLM_METAL_PIPELINE_ROPE,
    LLM_METAL_PIPELINE_ROPE_BACKWARD,
    LLM_METAL_PIPELINE_ATTENTION_FORWARD,
    LLM_METAL_PIPELINE_ATTENTION_BACKWARD,
    LLM_METAL_PIPELINE_ADAMW,
    LLM_METAL_PIPELINE_COUNT
} llm_metal_pipeline;

typedef struct llm_metal_buffer {
    id<MTLBuffer> handle;
    size_t byte_count;
    size_t capacity;
    struct llm_metal_buffer *next;
} llm_metal_buffer;

typedef struct llm_metal_matmul_tuning {
    uint32_t rows;
    uint32_t inner_size;
    uint32_t columns;
    llm_dtype dtype;
    llm_metal_pipeline pipeline;
    struct llm_metal_matmul_tuning *next;
} llm_metal_matmul_tuning;

/** Cached MPS GEMM object. Matrices are deliberately not cached: they wrap caller buffers. */
typedef struct llm_metal_mps_gemm {
    uint32_t left_rows;
    uint32_t left_columns;
    uint32_t right_rows;
    uint32_t right_columns;
    uint8_t transpose_left;
    uint8_t transpose_right;
    MPSMatrixMultiplication *kernel;
    struct llm_metal_mps_gemm *next;
} llm_metal_mps_gemm;

typedef struct llm_metal_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipelines[LLM_METAL_PIPELINE_COUNT];
    llm_metal_buffer *buffers;
    llm_metal_buffer *cached_buffers;
    llm_metal_matmul_tuning *matmul_tunings;
    size_t matmul_tuning_count;
    llm_metal_mps_gemm *mps_gemms;
    size_t mps_gemm_count;
    pthread_mutex_t buffer_mutex;
    char *device_name;
    id<MTLCommandBuffer> batch_command_buffer;
    id<MTLComputeCommandEncoder> batch_compute_encoder;
    llm_metal_backend_metrics metrics;
    int batch_active;
} llm_metal_context;

llm_metal_buffer *llm_metal_buffer_from_memory(void *memory);
const llm_metal_buffer *llm_metal_buffer_from_const_memory(const void *memory);
int llm_metal_context_contains_buffer(llm_metal_context *context, const void *memory);
id<MTLCommandBuffer> llm_metal_acquire_command_buffer(llm_metal_context *context);
id<MTLComputeCommandEncoder> llm_metal_acquire_compute_encoder(llm_metal_context *context,
                                                               id<MTLCommandBuffer> command_buffer);
void llm_metal_close_batch_compute_encoder(llm_metal_context *context);
llm_status llm_metal_submit(llm_metal_context *context, id<MTLCommandBuffer> command_buffer);
llm_status llm_metal_flush(llm_metal_context *context);

llm_status llm_metal_allocate(void *context, size_t byte_count, void **out_memory);
void llm_metal_deallocate(void *context, void *memory);
llm_status llm_metal_zero(void *context, void *memory, size_t byte_count);
llm_status llm_metal_copy(void *context, const void *source, void *destination, size_t byte_count);

llm_status llm_metal_fill_f32(void *context, float *values, size_t value_count, float value);
llm_status llm_metal_add_f32(void *context, const float *left, const float *right, float *output,
                             size_t value_count);
llm_status llm_metal_multiply_f32(void *context, const float *left, const float *right,
                                  float *output, size_t value_count);
llm_status llm_metal_scale_f32(void *context, const float *input, float scale, float *output,
                               size_t value_count);
llm_status llm_metal_reduce_sum_last_f32(void *context, const float *input, float *output,
                                         size_t outer_count, size_t reduction_size);
llm_status llm_metal_reduce_max_last_f32(void *context, const float *input, float *output,
                                         size_t outer_count, size_t reduction_size);
llm_status llm_metal_reduce_mean_square_last_f32(void *context, const float *input, float *output,
                                                 size_t outer_count, size_t reduction_size);
llm_status llm_metal_accumulate_sum_squares_f32(void *context, const float *input,
                                                float *accumulator, size_t value_count);
llm_status llm_metal_matmul_f32(void *context, const float *left, const float *right, float *output,
                                size_t rows, size_t inner_size, size_t columns);
llm_status llm_metal_matmul_ex_f32(void *context, const float *left, const float *right,
                                   float *output, size_t left_rows, size_t left_columns,
                                   size_t right_rows, size_t right_columns, int transpose_left,
                                   int transpose_right);
llm_status llm_metal_accumulate_f32(void *context, const float *source, float *destination,
                                    size_t value_count);
llm_status llm_metal_silu_f32(void *context, const float *input, float *output, size_t value_count);
llm_status llm_metal_silu_backward_f32(void *context, const float *input,
                                       const float *output_gradient, float *input_gradient,
                                       size_t value_count);
llm_status llm_metal_rms_norm_f32(void *context, const float *input, const float *weight,
                                  float epsilon, float *output, size_t outer_count,
                                  size_t row_width);
llm_status llm_metal_rms_norm_backward_f32(void *context, const float *input, const float *weight,
                                           const float *output_gradient, float epsilon,
                                           float *input_gradient, float *weight_gradient,
                                           size_t outer_count, size_t row_width);
llm_status llm_metal_rope_f32(void *context, const float *input, const float *cos_table,
                              const float *sin_table, size_t batch_count, size_t sequence_length,
                              size_t head_count, size_t head_dimension, float *output);
llm_status llm_metal_rope_backward_f32(void *context, const float *output_gradient,
                                       const float *cos_table, const float *sin_table,
                                       size_t batch_count, size_t sequence_length,
                                       size_t head_count, size_t head_dimension,
                                       float *input_gradient);
llm_status llm_metal_attention_forward_f32(void *context, const float *query, const float *key,
                                           const float *value, float scale, size_t batch_count,
                                           size_t sequence_length, size_t query_head_count,
                                           size_t key_value_head_count, size_t head_dimension,
                                           float *output);
llm_status llm_metal_attention_backward_f32(void *context, const float *query, const float *key,
                                            const float *value, const float *output_gradient,
                                            float scale, size_t batch_count, size_t sequence_length,
                                            size_t query_head_count, size_t key_value_head_count,
                                            size_t head_dimension, float *query_gradient,
                                            float *key_gradient, float *value_gradient);
llm_status llm_metal_gather_rows_f32(void *context, const float *table, size_t row_count,
                                     size_t row_width, const uint32_t *indices, size_t index_count,
                                     float *output);
llm_status llm_metal_scatter_add_rows_f32(void *context, const float *source,
                                          const uint32_t *indices, size_t index_count,
                                          size_t row_width, size_t row_count, float *table);
llm_status llm_metal_softmax_last_f32(void *context, const float *input, float *output,
                                      size_t outer_count, size_t row_width);
llm_status llm_metal_cross_entropy_forward_f32(void *context, const float *logits,
                                               const uint32_t *targets,
                                               const uint32_t *loss_mask, size_t row_count,
                                               size_t vocabulary_size,
                                               size_t normalization_row_count, float *loss);
llm_status llm_metal_cross_entropy_backward_f32(void *context, const float *logits,
                                                const uint32_t *targets,
                                                const uint32_t *loss_mask, size_t row_count,
                                                size_t vocabulary_size,
                                                size_t normalization_row_count, float *gradient);
llm_status llm_metal_adamw_update_f32(void *context, float *parameter, float *gradient,
                                      float *first_moment, float *second_moment, size_t value_count,
                                      float learning_rate, float beta1, float beta2, float epsilon,
                                      float weight_decay, float gradient_scale,
                                      unsigned long long step, int zero_gradient);

#endif

#endif
