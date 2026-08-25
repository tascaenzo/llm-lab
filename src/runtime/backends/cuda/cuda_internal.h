#ifndef LLM_LAB_CUDA_INTERNAL_H
#define LLM_LAB_CUDA_INTERNAL_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "backend_internal.h"

#ifdef __cplusplus

#include <cublas_v2.h>
#include <cuda_runtime.h>

/** Sticky device flags observed after a dispatch. Index order is part of the layout. */
#define LLM_CUDA_FLAG_NON_FINITE 0
#define LLM_CUDA_FLAG_INVALID_INDEX 1
#define LLM_CUDA_FLAG_COUNT 2

/** Threads per block for one-dimensional elementwise dispatches. */
#define LLM_CUDA_ELEMENTWISE_BLOCK 256U
/** Threads per block for row-oriented dispatches that reduce across a row. */
#define LLM_CUDA_ROW_BLOCK 256U
/** Threads per block for the attention kernels, which reduce over the head dimension. */
#define LLM_CUDA_ATTENTION_BLOCK 128U

/**
 * One device allocation. Callers only ever see the address of this wrapper, never
 * the device pointer: the runtime facade passes storage addresses around as plain
 * float pointers, and a device address would be dereferenced by host code.
 */
typedef struct llm_cuda_buffer {
    void *pointer;
    size_t byte_count;
    size_t capacity;
    struct llm_cuda_buffer *next;
} llm_cuda_buffer;

typedef enum llm_cuda_math_mode {
    LLM_CUDA_MATH_F32 = 0,
    LLM_CUDA_MATH_TF32,
    /** FP32 storage and output with BF16-rounded GEMM multiplicands. */
    LLM_CUDA_MATH_BF16_COMPUTE
} llm_cuda_math_mode;

typedef enum llm_cuda_numerics_mode {
    /** Scan every floating-point result. Used by the contract tests and debugging. */
    LLM_CUDA_NUMERICS_STRICT = 0,
    /** Scan loss and validate updated parameters inside AdamW, relying on propagation elsewhere. */
    LLM_CUDA_NUMERICS_STEP
} llm_cuda_numerics_mode;

typedef struct llm_cuda_context {
    int device_index;
    cudaStream_t stream;
    cublasHandle_t blas;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    llm_cuda_buffer *buffers;
    llm_cuda_buffer *cached_buffers;
    pthread_mutex_t buffer_mutex;
    char *device_name;
    /** Device-resident sticky flags, mirrored into pinned host memory when read. */
    int *device_flags;
    int *host_flags;
    llm_cuda_backend_metrics metrics;
    int batch_active;
    int events_enabled;
    int timing_active;
    llm_cuda_math_mode math_mode;
    llm_cuda_numerics_mode numerics_mode;
} llm_cuda_context;

/** Returns the device pointer behind an opaque storage handle. */
void *llm_cuda_device_pointer(void *memory);
const void *llm_cuda_device_const_pointer(const void *memory);
int llm_cuda_context_contains_buffer(llm_cuda_context *context, const void *memory);

/**
 * Closes one dispatch. Outside a batch this synchronizes the stream, folds the GPU
 * time into the metrics and converts any sticky device flag into a status. Inside a
 * batch it returns immediately: the flags are consumed by llm_cuda_flush instead.
 */
llm_status llm_cuda_finish(llm_cuda_context *context);

/** Waits for every queued command and consumes the sticky flags. */
llm_status llm_cuda_flush(llm_cuda_context *context);

/** Starts an event interval when CUDA timing is available and no interval is open. */
void llm_cuda_start_timing(llm_cuda_context *context);

/** Queues a scan that raises the non-finite flag when any value is not finite. */
llm_status llm_cuda_check_finite(llm_cuda_context *context, const void *memory, size_t value_count);

/** Queues a scan that raises the invalid-index flag for indices outside [0, bound). */
llm_status llm_cuda_check_indices(llm_cuda_context *context, const void *indices, size_t count,
                                  size_t bound);

llm_status llm_cuda_allocate(void *context, size_t byte_count, void **out_memory);
void llm_cuda_deallocate(void *context, void *memory);
llm_status llm_cuda_zero(void *context, void *memory, size_t byte_count);
llm_status llm_cuda_copy(void *context, const void *source, void *destination, size_t byte_count);

llm_status llm_cuda_fill_f32(void *context, float *values, size_t value_count, float value);
llm_status llm_cuda_add_f32(void *context, const float *left, const float *right, float *output,
                            size_t value_count);
llm_status llm_cuda_multiply_f32(void *context, const float *left, const float *right,
                                 float *output, size_t value_count);
llm_status llm_cuda_scale_f32(void *context, const float *input, float scale, float *output,
                              size_t value_count);
llm_status llm_cuda_reduce_sum_last_f32(void *context, const float *input, float *output,
                                        size_t outer_count, size_t reduction_size);
llm_status llm_cuda_reduce_max_last_f32(void *context, const float *input, float *output,
                                        size_t outer_count, size_t reduction_size);
llm_status llm_cuda_reduce_mean_square_last_f32(void *context, const float *input, float *output,
                                                size_t outer_count, size_t reduction_size);
llm_status llm_cuda_accumulate_sum_squares_f32(void *context, const float *input,
                                               float *accumulator, size_t value_count);
llm_status llm_cuda_matmul_f32(void *context, const float *left, const float *right, float *output,
                               size_t rows, size_t inner_size, size_t columns);
llm_status llm_cuda_matmul_ex_f32(void *context, const float *left, const float *right,
                                  float *output, size_t left_rows, size_t left_columns,
                                  size_t right_rows, size_t right_columns, int transpose_left,
                                  int transpose_right);
llm_status llm_cuda_gather_rows_f32(void *context, const float *table, size_t row_count,
                                    size_t row_width, const uint32_t *indices, size_t index_count,
                                    float *output);
llm_status llm_cuda_scatter_add_rows_f32(void *context, const float *source,
                                         const uint32_t *indices, size_t index_count,
                                         size_t row_width, size_t row_count, float *table);
llm_status llm_cuda_accumulate_f32(void *context, const float *source, float *destination,
                                   size_t value_count);
llm_status llm_cuda_silu_f32(void *context, const float *input, float *output, size_t value_count);
llm_status llm_cuda_silu_backward_f32(void *context, const float *input,
                                      const float *output_gradient, float *input_gradient,
                                      size_t value_count);
llm_status llm_cuda_rms_norm_f32(void *context, const float *input, const float *weight,
                                 float epsilon, float *output, size_t outer_count,
                                 size_t row_width);
llm_status llm_cuda_rms_norm_backward_f32(void *context, const float *input, const float *weight,
                                          const float *output_gradient, float epsilon,
                                          float *input_gradient, float *weight_gradient,
                                          size_t outer_count, size_t row_width);
llm_status llm_cuda_rope_f32(void *context, const float *input, const float *cos_table,
                             const float *sin_table, size_t batch_count, size_t sequence_length,
                             size_t head_count, size_t head_dimension, float *output);
llm_status llm_cuda_rope_backward_f32(void *context, const float *output_gradient,
                                      const float *cos_table, const float *sin_table,
                                      size_t batch_count, size_t sequence_length, size_t head_count,
                                      size_t head_dimension, float *input_gradient);
llm_status llm_cuda_attention_forward_f32(void *context, const float *query, const float *key,
                                          const float *value, float scale, size_t batch_count,
                                          size_t sequence_length, size_t query_head_count,
                                          size_t key_value_head_count, size_t head_dimension,
                                          float *output);
llm_status llm_cuda_attention_backward_f32(void *context, const float *query, const float *key,
                                           const float *value, const float *output_gradient,
                                           float scale, size_t batch_count, size_t sequence_length,
                                           size_t query_head_count, size_t key_value_head_count,
                                           size_t head_dimension, float *query_gradient,
                                           float *key_gradient, float *value_gradient);
llm_status llm_cuda_softmax_last_f32(void *context, const float *input, float *output,
                                     size_t outer_count, size_t row_width);
llm_status llm_cuda_cross_entropy_forward_f32(void *context, const float *logits,
                                              const uint32_t *targets, size_t row_count,
                                              size_t vocabulary_size, float *loss);
llm_status llm_cuda_cross_entropy_backward_f32(void *context, const float *logits,
                                               const uint32_t *targets, size_t row_count,
                                               size_t vocabulary_size, float *gradient);
llm_status llm_cuda_adamw_update_f32(void *context, float *parameter, float *gradient,
                                     float *first_moment, float *second_moment, size_t value_count,
                                     float learning_rate, float beta1, float beta2, float epsilon,
                                     float weight_decay, float gradient_scale,
                                     unsigned long long step, int zero_gradient);

/* Kernel launchers. Each one only enqueues work on the context stream. */
void llm_cuda_launch_fill(cudaStream_t stream, float *output, size_t count, float value);
void llm_cuda_launch_add(cudaStream_t stream, const float *left, const float *right, float *output,
                         size_t count);
void llm_cuda_launch_multiply(cudaStream_t stream, const float *left, const float *right,
                              float *output, size_t count);
void llm_cuda_launch_scale(cudaStream_t stream, const float *input, float scale, float *output,
                           size_t count);
void llm_cuda_launch_accumulate(cudaStream_t stream, const float *source, float *destination,
                                size_t count);
void llm_cuda_launch_silu(cudaStream_t stream, const float *input, float *output, size_t count);
void llm_cuda_launch_silu_backward(cudaStream_t stream, const float *input,
                                   const float *output_gradient, float *input_gradient,
                                   size_t count);
void llm_cuda_launch_reduce_sum_last(cudaStream_t stream, const float *input, float *output,
                                     size_t outer_count, size_t reduction_size);
void llm_cuda_launch_reduce_max_last(cudaStream_t stream, const float *input, float *output,
                                     size_t outer_count, size_t reduction_size);
void llm_cuda_launch_reduce_mean_square_last(cudaStream_t stream, const float *input, float *output,
                                             size_t outer_count, size_t reduction_size);
void llm_cuda_launch_accumulate_sum_squares(cudaStream_t stream, const float *input,
                                            float *accumulator, size_t value_count);
void llm_cuda_launch_softmax_last(cudaStream_t stream, const float *input, float *output,
                                  size_t outer_count, size_t row_width);
void llm_cuda_launch_gather_rows(cudaStream_t stream, const float *table, const uint32_t *indices,
                                 float *output, size_t row_count, size_t row_width,
                                 size_t index_count);
void llm_cuda_launch_scatter_add_rows(cudaStream_t stream, const float *source,
                                      const uint32_t *indices, float *table, size_t row_count,
                                      size_t row_width, size_t index_count);
void llm_cuda_launch_rms_norm(cudaStream_t stream, const float *input, const float *weight,
                              float epsilon, float *output, size_t outer_count, size_t row_width);
void llm_cuda_launch_rms_norm_backward(cudaStream_t stream, const float *input, const float *weight,
                                       const float *output_gradient, float epsilon,
                                       float *input_gradient, float *weight_gradient,
                                       size_t outer_count, size_t row_width);
void llm_cuda_launch_rope(cudaStream_t stream, const float *input, const float *cos_table,
                          const float *sin_table, float *output, size_t pair_count,
                          size_t sequence_length, size_t pairs_per_head, size_t head_count);
void llm_cuda_launch_rope_backward(cudaStream_t stream, const float *output_gradient,
                                   const float *cos_table, const float *sin_table,
                                   float *input_gradient, size_t pair_count, size_t sequence_length,
                                   size_t pairs_per_head, size_t head_count);
void llm_cuda_launch_attention_forward(cudaStream_t stream, const float *query, const float *key,
                                       const float *value, float *output, float scale,
                                       size_t query_rows, size_t sequence_length,
                                       size_t query_head_count, size_t key_value_head_count,
                                       size_t head_dimension);
void llm_cuda_launch_attention_backward(cudaStream_t stream, const float *query, const float *key,
                                        const float *value, const float *output_gradient,
                                        float *query_gradient, float *key_gradient,
                                        float *value_gradient, float scale, size_t query_rows,
                                        size_t sequence_length, size_t query_head_count,
                                        size_t key_value_head_count, size_t head_dimension);
void llm_cuda_launch_cross_entropy_forward(cudaStream_t stream, const float *logits,
                                           const uint32_t *targets, float *loss, size_t row_count,
                                           size_t vocabulary_size);
void llm_cuda_launch_cross_entropy_backward(cudaStream_t stream, const float *logits,
                                            const uint32_t *targets, float *gradient,
                                            size_t row_count, size_t vocabulary_size);
void llm_cuda_launch_adamw(cudaStream_t stream, float *parameter, float *gradient,
                           float *first_moment, float *second_moment, size_t count,
                           float learning_rate, float beta1, float beta2, float epsilon,
                           float weight_decay, float gradient_scale, float inverse_first_bias,
                           float inverse_second_bias, int zero_gradient, int *flags);
void llm_cuda_launch_check_finite(cudaStream_t stream, const float *values, size_t count,
                                  int *flags);
void llm_cuda_launch_check_indices(cudaStream_t stream, const uint32_t *indices, size_t count,
                                   uint32_t bound, int *flags);

#endif /* __cplusplus */

#endif
