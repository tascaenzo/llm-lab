#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "metal_internal.h"

#define LLM_METAL_MAX_THREADS_1D 256U
#define LLM_METAL_MATMUL_TILE 16U
#define LLM_METAL_ELEMENTWISE_VECTOR_WIDTH 4U
#define LLM_METAL_SUM_SQUARES_VALUES_PER_GROUP 4096U

typedef struct metal_elementwise_parameters {
    uint32_t count;
    float scalar;
} metal_elementwise_parameters;

typedef struct metal_reduction_parameters {
    uint32_t outer_count;
    uint32_t reduction_size;
} metal_reduction_parameters;

typedef struct metal_matmul_parameters {
    uint32_t rows;
    uint32_t inner_size;
    uint32_t columns;
} metal_matmul_parameters;

typedef struct metal_gather_parameters {
    uint32_t row_count;
    uint32_t row_width;
    uint32_t index_count;
} metal_gather_parameters;

typedef struct metal_cross_entropy_parameters {
    uint32_t row_count;
    uint32_t vocabulary_size;
} metal_cross_entropy_parameters;

typedef struct metal_rms_norm_parameters {
    uint32_t outer_count;
    uint32_t row_width;
    float epsilon;
} metal_rms_norm_parameters;

typedef struct metal_rope_parameters {
    uint32_t sequence_length;
    uint32_t head_count;
    uint32_t pairs_per_head;
} metal_rope_parameters;

typedef struct metal_attention_parameters {
    uint32_t batch_count;
    uint32_t sequence_length;
    uint32_t query_head_count;
    uint32_t key_value_head_count;
    uint32_t head_dimension;
    float scale;
} metal_attention_parameters;

typedef struct metal_adamw_parameters {
    uint32_t count;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float gradient_scale;
    float inverse_first_bias;
    float inverse_second_bias;
    uint32_t zero_gradient;
} metal_adamw_parameters;

#define LLM_METAL_AUTOTUNE_TRIALS 3U
#define LLM_METAL_AUTOTUNE_CACHE_LIMIT 128U
#define LLM_METAL_MPS_GEMM_CACHE_LIMIT 128U

static int metal_size_to_u32(size_t value, uint32_t *out_value) {
    if (value == 0U || value > UINT32_MAX || out_value == NULL) {
        return 0;
    }
    *out_value = (uint32_t)value;
    return 1;
}

static id<MTLBuffer> metal_buffer_handle(const void *memory) {
    return llm_metal_buffer_from_const_memory(memory)->handle;
}

static void metal_end_compute(llm_metal_context *context, id<MTLComputeCommandEncoder> encoder) {
    if (context->batch_active == 0) {
        [encoder endEncoding];
    }
}

static llm_status metal_begin_compute(llm_metal_context *context, llm_metal_pipeline pipeline,
                                      id<MTLCommandBuffer> *out_command_buffer,
                                      id<MTLComputeCommandEncoder> *out_encoder) {
    if (context == NULL || pipeline >= LLM_METAL_PIPELINE_COUNT || out_command_buffer == NULL ||
        out_encoder == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    id<MTLCommandBuffer> command_buffer = llm_metal_acquire_command_buffer(context);
    id<MTLComputeCommandEncoder> encoder =
        llm_metal_acquire_compute_encoder(context, command_buffer);
    if (command_buffer == nil || encoder == nil) {
        return LLM_BACKEND_ERROR;
    }
    [encoder setComputePipelineState:context->pipelines[pipeline]];
    ++context->metrics.kernel_dispatches;
    *out_command_buffer = command_buffer;
    *out_encoder = encoder;
    return LLM_OK;
}

static llm_status metal_dispatch_1d(llm_metal_context *context, llm_metal_pipeline pipeline,
                                    id<MTLCommandBuffer> command_buffer,
                                    id<MTLComputeCommandEncoder> encoder, size_t count) {
    const NSUInteger maximum_threads = [context->pipelines[pipeline] maxTotalThreadsPerThreadgroup];
    const NSUInteger threads =
        maximum_threads < LLM_METAL_MAX_THREADS_1D ? maximum_threads : LLM_METAL_MAX_THREADS_1D;
    if (threads == 0U || count > NSUIntegerMax) {
        metal_end_compute(context, encoder);
        return LLM_OVERFLOW;
    }
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1U, 1U)
        threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    metal_end_compute(context, encoder);
    return llm_metal_submit(context, command_buffer);
}

static llm_status metal_dispatch_row_groups(llm_metal_context *context, llm_metal_pipeline pipeline,
                                            id<MTLCommandBuffer> command_buffer,
                                            id<MTLComputeCommandEncoder> encoder,
                                            size_t group_count, size_t reduction_size) {
    id<MTLComputePipelineState> state = context->pipelines[pipeline];
    const NSUInteger width = [state threadExecutionWidth];
    const NSUInteger maximum = [state maxTotalThreadsPerThreadgroup];
    const NSUInteger limit =
        maximum < LLM_METAL_MAX_THREADS_1D ? maximum : LLM_METAL_MAX_THREADS_1D;
    NSUInteger threads = width;
    while (threads < reduction_size && threads <= limit / 2U) {
        threads *= 2U;
    }
    if (threads == 0U || group_count > NSUIntegerMax) {
        metal_end_compute(context, encoder);
        return LLM_OVERFLOW;
    }
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)group_count, 1U, 1U)
            threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    metal_end_compute(context, encoder);
    return llm_metal_submit(context, command_buffer);
}

static llm_status metal_dispatch_attention_groups(llm_metal_context *context,
                                                  llm_metal_pipeline pipeline,
                                                  id<MTLCommandBuffer> command_buffer,
                                                  id<MTLComputeCommandEncoder> encoder,
                                                  size_t group_count, size_t head_dimension,
                                                  size_t scratch_float_count) {
    id<MTLComputePipelineState> state = context->pipelines[pipeline];
    const NSUInteger width = [state threadExecutionWidth];
    const NSUInteger maximum = [state maxTotalThreadsPerThreadgroup];
    const NSUInteger limit =
        maximum < LLM_METAL_MAX_THREADS_1D ? maximum : LLM_METAL_MAX_THREADS_1D;
    /*
     * The forward pass reduces once per key position and does little else, so
     * staying inside one SIMD group pays: each reduction becomes a single
     * hardware instruction with no threadgroup barrier. The backward pass ends
     * with a loop parallel over the head dimension, which wants the wider
     * threadgroup instead.
     */
    NSUInteger threads = width;
    if (pipeline != LLM_METAL_PIPELINE_ATTENTION_FORWARD) {
        while (threads < head_dimension && threads <= limit / 2U) {
            threads *= 2U;
        }
    }
    if (threads == 0U || group_count > NSUIntegerMax ||
        scratch_float_count > SIZE_MAX / sizeof(float) ||
        scratch_float_count * sizeof(float) > NSUIntegerMax) {
        metal_end_compute(context, encoder);
        return LLM_OVERFLOW;
    }
    [encoder setThreadgroupMemoryLength:(NSUInteger)(scratch_float_count * sizeof(float))
                                atIndex:0U];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)group_count, 1U, 1U)
            threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    metal_end_compute(context, encoder);
    return llm_metal_submit(context, command_buffer);
}

static size_t metal_elementwise_thread_count(size_t value_count) {
    return value_count / LLM_METAL_ELEMENTWISE_VECTOR_WIDTH +
           (value_count % LLM_METAL_ELEMENTWISE_VECTOR_WIDTH == 0U ? 0U : 1U);
}

static int metal_buffer_values_are_finite(const void *memory, size_t value_count) {
    const float *values = [metal_buffer_handle(memory) contents];
    for (size_t index = 0U; index < value_count; ++index) {
        if (isfinite(values[index]) == 0) {
            return 0;
        }
    }
    return 1;
}

llm_status llm_metal_fill_f32(void *opaque_context, float *values, size_t value_count,
                              float value) {
    if (opaque_context == NULL || values == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    parameters.scalar = value;
    llm_metal_context *context = opaque_context;

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_FILL, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(values) offset:0U atIndex:0U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_FILL, command_buffer, encoder,
                                 metal_elementwise_thread_count(value_count));
    }
}

static llm_status metal_binary_elementwise(void *opaque_context, llm_metal_pipeline pipeline,
                                           const float *left, const float *right, float *output,
                                           size_t value_count) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status = metal_begin_compute(context, pipeline, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(left) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(right) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:2U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
        return metal_dispatch_1d(context, pipeline, command_buffer, encoder,
                                 metal_elementwise_thread_count(value_count));
    }
}

llm_status llm_metal_add_f32(void *context, const float *left, const float *right, float *output,
                             size_t value_count) {
    return metal_binary_elementwise(context, LLM_METAL_PIPELINE_ADD, left, right, output,
                                    value_count);
}

llm_status llm_metal_multiply_f32(void *context, const float *left, const float *right,
                                  float *output, size_t value_count) {
    return metal_binary_elementwise(context, LLM_METAL_PIPELINE_MULTIPLY, left, right, output,
                                    value_count);
}

llm_status llm_metal_scale_f32(void *opaque_context, const float *input, float scale, float *output,
                               size_t value_count) {
    if (opaque_context == NULL || input == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    parameters.scalar = scale;
    llm_metal_context *context = opaque_context;

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_SCALE, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:1U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_SCALE, command_buffer, encoder,
                                 metal_elementwise_thread_count(value_count));
    }
}

llm_status llm_metal_accumulate_f32(void *opaque_context, const float *source, float *destination,
                                    size_t value_count) {
    if (opaque_context == NULL || source == NULL || destination == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_ACCUMULATE, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(source) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(destination) offset:0U atIndex:1U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_ACCUMULATE, command_buffer, encoder,
                                 metal_elementwise_thread_count(value_count));
    }
}

llm_status llm_metal_silu_f32(void *opaque_context, const float *input, float *output,
                              size_t value_count) {
    if (opaque_context == NULL || input == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_SILU, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:1U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_SILU, command_buffer, encoder,
                                 metal_elementwise_thread_count(value_count));
    }
}

llm_status llm_metal_silu_backward_f32(void *opaque_context, const float *input,
                                       const float *output_gradient, float *input_gradient,
                                       size_t value_count) {
    if (opaque_context == NULL || input == NULL || output_gradient == NULL ||
        input_gradient == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status = metal_begin_compute(context, LLM_METAL_PIPELINE_SILU_BACKWARD,
                                                &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(output_gradient) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(input_gradient) offset:0U atIndex:2U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_SILU_BACKWARD, command_buffer, encoder,
                                 metal_elementwise_thread_count(value_count));
    }
}

static llm_status metal_rms_norm_parameters_create(size_t outer_count, size_t row_width,
                                                   float epsilon,
                                                   metal_rms_norm_parameters *out_parameters) {
    if (out_parameters == NULL || !isfinite(epsilon) || epsilon <= 0.0F ||
        metal_size_to_u32(outer_count, &out_parameters->outer_count) == 0 ||
        metal_size_to_u32(row_width, &out_parameters->row_width) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    out_parameters->epsilon = epsilon;
    return LLM_OK;
}

llm_status llm_metal_rms_norm_f32(void *opaque_context, const float *input, const float *weight,
                                  float epsilon, float *output, size_t outer_count,
                                  size_t row_width) {
    if (opaque_context == NULL || input == NULL || weight == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_rms_norm_parameters parameters = {0};
    llm_status status =
        metal_rms_norm_parameters_create(outer_count, row_width, epsilon, &parameters);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_RMS_NORM, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(weight) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:2U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
        status = metal_dispatch_row_groups(context, LLM_METAL_PIPELINE_RMS_NORM, command_buffer,
                                           encoder, outer_count, row_width);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(output, outer_count * row_width) != 0
                   ? LLM_OK
                   : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_rms_norm_backward_f32(void *opaque_context, const float *input,
                                           const float *weight, const float *output_gradient,
                                           float epsilon, float *input_gradient,
                                           float *weight_gradient, size_t outer_count,
                                           size_t row_width) {
    if (opaque_context == NULL || input == NULL || weight == NULL || output_gradient == NULL ||
        input_gradient == NULL || weight_gradient == NULL || outer_count > SIZE_MAX / row_width) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_rms_norm_parameters parameters = {0};
    llm_status status =
        metal_rms_norm_parameters_create(outer_count, row_width, epsilon, &parameters);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    status = llm_metal_zero(context, weight_gradient, row_width * sizeof(float));
    if (status != LLM_OK) {
        return status;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status = metal_begin_compute(context, LLM_METAL_PIPELINE_RMS_NORM_BACKWARD, &command_buffer,
                                     &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(weight) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(output_gradient) offset:0U atIndex:2U];
        [encoder setBuffer:metal_buffer_handle(input_gradient) offset:0U atIndex:3U];
        [encoder setBuffer:metal_buffer_handle(weight_gradient) offset:0U atIndex:4U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5U];
        status = metal_dispatch_row_groups(context, LLM_METAL_PIPELINE_RMS_NORM_BACKWARD,
                                           command_buffer, encoder, outer_count, row_width);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(input_gradient, outer_count * row_width) != 0 &&
                       metal_buffer_values_are_finite(weight_gradient, row_width) != 0
                   ? LLM_OK
                   : LLM_NUMERICAL_ERROR;
    }
}

static llm_status metal_rope(void *opaque_context, llm_metal_pipeline pipeline, const float *input,
                             const float *cos_table, const float *sin_table, size_t batch_count,
                             size_t sequence_length, size_t head_count, size_t head_dimension,
                             float *output) {
    if (opaque_context == NULL || input == NULL || cos_table == NULL || sin_table == NULL ||
        output == NULL || head_dimension % 2U != 0U || batch_count == 0U || sequence_length == 0U ||
        head_count == 0U || head_dimension == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t pairs_per_head = head_dimension / 2U;
    if (batch_count > SIZE_MAX / sequence_length ||
        batch_count * sequence_length > SIZE_MAX / head_count ||
        batch_count * sequence_length * head_count > SIZE_MAX / pairs_per_head) {
        return LLM_OVERFLOW;
    }
    const size_t pair_count = batch_count * sequence_length * head_count * pairs_per_head;
    metal_rope_parameters parameters = {0};
    if (metal_size_to_u32(sequence_length, &parameters.sequence_length) == 0 ||
        metal_size_to_u32(head_count, &parameters.head_count) == 0 ||
        metal_size_to_u32(pairs_per_head, &parameters.pairs_per_head) == 0 ||
        metal_size_to_u32(pair_count, &(uint32_t){0}) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status = metal_begin_compute(context, pipeline, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(cos_table) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(sin_table) offset:0U atIndex:2U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:3U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4U];
        status = metal_dispatch_1d(context, pipeline, command_buffer, encoder, pair_count);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(output, pair_count * 2U) != 0 ? LLM_OK
                                                                            : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_rope_f32(void *context, const float *input, const float *cos_table,
                              const float *sin_table, size_t batch_count, size_t sequence_length,
                              size_t head_count, size_t head_dimension, float *output) {
    return metal_rope(context, LLM_METAL_PIPELINE_ROPE, input, cos_table, sin_table, batch_count,
                      sequence_length, head_count, head_dimension, output);
}

llm_status llm_metal_rope_backward_f32(void *context, const float *output_gradient,
                                       const float *cos_table, const float *sin_table,
                                       size_t batch_count, size_t sequence_length,
                                       size_t head_count, size_t head_dimension,
                                       float *input_gradient) {
    return metal_rope(context, LLM_METAL_PIPELINE_ROPE_BACKWARD, output_gradient, cos_table,
                      sin_table, batch_count, sequence_length, head_count, head_dimension,
                      input_gradient);
}

static llm_status metal_attention_parameters_create(
    size_t batch_count, size_t sequence_length, size_t query_head_count,
    size_t key_value_head_count, size_t head_dimension, float scale,
    metal_attention_parameters *out_parameters, size_t *out_query_rows, size_t *out_query_values,
    size_t *out_key_value_values) {
    if (out_parameters == NULL || out_query_rows == NULL || out_query_values == NULL ||
        out_key_value_values == NULL || !isfinite(scale) || scale <= 0.0F || batch_count == 0U ||
        sequence_length == 0U || query_head_count == 0U || key_value_head_count == 0U ||
        head_dimension == 0U || query_head_count % key_value_head_count != 0U ||
        metal_size_to_u32(batch_count, &out_parameters->batch_count) == 0 ||
        metal_size_to_u32(sequence_length, &out_parameters->sequence_length) == 0 ||
        metal_size_to_u32(query_head_count, &out_parameters->query_head_count) == 0 ||
        metal_size_to_u32(key_value_head_count, &out_parameters->key_value_head_count) == 0 ||
        metal_size_to_u32(head_dimension, &out_parameters->head_dimension) == 0 ||
        batch_count > SIZE_MAX / sequence_length ||
        batch_count * sequence_length > SIZE_MAX / query_head_count ||
        batch_count * sequence_length * query_head_count > SIZE_MAX / head_dimension ||
        batch_count * sequence_length > SIZE_MAX / key_value_head_count ||
        batch_count * sequence_length * key_value_head_count > SIZE_MAX / head_dimension) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_query_rows = batch_count * sequence_length * query_head_count;
    *out_query_values = *out_query_rows * head_dimension;
    *out_key_value_values = batch_count * sequence_length * key_value_head_count * head_dimension;
    if (metal_size_to_u32(*out_query_rows, &(uint32_t){0}) == 0) {
        return LLM_OVERFLOW;
    }
    out_parameters->scale = scale;
    return LLM_OK;
}

llm_status llm_metal_attention_forward_f32(void *opaque_context, const float *query,
                                           const float *key, const float *value, float scale,
                                           size_t batch_count, size_t sequence_length,
                                           size_t query_head_count, size_t key_value_head_count,
                                           size_t head_dimension, float *output) {
    if (opaque_context == NULL || query == NULL || key == NULL || value == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_attention_parameters parameters = {0};
    size_t query_rows = 0U, query_values = 0U, key_value_values = 0U;
    llm_status status = metal_attention_parameters_create(
        batch_count, sequence_length, query_head_count, key_value_head_count, head_dimension, scale,
        &parameters, &query_rows, &query_values, &key_value_values);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status = metal_begin_compute(context, LLM_METAL_PIPELINE_ATTENTION_FORWARD, &command_buffer,
                                     &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(query) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(key) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(value) offset:0U atIndex:2U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:3U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4U];
        status = metal_dispatch_attention_groups(context, LLM_METAL_PIPELINE_ATTENTION_FORWARD,
                                                 command_buffer, encoder, query_rows,
                                                 head_dimension, 32U + head_dimension);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(output, query_values) != 0 ? LLM_OK
                                                                         : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_attention_backward_f32(void *opaque_context, const float *query,
                                            const float *key, const float *value,
                                            const float *output_gradient, float scale,
                                            size_t batch_count, size_t sequence_length,
                                            size_t query_head_count, size_t key_value_head_count,
                                            size_t head_dimension, float *query_gradient,
                                            float *key_gradient, float *value_gradient) {
    if (opaque_context == NULL || query == NULL || key == NULL || value == NULL ||
        output_gradient == NULL || query_gradient == NULL || key_gradient == NULL ||
        value_gradient == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_attention_parameters parameters = {0};
    size_t query_rows = 0U, query_values = 0U, key_value_values = 0U;
    llm_status status = metal_attention_parameters_create(
        batch_count, sequence_length, query_head_count, key_value_head_count, head_dimension, scale,
        &parameters, &query_rows, &query_values, &key_value_values);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    status = llm_metal_zero(context, key_gradient, key_value_values * sizeof(float));
    if (status == LLM_OK) {
        status = llm_metal_zero(context, value_gradient, key_value_values * sizeof(float));
    }
    if (status != LLM_OK) {
        return status;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status = metal_begin_compute(context, LLM_METAL_PIPELINE_ATTENTION_BACKWARD,
                                     &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(query) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(key) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(value) offset:0U atIndex:2U];
        [encoder setBuffer:metal_buffer_handle(output_gradient) offset:0U atIndex:3U];
        [encoder setBuffer:metal_buffer_handle(query_gradient) offset:0U atIndex:4U];
        [encoder setBuffer:metal_buffer_handle(key_gradient) offset:0U atIndex:5U];
        [encoder setBuffer:metal_buffer_handle(value_gradient) offset:0U atIndex:6U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:7U];
        if (sequence_length > (SIZE_MAX - 1U) / 2U) {
            metal_end_compute(context, encoder);
            return LLM_OVERFLOW;
        }
        status = metal_dispatch_attention_groups(context, LLM_METAL_PIPELINE_ATTENTION_BACKWARD,
                                                 command_buffer, encoder, query_rows,
                                                 head_dimension, sequence_length * 2U + 1U);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(query_gradient, query_values) != 0 &&
                       metal_buffer_values_are_finite(key_gradient, key_value_values) != 0 &&
                       metal_buffer_values_are_finite(value_gradient, key_value_values) != 0
                   ? LLM_OK
                   : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_adamw_update_f32(void *opaque_context, float *parameter, float *gradient,
                                      float *first_moment, float *second_moment, size_t value_count,
                                      float learning_rate, float beta1, float beta2, float epsilon,
                                      float weight_decay, float gradient_scale,
                                      unsigned long long step, int zero_gradient) {
    if (opaque_context == NULL || parameter == NULL || gradient == NULL || first_moment == NULL ||
        second_moment == NULL || !isfinite(learning_rate) || learning_rate < 0.0F ||
        !isfinite(beta1) || beta1 < 0.0F || beta1 >= 1.0F || !isfinite(beta2) || beta2 < 0.0F ||
        beta2 >= 1.0F || !isfinite(epsilon) || epsilon <= 0.0F || !isfinite(weight_decay) ||
        weight_decay < 0.0F || !isfinite(gradient_scale) || step == 0ULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_adamw_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    const float first_bias = 1.0F - (float)pow((double)beta1, (double)step);
    const float second_bias = 1.0F - (float)pow((double)beta2, (double)step);
    if (!isfinite(first_bias) || !isfinite(second_bias) || first_bias <= 0.0F ||
        second_bias <= 0.0F) {
        return LLM_NUMERICAL_ERROR;
    }
    parameters.learning_rate = learning_rate;
    parameters.beta1 = beta1;
    parameters.beta2 = beta2;
    parameters.epsilon = epsilon;
    parameters.weight_decay = weight_decay;
    parameters.gradient_scale = gradient_scale;
    parameters.inverse_first_bias = 1.0F / first_bias;
    parameters.inverse_second_bias = 1.0F / second_bias;
    parameters.zero_gradient = zero_gradient != 0 ? 1U : 0U;
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_ADAMW, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(parameter) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(gradient) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(first_moment) offset:0U atIndex:2U];
        [encoder setBuffer:metal_buffer_handle(second_moment) offset:0U atIndex:3U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4U];
        status = metal_dispatch_1d(context, LLM_METAL_PIPELINE_ADAMW, command_buffer, encoder,
                                   value_count);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(parameter, value_count) != 0 &&
                       metal_buffer_values_are_finite(first_moment, value_count) != 0 &&
                       metal_buffer_values_are_finite(second_moment, value_count) != 0
                   ? LLM_OK
                   : LLM_NUMERICAL_ERROR;
    }
}

static llm_status metal_reduce(void *opaque_context, llm_metal_pipeline pipeline,
                               const float *input, float *output, size_t outer_count,
                               size_t reduction_size) {
    if (opaque_context == NULL || input == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_reduction_parameters parameters = {0};
    if (metal_size_to_u32(outer_count, &parameters.outer_count) == 0 ||
        metal_size_to_u32(reduction_size, &parameters.reduction_size) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        llm_status status = metal_begin_compute(context, pipeline, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:1U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
        status = metal_dispatch_row_groups(context, pipeline, command_buffer, encoder, outer_count,
                                           reduction_size);
        if (status != LLM_OK) {
            return status;
        }
        if (context->batch_active != 0) {
            return LLM_OK;
        }
        return metal_buffer_values_are_finite(output, outer_count) != 0 ? LLM_OK
                                                                        : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_reduce_sum_last_f32(void *context, const float *input, float *output,
                                         size_t outer_count, size_t reduction_size) {
    return metal_reduce(context, LLM_METAL_PIPELINE_REDUCE_SUM, input, output, outer_count,
                        reduction_size);
}

llm_status llm_metal_reduce_max_last_f32(void *context, const float *input, float *output,
                                         size_t outer_count, size_t reduction_size) {
    return metal_reduce(context, LLM_METAL_PIPELINE_REDUCE_MAX, input, output, outer_count,
                        reduction_size);
}

llm_status llm_metal_reduce_mean_square_last_f32(void *context, const float *input, float *output,
                                                 size_t outer_count, size_t reduction_size) {
    return metal_reduce(context, LLM_METAL_PIPELINE_REDUCE_MEAN_SQUARE, input, output, outer_count,
                        reduction_size);
}

llm_status llm_metal_accumulate_sum_squares_f32(void *opaque_context, const float *input,
                                                float *accumulator, size_t value_count) {
    if (opaque_context == NULL || input == NULL || accumulator == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_elementwise_parameters parameters = {0};
    if (metal_size_to_u32(value_count, &parameters.count) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        const llm_metal_pipeline pipeline = LLM_METAL_PIPELINE_ACCUMULATE_SUM_SQUARES;
        llm_status status = metal_begin_compute(context, pipeline, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(accumulator) offset:0U atIndex:1U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
        const size_t group_count = (value_count + LLM_METAL_SUM_SQUARES_VALUES_PER_GROUP - 1U) /
                                   LLM_METAL_SUM_SQUARES_VALUES_PER_GROUP;
        status = metal_dispatch_row_groups(context, pipeline, command_buffer, encoder, group_count,
                                           LLM_METAL_SUM_SQUARES_VALUES_PER_GROUP);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(accumulator, 1U) != 0 ? LLM_OK : LLM_NUMERICAL_ERROR;
    }
}

static size_t metal_matmul_pipeline_tile(llm_metal_pipeline pipeline) {
    return pipeline == LLM_METAL_PIPELINE_MATMUL_LARGE ? 32U : 16U;
}

static llm_status metal_run_matmul_pipeline(llm_metal_context *context, llm_metal_pipeline pipeline,
                                            const void *left, const void *right, float *output,
                                            const metal_matmul_parameters *parameters) {
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    llm_status status = metal_begin_compute(context, pipeline, &command_buffer, &encoder);
    if (status != LLM_OK) {
        return status;
    }
    id<MTLComputePipelineState> state = context->pipelines[pipeline];
    const int use_simdgroup = pipeline == LLM_METAL_PIPELINE_MATMUL_SIMDGROUP;
    const NSUInteger required_threads = use_simdgroup != 0 ? 128U : 256U;
    if (state == nil || [state maxTotalThreadsPerThreadgroup] < required_threads) {
        if (context->batch_active != 0) {
            llm_metal_close_batch_compute_encoder(context);
        } else {
            [encoder endEncoding];
        }
        return LLM_BACKEND_ERROR;
    }
    const size_t tile = metal_matmul_pipeline_tile(pipeline);
    [encoder setBuffer:metal_buffer_handle(left) offset:0U atIndex:0U];
    [encoder setBuffer:metal_buffer_handle(right) offset:0U atIndex:1U];
    [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:2U];
    [encoder setBytes:parameters length:sizeof(*parameters) atIndex:3U];
    const MTLSize threadgroups = MTLSizeMake(((size_t)parameters->columns + tile - 1U) / tile,
                                             ((size_t)parameters->rows + tile - 1U) / tile, 1U);
    const MTLSize threads =
        use_simdgroup != 0 ? MTLSizeMake(128U, 1U, 1U) : MTLSizeMake(16U, 16U, 1U);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
    metal_end_compute(context, encoder);
    return llm_metal_submit(context, command_buffer);
}

static int metal_matmul_cache_lookup(llm_metal_context *context, llm_dtype dtype, uint32_t rows,
                                     uint32_t inner_size, uint32_t columns,
                                     llm_metal_pipeline *out_pipeline) {
    int found = 0;
    (void)pthread_mutex_lock(&context->buffer_mutex);
    for (const llm_metal_matmul_tuning *entry = context->matmul_tunings; entry != NULL;
         entry = entry->next) {
        if (entry->dtype == dtype && entry->rows == rows && entry->inner_size == inner_size &&
            entry->columns == columns) {
            *out_pipeline = entry->pipeline;
            found = 1;
            break;
        }
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);
    return found;
}

static void metal_matmul_cache_insert(llm_metal_context *context, llm_dtype dtype, uint32_t rows,
                                      uint32_t inner_size, uint32_t columns,
                                      llm_metal_pipeline pipeline) {
    llm_metal_matmul_tuning *entry = malloc(sizeof(*entry));
    if (entry == NULL) {
        return;
    }
    *entry = (llm_metal_matmul_tuning){
        .rows = rows,
        .inner_size = inner_size,
        .columns = columns,
        .dtype = dtype,
        .pipeline = pipeline,
    };
    (void)pthread_mutex_lock(&context->buffer_mutex);
    if (context->matmul_tuning_count < LLM_METAL_AUTOTUNE_CACHE_LIMIT) {
        entry->next = context->matmul_tunings;
        context->matmul_tunings = entry;
        ++context->matmul_tuning_count;
        entry = NULL;
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);
    free(entry);
}

static llm_status metal_select_matmul_pipeline(llm_metal_context *context, llm_dtype dtype,
                                               const void *left, const void *right, float *output,
                                               const metal_matmul_parameters *parameters,
                                               const llm_metal_pipeline *candidates,
                                               size_t candidate_count,
                                               llm_metal_pipeline *out_pipeline) {
    const char *override = getenv("LLM_METAL_MATMUL_PIPELINE");
    if (override != NULL) {
        const llm_metal_pipeline requested =
            strcmp(override, "tile16") == 0      ? candidates[0]
            : strcmp(override, "tile32") == 0    ? candidates[candidate_count > 1U ? 1U : 0U]
            : strcmp(override, "simdgroup") == 0 ? LLM_METAL_PIPELINE_MATMUL_SIMDGROUP
                                                 : LLM_METAL_PIPELINE_COUNT;
        for (size_t index = 0U; index < candidate_count; ++index) {
            if (candidates[index] == requested) {
                *out_pipeline = requested;
                return LLM_OK;
            }
        }
    }
    if (metal_matmul_cache_lookup(context, dtype, parameters->rows, parameters->inner_size,
                                  parameters->columns, out_pipeline) != 0) {
        return LLM_OK;
    }
    if (context->batch_active != 0) {
        *out_pipeline = candidate_count > 1U ? candidates[1] : candidates[0];
        return LLM_OK;
    }

    double samples[3][LLM_METAL_AUTOTUNE_TRIALS] = {{0}};
    for (size_t index = 0U; index < candidate_count; ++index) {
        const llm_status status =
            metal_run_matmul_pipeline(context, candidates[index], left, right, output, parameters);
        if (status != LLM_OK) {
            return status;
        }
    }
    for (size_t trial = 0U; trial < LLM_METAL_AUTOTUNE_TRIALS; ++trial) {
        for (size_t offset = 0U; offset < candidate_count; ++offset) {
            const size_t index = (trial + offset) % candidate_count;
            const NSTimeInterval started = [NSDate timeIntervalSinceReferenceDate];
            const llm_status status = metal_run_matmul_pipeline(context, candidates[index], left,
                                                                right, output, parameters);
            if (status != LLM_OK) {
                return status;
            }
            const double elapsed = [NSDate timeIntervalSinceReferenceDate] - started;
            samples[index][trial] = context->metrics.last_gpu_seconds > 0.0
                                        ? context->metrics.last_gpu_seconds
                                        : elapsed;
        }
    }

    double best_seconds = INFINITY;
    llm_metal_pipeline best_pipeline = candidates[0];
    for (size_t index = 0U; index < candidate_count; ++index) {
        double first = samples[index][0];
        double second = samples[index][1];
        double third = samples[index][2];
        if (first > second) {
            const double temporary = first;
            first = second;
            second = temporary;
        }
        if (second > third) {
            const double temporary = second;
            second = third;
            third = temporary;
        }
        const double candidate_seconds = first > second ? first : second;
        if (candidate_seconds < best_seconds) {
            best_seconds = candidate_seconds;
            best_pipeline = candidates[index];
        }
    }
    metal_matmul_cache_insert(context, dtype, parameters->rows, parameters->inner_size,
                              parameters->columns, best_pipeline);
    *out_pipeline = best_pipeline;
    return LLM_OK;
}

static llm_status metal_matmul(void *opaque_context, const void *left, const void *right,
                               float *output, size_t rows, size_t inner_size, size_t columns) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_matmul_parameters parameters = {0};
    if (metal_size_to_u32(rows, &parameters.rows) == 0 ||
        metal_size_to_u32(inner_size, &parameters.inner_size) == 0 ||
        metal_size_to_u32(columns, &parameters.columns) == 0 || rows > SIZE_MAX / columns) {
        return LLM_OVERFLOW;
    }
    /*
     * The output projection is very wide ([B*T,C] x [C,V]). MPS is both more
     * reliable for its non-tile-aligned tail and a better fit for this GEMM;
     * retain custom kernels for the compact Transformer projections.
     */
    if (columns >= 1024U) {
        return llm_metal_matmul_ex_f32(opaque_context, left, right, output, rows, inner_size,
                                       inner_size, columns, 0, 0);
    }
    llm_metal_pipeline candidates[3] = {0};
    size_t candidate_count = 0U;
    candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL;
    candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_LARGE;
    llm_metal_context *context = opaque_context;
    if (context->pipelines[LLM_METAL_PIPELINE_MATMUL_SIMDGROUP] != nil && rows % 16U == 0U &&
        inner_size % 16U == 0U && columns % 16U == 0U) {
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_SIMDGROUP;
    }

    @autoreleasepool {
        llm_metal_pipeline selected = candidates[0];
        llm_status status =
            metal_select_matmul_pipeline(context, LLM_DTYPE_F32, left, right, output, &parameters,
                                         candidates, candidate_count, &selected);
        if (status == LLM_OK) {
            status = metal_run_matmul_pipeline(context, selected, left, right, output, &parameters);
        }
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(output, rows * columns) != 0 ? LLM_OK
                                                                           : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_matmul_f32(void *context, const float *left, const float *right, float *output,
                                size_t rows, size_t inner_size, size_t columns) {
    return metal_matmul(context, left, right, output, rows, inner_size, columns);
}

static int metal_mps_size(size_t value, NSUInteger *out_value) {
    if (value == 0U || value > NSUIntegerMax || out_value == NULL) {
        return 0;
    }
    *out_value = (NSUInteger)value;
    return 1;
}

static MPSMatrix *metal_mps_wrap_matrix(const void *memory, size_t rows, size_t columns) {
    NSUInteger mps_rows = 0U;
    NSUInteger mps_columns = 0U;
    if (memory == NULL || metal_mps_size(rows, &mps_rows) == 0 ||
        metal_mps_size(columns, &mps_columns) == 0 || columns > SIZE_MAX / sizeof(float)) {
        return nil;
    }
    const llm_metal_buffer *buffer = llm_metal_buffer_from_const_memory(memory);
    if (rows > SIZE_MAX / (columns * sizeof(float)) ||
        buffer->byte_count < rows * columns * sizeof(float)) {
        return nil;
    }
    MPSMatrixDescriptor *descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:mps_rows
                                              columns:mps_columns
                                             rowBytes:mps_columns * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    return [[[MPSMatrix alloc] initWithBuffer:buffer->handle descriptor:descriptor] autorelease];
}

static MPSMatrixMultiplication *
metal_mps_gemm_get_or_create(llm_metal_context *context, uint32_t left_rows, uint32_t left_columns,
                             uint32_t right_rows, uint32_t right_columns, int transpose_left,
                             int transpose_right) {
    if (context == NULL) {
        return nil;
    }
    const uint8_t left_transposed = transpose_left != 0 ? 1U : 0U;
    const uint8_t right_transposed = transpose_right != 0 ? 1U : 0U;
    (void)pthread_mutex_lock(&context->buffer_mutex);
    for (llm_metal_mps_gemm *entry = context->mps_gemms; entry != NULL; entry = entry->next) {
        if (entry->left_rows == left_rows && entry->left_columns == left_columns &&
            entry->right_rows == right_rows && entry->right_columns == right_columns &&
            entry->transpose_left == left_transposed &&
            entry->transpose_right == right_transposed) {
            MPSMatrixMultiplication *kernel = [entry->kernel retain];
            (void)pthread_mutex_unlock(&context->buffer_mutex);
            return [kernel autorelease];
        }
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);

    const uint32_t result_rows = transpose_left != 0 ? left_columns : left_rows;
    const uint32_t interior_columns = transpose_left != 0 ? left_rows : left_columns;
    const uint32_t result_columns = transpose_right != 0 ? right_rows : right_columns;
    MPSMatrixMultiplication *kernel =
        [[MPSMatrixMultiplication alloc] initWithDevice:context->device
                                          transposeLeft:transpose_left != 0
                                         transposeRight:transpose_right != 0
                                             resultRows:result_rows
                                          resultColumns:result_columns
                                        interiorColumns:interior_columns
                                                  alpha:1.0
                                                   beta:0.0];
    if (kernel == nil) {
        return nil;
    }
    llm_metal_mps_gemm *entry = calloc(1U, sizeof(*entry));
    if (entry == NULL) {
        return [kernel autorelease];
    }
    *entry = (llm_metal_mps_gemm){
        .left_rows = left_rows,
        .left_columns = left_columns,
        .right_rows = right_rows,
        .right_columns = right_columns,
        .transpose_left = left_transposed,
        .transpose_right = right_transposed,
        .kernel = [kernel retain],
    };
    (void)pthread_mutex_lock(&context->buffer_mutex);
    if (context->mps_gemm_count < LLM_METAL_MPS_GEMM_CACHE_LIMIT) {
        entry->next = context->mps_gemms;
        context->mps_gemms = entry;
        ++context->mps_gemm_count;
        entry = NULL;
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);
    if (entry != NULL) {
        [entry->kernel release];
        free(entry);
    }
    return [kernel autorelease];
}

llm_status llm_metal_matmul_ex_f32(void *opaque_context, const float *left, const float *right,
                                   float *output, size_t left_rows, size_t left_columns,
                                   size_t right_rows, size_t right_columns, int transpose_left,
                                   int transpose_right) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL ||
        (transpose_left != 0 && transpose_left != 1) ||
        (transpose_right != 0 && transpose_right != 1)) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t result_rows = transpose_left != 0 ? left_columns : left_rows;
    const size_t interior_columns = transpose_left != 0 ? left_rows : left_columns;
    const size_t right_interior_columns = transpose_right != 0 ? right_columns : right_rows;
    const size_t result_columns = transpose_right != 0 ? right_rows : right_columns;
    uint32_t checked_left_rows = 0U;
    uint32_t checked_left_columns = 0U;
    uint32_t checked_right_rows = 0U;
    uint32_t checked_right_columns = 0U;
    if (interior_columns != right_interior_columns || result_rows > SIZE_MAX / result_columns ||
        metal_size_to_u32(left_rows, &checked_left_rows) == 0 ||
        metal_size_to_u32(left_columns, &checked_left_columns) == 0 ||
        metal_size_to_u32(right_rows, &checked_right_rows) == 0 ||
        metal_size_to_u32(right_columns, &checked_right_columns) == 0) {
        return LLM_OVERFLOW;
    }
    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        MPSMatrix *left_matrix = metal_mps_wrap_matrix(left, left_rows, left_columns);
        MPSMatrix *right_matrix = metal_mps_wrap_matrix(right, right_rows, right_columns);
        MPSMatrix *output_matrix = metal_mps_wrap_matrix(output, result_rows, result_columns);
        MPSMatrixMultiplication *kernel = metal_mps_gemm_get_or_create(
            context, checked_left_rows, checked_left_columns, checked_right_rows,
            checked_right_columns, transpose_left, transpose_right);
        if (left_matrix == nil || right_matrix == nil || output_matrix == nil || kernel == nil) {
            return LLM_BACKEND_ERROR;
        }
        llm_metal_close_batch_compute_encoder(context);
        id<MTLCommandBuffer> command_buffer = llm_metal_acquire_command_buffer(context);
        if (command_buffer == nil) {
            return LLM_BACKEND_ERROR;
        }
        [kernel encodeToCommandBuffer:command_buffer
                           leftMatrix:left_matrix
                          rightMatrix:right_matrix
                         resultMatrix:output_matrix];
        ++context->metrics.kernel_dispatches;
        const llm_status status = llm_metal_submit(context, command_buffer);
        if (status != LLM_OK || context->batch_active != 0) {
            return status;
        }
        return metal_buffer_values_are_finite(output, result_rows * result_columns) != 0
                   ? LLM_OK
                   : LLM_NUMERICAL_ERROR;
    }
}

static int metal_indices_are_valid(llm_metal_context *context, const uint32_t *indices,
                                   size_t index_count, size_t row_count) {
    if (llm_metal_flush(context) != LLM_OK) {
        return 0;
    }
    const uint32_t *values = [metal_buffer_handle(indices) contents];
    for (size_t index = 0U; index < index_count; ++index) {
        if ((size_t)values[index] >= row_count) {
            return 0;
        }
    }
    return 1;
}

static llm_status metal_gather_parameters_create(size_t row_count, size_t row_width,
                                                 size_t index_count,
                                                 metal_gather_parameters *out_parameters) {
    if (out_parameters == NULL || row_count == 0U || row_width == 0U || index_count == 0U ||
        index_count > UINT32_MAX / row_width ||
        metal_size_to_u32(row_count, &out_parameters->row_count) == 0 ||
        metal_size_to_u32(row_width, &out_parameters->row_width) == 0 ||
        metal_size_to_u32(index_count, &out_parameters->index_count) == 0) {
        return LLM_OVERFLOW;
    }
    return LLM_OK;
}

llm_status llm_metal_gather_rows_f32(void *opaque_context, const float *table, size_t row_count,
                                     size_t row_width, const uint32_t *indices, size_t index_count,
                                     float *output) {
    if (opaque_context == NULL || table == NULL || indices == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_gather_parameters parameters = {0};
    llm_status status =
        metal_gather_parameters_create(row_count, row_width, index_count, &parameters);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    if (metal_indices_are_valid(context, indices, index_count, row_count) == 0) {
        return LLM_INVALID_INDEX;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status = metal_begin_compute(context, LLM_METAL_PIPELINE_GATHER, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(table) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(indices) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:2U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_GATHER, command_buffer, encoder,
                                 index_count * row_width);
    }
}

llm_status llm_metal_scatter_add_rows_f32(void *opaque_context, const float *source,
                                          const uint32_t *indices, size_t index_count,
                                          size_t row_width, size_t row_count, float *table) {
    if (opaque_context == NULL || source == NULL || indices == NULL || table == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_gather_parameters parameters = {0};
    llm_status status =
        metal_gather_parameters_create(row_count, row_width, index_count, &parameters);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    if (metal_indices_are_valid(context, indices, index_count, row_count) == 0) {
        return LLM_INVALID_INDEX;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status =
            metal_begin_compute(context, LLM_METAL_PIPELINE_SCATTER_ADD, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(source) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(indices) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(table) offset:0U atIndex:2U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
        return metal_dispatch_1d(context, LLM_METAL_PIPELINE_SCATTER_ADD, command_buffer, encoder,
                                 index_count * row_width);
    }
}

llm_status llm_metal_softmax_last_f32(void *opaque_context, const float *input, float *output,
                                      size_t outer_count, size_t row_width) {
    llm_status status = metal_reduce(opaque_context, LLM_METAL_PIPELINE_SOFTMAX, input, output,
                                     outer_count, row_width);
    if (status != LLM_OK) {
        return status;
    }
    llm_metal_context *context = opaque_context;
    if (context->batch_active != 0) {
        return LLM_OK;
    }
    if (outer_count > SIZE_MAX / row_width) {
        return LLM_OVERFLOW;
    }
    return metal_buffer_values_are_finite(output, outer_count * row_width) != 0
               ? LLM_OK
               : LLM_NUMERICAL_ERROR;
}

static llm_status
metal_cross_entropy_parameters_create(llm_metal_context *context, const uint32_t *targets,
                                      size_t row_count, size_t vocabulary_size,
                                      metal_cross_entropy_parameters *out_parameters) {
    if (targets == NULL || out_parameters == NULL ||
        metal_size_to_u32(row_count, &out_parameters->row_count) == 0 ||
        metal_size_to_u32(vocabulary_size, &out_parameters->vocabulary_size) == 0) {
        return LLM_OVERFLOW;
    }
    return metal_indices_are_valid(context, targets, row_count, vocabulary_size) != 0
               ? LLM_OK
               : LLM_INVALID_INDEX;
}

static llm_status metal_cross_entropy_dispatch(void *opaque_context, llm_metal_pipeline pipeline,
                                               const float *logits, const uint32_t *targets,
                                               float *output, size_t row_count,
                                               size_t vocabulary_size, size_t output_count) {
    if (opaque_context == NULL || logits == NULL || targets == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_metal_context *context = opaque_context;
    metal_cross_entropy_parameters parameters = {0};
    llm_status status = metal_cross_entropy_parameters_create(context, targets, row_count,
                                                              vocabulary_size, &parameters);
    if (status != LLM_OK) {
        return status;
    }
    if (pipeline == LLM_METAL_PIPELINE_CROSS_ENTROPY_FORWARD) {
        /* The kernel accumulates into this scalar, so it must start at zero.
           Clearing it with a queued blit keeps the order against any work
           already encoded in an open batch; a plain host store would not. */
        status = llm_metal_zero(context, output, sizeof(float));
        if (status != LLM_OK) {
            return status;
        }
    }
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        status = metal_begin_compute(context, pipeline, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder setBuffer:metal_buffer_handle(logits) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(targets) offset:0U atIndex:1U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:2U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
        status = metal_dispatch_row_groups(context, pipeline, command_buffer, encoder, row_count,
                                           vocabulary_size);
        if (status != LLM_OK) {
            return status;
        }
        if (context->batch_active != 0) {
            return LLM_OK;
        }
        return metal_buffer_values_are_finite(output, output_count) != 0 ? LLM_OK
                                                                         : LLM_NUMERICAL_ERROR;
    }
}

llm_status llm_metal_cross_entropy_forward_f32(void *context, const float *logits,
                                               const uint32_t *targets, size_t row_count,
                                               size_t vocabulary_size, float *loss) {
    return metal_cross_entropy_dispatch(context, LLM_METAL_PIPELINE_CROSS_ENTROPY_FORWARD, logits,
                                        targets, loss, row_count, vocabulary_size, 1U);
}

llm_status llm_metal_cross_entropy_backward_f32(void *context, const float *logits,
                                                const uint32_t *targets, size_t row_count,
                                                size_t vocabulary_size, float *gradient) {
    if (row_count > SIZE_MAX / vocabulary_size) {
        return LLM_OVERFLOW;
    }
    return metal_cross_entropy_dispatch(context, LLM_METAL_PIPELINE_CROSS_ENTROPY_BACKWARD, logits,
                                        targets, gradient, row_count, vocabulary_size,
                                        row_count * vocabulary_size);
}
