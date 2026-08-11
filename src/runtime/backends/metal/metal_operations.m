#import <Foundation/Foundation.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "metal_internal.h"

#define LLM_METAL_MAX_THREADS_1D 256U
#define LLM_METAL_MATMUL_TILE 16U
#define LLM_METAL_ELEMENTWISE_VECTOR_WIDTH 4U

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

#define LLM_METAL_AUTOTUNE_TRIALS 3U
#define LLM_METAL_AUTOTUNE_CACHE_LIMIT 128U

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

llm_status llm_metal_cast(void *opaque_context, const void *input, llm_dtype input_dtype,
                          void *output, llm_dtype output_dtype, size_t value_count) {
    if (opaque_context == NULL || input == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_metal_pipeline pipeline = LLM_METAL_PIPELINE_COUNT;
    if (input_dtype == LLM_DTYPE_F32 && output_dtype == LLM_DTYPE_F16) {
        pipeline = LLM_METAL_PIPELINE_CAST_F32_F16;
    } else if (input_dtype == LLM_DTYPE_F16 && output_dtype == LLM_DTYPE_F32) {
        pipeline = LLM_METAL_PIPELINE_CAST_F16_F32;
    } else if (input_dtype == LLM_DTYPE_F32 && output_dtype == LLM_DTYPE_BF16) {
        pipeline = LLM_METAL_PIPELINE_CAST_F32_BF16;
    } else if (input_dtype == LLM_DTYPE_BF16 && output_dtype == LLM_DTYPE_F32) {
        pipeline = LLM_METAL_PIPELINE_CAST_BF16_F32;
    } else {
        return LLM_UNSUPPORTED_DTYPE;
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
        [encoder setBuffer:metal_buffer_handle(input) offset:0U atIndex:0U];
        [encoder setBuffer:metal_buffer_handle(output) offset:0U atIndex:1U];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
        return metal_dispatch_1d(context, pipeline, command_buffer, encoder,
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

static size_t metal_matmul_pipeline_tile(llm_metal_pipeline pipeline) {
    return pipeline == LLM_METAL_PIPELINE_MATMUL_LARGE ||
                   pipeline == LLM_METAL_PIPELINE_MATMUL_F16_LARGE ||
                   pipeline == LLM_METAL_PIPELINE_MATMUL_BF16_LARGE
               ? 32U
               : 16U;
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
                               llm_dtype dtype, float *output, size_t rows, size_t inner_size,
                               size_t columns) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    metal_matmul_parameters parameters = {0};
    if (metal_size_to_u32(rows, &parameters.rows) == 0 ||
        metal_size_to_u32(inner_size, &parameters.inner_size) == 0 ||
        metal_size_to_u32(columns, &parameters.columns) == 0 || rows > SIZE_MAX / columns) {
        return LLM_OVERFLOW;
    }
    llm_metal_pipeline candidates[3] = {0};
    size_t candidate_count = 0U;
    if (dtype == LLM_DTYPE_F32) {
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL;
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_LARGE;
        llm_metal_context *context = opaque_context;
        if (context->pipelines[LLM_METAL_PIPELINE_MATMUL_SIMDGROUP] != nil && rows % 16U == 0U &&
            inner_size % 16U == 0U && columns % 16U == 0U) {
            candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_SIMDGROUP;
        }
    } else if (dtype == LLM_DTYPE_F16) {
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_F16;
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_F16_LARGE;
    } else if (dtype == LLM_DTYPE_BF16) {
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_BF16;
        candidates[candidate_count++] = LLM_METAL_PIPELINE_MATMUL_BF16_LARGE;
    } else {
        return LLM_UNSUPPORTED_DTYPE;
    }

    llm_metal_context *context = opaque_context;
    @autoreleasepool {
        llm_metal_pipeline selected = candidates[0];
        llm_status status =
            metal_select_matmul_pipeline(context, dtype, left, right, output, &parameters,
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
    return metal_matmul(context, left, right, LLM_DTYPE_F32, output, rows, inner_size, columns);
}

llm_status llm_metal_matmul_mixed_f32(void *context, const void *left, const void *right,
                                      llm_dtype input_dtype, float *output, size_t rows,
                                      size_t inner_size, size_t columns) {
    return metal_matmul(context, left, right, input_dtype, output, rows, inner_size, columns);
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
        float *loss_value = [metal_buffer_handle(output) contents];
        loss_value[0] = 0.0F;
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
