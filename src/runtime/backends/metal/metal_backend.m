#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metal_internal.h"
#include "metal_kernels_embedded.h"

static const char *const metal_pipeline_names[LLM_METAL_PIPELINE_COUNT] = {
    "llm_fill_f32",
    "llm_add_f32",
    "llm_multiply_f32",
    "llm_scale_f32",
    "llm_reduce_sum_last_f32",
    "llm_reduce_max_last_f32",
    "llm_reduce_mean_square_last_f32",
    "llm_matmul_f32",
    "llm_matmul_f32_tiled32",
    "llm_matmul_f32_simdgroup",
    "llm_gather_rows_f32",
    "llm_scatter_add_rows_f32",
    "llm_softmax_last_f32",
    "llm_cross_entropy_forward_f32",
    "llm_cross_entropy_backward_f32",
};

static char *metal_copy_device_name(id<MTLDevice> device) {
    const char *name = [[device name] UTF8String];
    if (name == NULL) {
        return NULL;
    }
    const size_t length = strlen(name);
    char *copy = malloc(length + 1U);
    if (copy != NULL) {
        (void)memcpy(copy, name, length + 1U);
    }
    return copy;
}

static void metal_release_pipelines(llm_metal_context *context) {
    for (size_t index = 0U; index < LLM_METAL_PIPELINE_COUNT; ++index) {
        [context->pipelines[index] release];
        context->pipelines[index] = nil;
    }
}

static void metal_destroy(void *opaque_context) {
    llm_metal_context *context = opaque_context;
    if (context == NULL) {
        return;
    }

    (void)llm_metal_flush(context);
    llm_metal_buffer *buffer = context->buffers;
    while (buffer != NULL) {
        llm_metal_buffer *next = buffer->next;
        [buffer->handle release];
        free(buffer);
        buffer = next;
    }
    buffer = context->cached_buffers;
    while (buffer != NULL) {
        llm_metal_buffer *next = buffer->next;
        [buffer->handle release];
        free(buffer);
        buffer = next;
    }
    llm_metal_matmul_tuning *tuning = context->matmul_tunings;
    while (tuning != NULL) {
        llm_metal_matmul_tuning *next = tuning->next;
        free(tuning);
        tuning = next;
    }
    metal_release_pipelines(context);
    [context->queue release];
    [context->device release];
    free(context->device_name);
    (void)pthread_mutex_destroy(&context->buffer_mutex);
    free(context);
}

static int metal_supports_dtype(const void *opaque_context, llm_dtype dtype) {
    return opaque_context != NULL && (dtype == LLM_DTYPE_F32 || dtype == LLM_DTYPE_U32);
}

static llm_status metal_synchronize(void *opaque_context) {
    return opaque_context == NULL ? LLM_INVALID_ARGUMENT
                                  : llm_metal_flush((llm_metal_context *)opaque_context);
}

static const llm_backend_ops *metal_backend_ops(void) {
    static const llm_backend_ops operations = {
        .destroy = metal_destroy,
        .supports_dtype = metal_supports_dtype,
        .allocate = llm_metal_allocate,
        .deallocate = llm_metal_deallocate,
        .zero = llm_metal_zero,
        .copy = llm_metal_copy,
        .fill_f32 = llm_metal_fill_f32,
        .add_f32 = llm_metal_add_f32,
        .multiply_f32 = llm_metal_multiply_f32,
        .scale_f32 = llm_metal_scale_f32,
        .reduce_sum_last_f32 = llm_metal_reduce_sum_last_f32,
        .reduce_max_last_f32 = llm_metal_reduce_max_last_f32,
        .reduce_mean_square_last_f32 = llm_metal_reduce_mean_square_last_f32,
        .matmul_f32 = llm_metal_matmul_f32,
        .gather_rows_f32 = llm_metal_gather_rows_f32,
        .scatter_add_rows_f32 = llm_metal_scatter_add_rows_f32,
        .softmax_last_f32 = llm_metal_softmax_last_f32,
        .cross_entropy_forward_f32 = llm_metal_cross_entropy_forward_f32,
        .cross_entropy_backward_f32 = llm_metal_cross_entropy_backward_f32,
        .synchronize = metal_synchronize,
    };
    return &operations;
}

static llm_status metal_create_pipelines(llm_metal_context *context) {
    const NSTimeInterval compilation_started = [NSDate timeIntervalSinceReferenceDate];
    NSError *error = nil;
    id<MTLLibrary> library = nil;
#if LLM_METAL_HAS_PRECOMPILED_LIBRARY
    dispatch_data_t library_data = dispatch_data_create(
        llm_metal_precompiled_library, llm_metal_precompiled_library_size,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (library_data != nil) {
        library = [context->device newLibraryWithData:library_data error:&error];
#if !OS_OBJECT_USE_OBJC
        dispatch_release(library_data);
#endif
    }
#endif
    if (library == nil) {
        NSString *source = [[NSString alloc] initWithBytes:llm_metal_kernel_source
                                                    length:llm_metal_kernel_source_size
                                                  encoding:NSUTF8StringEncoding];
        if (source == nil) {
            return LLM_BACKEND_ERROR;
        }
        error = nil;
        library = [context->device newLibraryWithSource:source options:nil error:&error];
        [source release];
    }
    if (library == nil) {
        fprintf(stderr, "metal: cannot compile kernel library: %s\n",
                error == nil ? "unknown error" : [[error localizedDescription] UTF8String]);
        return LLM_BACKEND_ERROR;
    }

    for (size_t index = 0U; index < LLM_METAL_PIPELINE_COUNT; ++index) {
        if (index == LLM_METAL_PIPELINE_MATMUL_SIMDGROUP) {
            continue;
        }
        NSString *function_name = [NSString stringWithUTF8String:metal_pipeline_names[index]];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            if (index == LLM_METAL_PIPELINE_MATMUL_SIMDGROUP) {
                continue;
            }
            fprintf(stderr, "metal: missing kernel function %s\n", metal_pipeline_names[index]);
            [library release];
            return LLM_BACKEND_ERROR;
        }
        context->pipelines[index] = [context->device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
        [function release];
        if (context->pipelines[index] == nil) {
            fprintf(stderr, "metal: cannot create pipeline %s: %s\n", metal_pipeline_names[index],
                    error == nil ? "unknown error" : [[error localizedDescription] UTF8String]);
            [library release];
            return LLM_BACKEND_ERROR;
        }
    }
    if ([context->device supportsFamily:MTLGPUFamilyApple7]) {
        id<MTLLibrary> simd_library = library;
#if !LLM_METAL_HAS_PRECOMPILED_LIBRARY
        NSString *simd_source = [[NSString alloc] initWithBytes:llm_metal_simdgroup_source
                                                         length:llm_metal_simdgroup_source_size
                                                       encoding:NSUTF8StringEncoding];
        simd_library = simd_source == nil ? nil
                                          : [context->device newLibraryWithSource:simd_source
                                                                          options:nil
                                                                            error:&error];
        [simd_source release];
#endif
        id<MTLFunction> function = [simd_library newFunctionWithName:@"llm_matmul_f32_simdgroup"];
        if (function != nil) {
            context->pipelines[LLM_METAL_PIPELINE_MATMUL_SIMDGROUP] =
                [context->device newComputePipelineStateWithFunction:function error:&error];
            [function release];
        }
#if !LLM_METAL_HAS_PRECOMPILED_LIBRARY
        [simd_library release];
#endif
    }
    [library release];
    context->metrics.pipeline_compilation_seconds =
        [NSDate timeIntervalSinceReferenceDate] - compilation_started;
    return LLM_OK;
}

int llm_backend_metal_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        const int available = device != nil;
        [device release];
        return available;
    }
}

llm_status llm_backend_metal_create(llm_backend **out_backend) {
    if (out_backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_backend = NULL;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return LLM_UNSUPPORTED_DEVICE;
        }

        llm_backend *backend = calloc(1U, sizeof(*backend));
        llm_metal_context *context = calloc(1U, sizeof(*context));
        if (backend == NULL || context == NULL) {
            free(backend);
            free(context);
            [device release];
            return LLM_ALLOCATION_FAILED;
        }
        if (pthread_mutex_init(&context->buffer_mutex, NULL) != 0) {
            free(context);
            free(backend);
            [device release];
            return LLM_BACKEND_ERROR;
        }

        context->device = device;
        context->queue = [device newCommandQueue];
        context->device_name = metal_copy_device_name(device);
        if (context->queue == nil || context->device_name == NULL) {
            metal_destroy(context);
            free(backend);
            return LLM_ALLOCATION_FAILED;
        }
        const llm_status pipeline_status = metal_create_pipelines(context);
        if (pipeline_status != LLM_OK) {
            metal_destroy(context);
            free(backend);
            return pipeline_status;
        }

        backend->device = LLM_DEVICE_METAL;
        backend->ops = metal_backend_ops();
        backend->context = context;
        *out_backend = backend;
        return LLM_OK;
    }
}

const char *llm_backend_metal_device_name(const llm_backend *backend) {
    if (backend == NULL || backend->device != LLM_DEVICE_METAL || backend->context == NULL) {
        return NULL;
    }
    const llm_metal_context *context = backend->context;
    return context->device_name;
}

llm_status llm_backend_metal_begin_batch(llm_backend *backend) {
    if (backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->device != LLM_DEVICE_METAL || backend->context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    llm_metal_context *context = backend->context;
    if (context->batch_active != 0) {
        return LLM_INVALID_ARGUMENT;
    }
    context->batch_active = 1;
    return LLM_OK;
}

llm_status llm_backend_metal_end_batch(llm_backend *backend) {
    if (backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->device != LLM_DEVICE_METAL || backend->context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    llm_metal_context *context = backend->context;
    if (context->batch_active == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    const llm_status status = llm_metal_flush(context);
    context->batch_active = 0;
    return status;
}

llm_status llm_backend_metal_get_metrics(const llm_backend *backend,
                                         llm_metal_backend_metrics *out_metrics) {
    if (backend == NULL || out_metrics == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->device != LLM_DEVICE_METAL || backend->context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    const llm_metal_context *context = backend->context;
    *out_metrics = context->metrics;
    return LLM_OK;
}

llm_status llm_backend_metal_reset_metrics(llm_backend *backend) {
    if (backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->device != LLM_DEVICE_METAL || backend->context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    llm_metal_context *context = backend->context;
    const double compilation_seconds = context->metrics.pipeline_compilation_seconds;
    const size_t active_count = context->metrics.active_buffer_count;
    const size_t cached_count = context->metrics.cached_buffer_count;
    const size_t cached_bytes = context->metrics.cached_buffer_bytes;
    context->metrics = (llm_metal_backend_metrics){0};
    context->metrics.pipeline_compilation_seconds = compilation_seconds;
    context->metrics.active_buffer_count = active_count;
    context->metrics.cached_buffer_count = cached_count;
    context->metrics.cached_buffer_bytes = cached_bytes;
    return LLM_OK;
}
