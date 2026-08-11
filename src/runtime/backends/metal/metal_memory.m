#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metal_internal.h"

#define LLM_METAL_POOL_MAX_BUFFERS 32U
#define LLM_METAL_POOL_MAX_BYTES (256U * 1024U * 1024U)

llm_metal_buffer *llm_metal_buffer_from_memory(void *memory) { return (llm_metal_buffer *)memory; }

const llm_metal_buffer *llm_metal_buffer_from_const_memory(const void *memory) {
    return (const llm_metal_buffer *)memory;
}

int llm_metal_context_contains_buffer(llm_metal_context *context, const void *memory) {
    if (context == NULL || memory == NULL) {
        return 0;
    }
    int found = 0;
    (void)pthread_mutex_lock(&context->buffer_mutex);
    for (const llm_metal_buffer *buffer = context->buffers; buffer != NULL; buffer = buffer->next) {
        if (buffer == memory) {
            found = 1;
            break;
        }
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);
    return found;
}

static llm_status metal_wait_and_record(llm_metal_context *context,
                                        id<MTLCommandBuffer> command_buffer) {
    if (command_buffer == nil) {
        return LLM_BACKEND_ERROR;
    }
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
        NSError *error = [command_buffer error];
        fprintf(stderr, "metal: command failed: %s\n",
                error == nil ? "unknown error" : [[error localizedDescription] UTF8String]);
        return LLM_BACKEND_ERROR;
    }
    const CFTimeInterval start = [command_buffer GPUStartTime];
    const CFTimeInterval end = [command_buffer GPUEndTime];
    if (end >= start) {
        context->metrics.last_gpu_seconds = end - start;
        context->metrics.total_gpu_seconds += end - start;
    }
    return LLM_OK;
}

id<MTLCommandBuffer> llm_metal_acquire_command_buffer(llm_metal_context *context) {
    if (context == NULL) {
        return nil;
    }
    if (context->batch_active == 0) {
        return [context->queue commandBuffer];
    }
    if (context->batch_command_buffer == nil) {
        context->batch_command_buffer = [[context->queue commandBuffer] retain];
    }
    return context->batch_command_buffer;
}

id<MTLComputeCommandEncoder>
llm_metal_acquire_compute_encoder(llm_metal_context *context, id<MTLCommandBuffer> command_buffer) {
    if (context == NULL || command_buffer == nil) {
        return nil;
    }
    if (context->batch_active == 0) {
        return [command_buffer computeCommandEncoder];
    }
    if (context->batch_compute_encoder == nil) {
        context->batch_compute_encoder = [[command_buffer computeCommandEncoder] retain];
    }
    return context->batch_compute_encoder;
}

void llm_metal_close_batch_compute_encoder(llm_metal_context *context) {
    if (context != NULL && context->batch_compute_encoder != nil) {
        [context->batch_compute_encoder endEncoding];
        [context->batch_compute_encoder release];
        context->batch_compute_encoder = nil;
    }
}

static llm_status metal_begin_blit(llm_metal_context *context,
                                   id<MTLCommandBuffer> *out_command_buffer,
                                   id<MTLBlitCommandEncoder> *out_encoder) {
    if (context == NULL || out_command_buffer == NULL || out_encoder == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_metal_close_batch_compute_encoder(context);
    id<MTLCommandBuffer> command_buffer = llm_metal_acquire_command_buffer(context);
    id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        return LLM_BACKEND_ERROR;
    }
    *out_command_buffer = command_buffer;
    *out_encoder = encoder;
    return LLM_OK;
}

llm_status llm_metal_submit(llm_metal_context *context, id<MTLCommandBuffer> command_buffer) {
    if (context == NULL || command_buffer == nil) {
        return LLM_INVALID_ARGUMENT;
    }
    if (context->batch_active != 0) {
        return command_buffer == context->batch_command_buffer ? LLM_OK : LLM_BACKEND_ERROR;
    }
    [command_buffer commit];
    ++context->metrics.submitted_command_buffers;
    return metal_wait_and_record(context, command_buffer);
}

llm_status llm_metal_flush(llm_metal_context *context) {
    if (context == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (context->batch_command_buffer == nil) {
        return LLM_OK;
    }
    llm_metal_close_batch_compute_encoder(context);
    id<MTLCommandBuffer> command_buffer = context->batch_command_buffer;
    context->batch_command_buffer = nil;
    [command_buffer commit];
    ++context->metrics.submitted_command_buffers;
    const llm_status status = metal_wait_and_record(context, command_buffer);
    [command_buffer release];
    return status;
}

llm_status llm_metal_allocate(void *opaque_context, size_t byte_count, void **out_memory) {
    if (opaque_context == NULL || byte_count == 0U || out_memory == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_memory = NULL;
    llm_metal_context *context = opaque_context;

    @autoreleasepool {
        llm_metal_buffer *buffer = NULL;
        (void)pthread_mutex_lock(&context->buffer_mutex);
        llm_metal_buffer **best_link = NULL;
        for (llm_metal_buffer **link = &context->cached_buffers; *link != NULL;
             link = &(*link)->next) {
            if ((*link)->capacity >= byte_count &&
                (best_link == NULL || (*link)->capacity < (*best_link)->capacity)) {
                best_link = link;
            }
        }
        if (best_link != NULL) {
            buffer = *best_link;
            *best_link = buffer->next;
            --context->metrics.cached_buffer_count;
            context->metrics.cached_buffer_bytes -= buffer->capacity;
            ++context->metrics.reused_buffer_allocations;
        }
        (void)pthread_mutex_unlock(&context->buffer_mutex);

        if (buffer != NULL) {
            buffer->byte_count = byte_count;
            (void)pthread_mutex_lock(&context->buffer_mutex);
            buffer->next = context->buffers;
            context->buffers = buffer;
            ++context->metrics.active_buffer_count;
            (void)pthread_mutex_unlock(&context->buffer_mutex);
            *out_memory = buffer;
            return LLM_OK;
        }

        id<MTLBuffer> handle = [context->device newBufferWithLength:byte_count
                                                            options:MTLResourceStorageModeShared];
        if (handle == nil) {
            return LLM_ALLOCATION_FAILED;
        }
        buffer = calloc(1U, sizeof(*buffer));
        if (buffer == NULL) {
            [handle release];
            return LLM_ALLOCATION_FAILED;
        }
        buffer->handle = handle;
        buffer->byte_count = byte_count;
        buffer->capacity = byte_count;

        (void)pthread_mutex_lock(&context->buffer_mutex);
        buffer->next = context->buffers;
        context->buffers = buffer;
        ++context->metrics.active_buffer_count;
        (void)pthread_mutex_unlock(&context->buffer_mutex);
        *out_memory = buffer;
        return LLM_OK;
    }
}

void llm_metal_deallocate(void *opaque_context, void *memory) {
    if (opaque_context == NULL || memory == NULL) {
        return;
    }
    llm_metal_context *context = opaque_context;
    llm_metal_buffer *target = memory;

    (void)pthread_mutex_lock(&context->buffer_mutex);
    llm_metal_buffer **link = &context->buffers;
    while (*link != NULL && *link != target) {
        link = &(*link)->next;
    }
    if (*link == target) {
        *link = target->next;
        --context->metrics.active_buffer_count;
    } else {
        target = NULL;
    }
    if (target != NULL && context->metrics.cached_buffer_count < LLM_METAL_POOL_MAX_BUFFERS &&
        context->metrics.cached_buffer_bytes <= LLM_METAL_POOL_MAX_BYTES &&
        target->capacity <= LLM_METAL_POOL_MAX_BYTES - context->metrics.cached_buffer_bytes) {
        target->byte_count = target->capacity;
        target->next = context->cached_buffers;
        context->cached_buffers = target;
        ++context->metrics.cached_buffer_count;
        context->metrics.cached_buffer_bytes += target->capacity;
        target = NULL;
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);

    if (target != NULL) {
        [target->handle release];
        free(target);
    }
}

llm_status llm_metal_zero(void *opaque_context, void *memory, size_t byte_count) {
    if (opaque_context == NULL || memory == NULL || byte_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_metal_context *context = opaque_context;
    llm_metal_buffer *buffer = llm_metal_buffer_from_memory(memory);
    if (buffer->byte_count != byte_count) {
        return LLM_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> encoder = nil;
        const llm_status status = metal_begin_blit(context, &command_buffer, &encoder);
        if (status != LLM_OK) {
            return status;
        }
        [encoder fillBuffer:buffer->handle range:NSMakeRange(0U, byte_count) value:0U];
        [encoder endEncoding];
        return llm_metal_submit(context, command_buffer);
    }
}

llm_status llm_metal_copy(void *opaque_context, const void *source, void *destination,
                          size_t byte_count) {
    if (opaque_context == NULL || source == NULL || destination == NULL || byte_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_metal_context *context = opaque_context;
    const int source_is_buffer = llm_metal_context_contains_buffer(context, source);
    const int destination_is_buffer = llm_metal_context_contains_buffer(context, destination);
    if (source_is_buffer == 0 && destination_is_buffer == 0) {
        return LLM_INVALID_ARGUMENT;
    }

    if (source_is_buffer != 0 && destination_is_buffer != 0) {
        const llm_metal_buffer *source_buffer = llm_metal_buffer_from_const_memory(source);
        llm_metal_buffer *destination_buffer = llm_metal_buffer_from_memory(destination);
        if (source_buffer->byte_count < byte_count || destination_buffer->byte_count < byte_count) {
            return LLM_INVALID_ARGUMENT;
        }
        @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = nil;
            id<MTLBlitCommandEncoder> encoder = nil;
            const llm_status status = metal_begin_blit(context, &command_buffer, &encoder);
            if (status != LLM_OK) {
                return status;
            }
            [encoder copyFromBuffer:source_buffer->handle
                       sourceOffset:0U
                           toBuffer:destination_buffer->handle
                  destinationOffset:0U
                               size:byte_count];
            [encoder endEncoding];
            return llm_metal_submit(context, command_buffer);
        }
    }

    if (source_is_buffer != 0) {
        const llm_metal_buffer *source_buffer = llm_metal_buffer_from_const_memory(source);
        if (source_buffer->byte_count < byte_count) {
            return LLM_INVALID_ARGUMENT;
        }
        const llm_status status = llm_metal_flush(context);
        if (status != LLM_OK) {
            return status;
        }
        (void)memcpy(destination, [source_buffer->handle contents], byte_count);
        return LLM_OK;
    }

    llm_metal_buffer *destination_buffer = llm_metal_buffer_from_memory(destination);
    if (destination_buffer->byte_count < byte_count) {
        return LLM_INVALID_ARGUMENT;
    }
    const llm_status status = llm_metal_flush(context);
    if (status != LLM_OK) {
        return status;
    }
    (void)memcpy([destination_buffer->handle contents], source, byte_count);
    return LLM_OK;
}
