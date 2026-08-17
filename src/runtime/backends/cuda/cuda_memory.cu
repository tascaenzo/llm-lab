/*
 * Device memory, stream bookkeeping and the host/device copy paths.
 *
 * Storage handles are wrapper addresses, never device pointers: the runtime
 * facade treats storage as a plain host-side pointer value, so llm_cuda_copy
 * decides the transfer direction by looking each side up in the allocation list.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_internal.h"

#define LLM_CUDA_POOL_MAX_BUFFERS 64U
#define LLM_CUDA_POOL_MAX_BYTES (1024U * 1024U * 1024U)

void *llm_cuda_device_pointer(void *memory) {
    return memory == NULL ? NULL : static_cast<llm_cuda_buffer *>(memory)->pointer;
}

const void *llm_cuda_device_const_pointer(const void *memory) {
    return memory == NULL ? NULL : static_cast<const llm_cuda_buffer *>(memory)->pointer;
}

int llm_cuda_context_contains_buffer(llm_cuda_context *context, const void *memory) {
    if (context == NULL || memory == NULL) {
        return 0;
    }
    int found = 0;
    (void)pthread_mutex_lock(&context->buffer_mutex);
    for (const llm_cuda_buffer *buffer = context->buffers; buffer != NULL; buffer = buffer->next) {
        if (buffer == memory) {
            found = 1;
            break;
        }
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);
    return found;
}

static llm_status cuda_report(cudaError_t error, const char *stage) {
    if (error == cudaSuccess) {
        return LLM_OK;
    }
    fprintf(stderr, "cuda: %s failed: %s\n", stage, cudaGetErrorString(error));
    return error == cudaErrorMemoryAllocation ? LLM_ALLOCATION_FAILED : LLM_BACKEND_ERROR;
}

/* Reads and clears the sticky flags. The caller must already have synchronized. */
static llm_status cuda_consume_flags(llm_cuda_context *context) {
    const cudaError_t copy_error =
        cudaMemcpy(context->host_flags, context->device_flags,
                   LLM_CUDA_FLAG_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    if (copy_error != cudaSuccess) {
        return cuda_report(copy_error, "flag readback");
    }
    const int non_finite = context->host_flags[LLM_CUDA_FLAG_NON_FINITE];
    const int invalid_index = context->host_flags[LLM_CUDA_FLAG_INVALID_INDEX];
    if (non_finite == 0 && invalid_index == 0) {
        return LLM_OK;
    }
    const cudaError_t clear_error =
        cudaMemset(context->device_flags, 0, LLM_CUDA_FLAG_COUNT * sizeof(int));
    if (clear_error != cudaSuccess) {
        return cuda_report(clear_error, "flag reset");
    }
    return invalid_index != 0 ? LLM_INVALID_INDEX : LLM_NUMERICAL_ERROR;
}

static llm_status cuda_synchronize_stream(llm_cuda_context *context) {
    if (context->events_enabled != 0) {
        (void)cudaEventRecord(context->stop_event, context->stream);
    }
    const cudaError_t error = cudaStreamSynchronize(context->stream);
    ++context->metrics.synchronizations;
    if (error != cudaSuccess) {
        return cuda_report(error, "stream synchronize");
    }
    if (context->events_enabled != 0) {
        float milliseconds = 0.0F;
        if (cudaEventElapsedTime(&milliseconds, context->start_event, context->stop_event) ==
            cudaSuccess) {
            context->metrics.last_gpu_seconds = (double)milliseconds / 1000.0;
            context->metrics.total_gpu_seconds += context->metrics.last_gpu_seconds;
        }
        (void)cudaEventRecord(context->start_event, context->stream);
    }
    return LLM_OK;
}

llm_status llm_cuda_finish(llm_cuda_context *context) {
    if (context == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return cuda_report(launch_error, "kernel launch");
    }
    if (context->batch_active != 0) {
        return LLM_OK;
    }
    const llm_status status = cuda_synchronize_stream(context);
    return status != LLM_OK ? status : cuda_consume_flags(context);
}

llm_status llm_cuda_flush(llm_cuda_context *context) {
    if (context == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const llm_status status = cuda_synchronize_stream(context);
    return status != LLM_OK ? status : cuda_consume_flags(context);
}

llm_status llm_cuda_check_finite(llm_cuda_context *context, const void *memory,
                                 size_t value_count) {
    if (context == NULL || memory == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_launch_check_finite(context->stream,
                                 static_cast<const float *>(llm_cuda_device_const_pointer(memory)),
                                 value_count, context->device_flags);
    ++context->metrics.kernel_launches;
    return LLM_OK;
}

llm_status llm_cuda_check_indices(llm_cuda_context *context, const void *indices, size_t count,
                                  size_t bound) {
    if (context == NULL || indices == NULL || count == 0U || bound == 0U || bound > UINT32_MAX) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_launch_check_indices(
        context->stream, static_cast<const uint32_t *>(llm_cuda_device_const_pointer(indices)),
        count, static_cast<uint32_t>(bound), context->device_flags);
    ++context->metrics.kernel_launches;
    return LLM_OK;
}

llm_status llm_cuda_allocate(void *opaque_context, size_t byte_count, void **out_memory) {
    if (opaque_context == NULL || byte_count == 0U || out_memory == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_memory = NULL;
    llm_cuda_context *context = static_cast<llm_cuda_context *>(opaque_context);

    llm_cuda_buffer *buffer = NULL;
    (void)pthread_mutex_lock(&context->buffer_mutex);
    llm_cuda_buffer **best_link = NULL;
    for (llm_cuda_buffer **link = &context->cached_buffers; *link != NULL; link = &(*link)->next) {
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

    if (buffer == NULL) {
        void *pointer = NULL;
        const cudaError_t error = cudaMalloc(&pointer, byte_count);
        if (error != cudaSuccess) {
            return cuda_report(error, "device allocation");
        }
        buffer = static_cast<llm_cuda_buffer *>(calloc(1U, sizeof(*buffer)));
        if (buffer == NULL) {
            (void)cudaFree(pointer);
            return LLM_ALLOCATION_FAILED;
        }
        buffer->pointer = pointer;
        buffer->capacity = byte_count;
    }
    buffer->byte_count = byte_count;

    (void)pthread_mutex_lock(&context->buffer_mutex);
    buffer->next = context->buffers;
    context->buffers = buffer;
    ++context->metrics.active_buffer_count;
    context->metrics.active_buffer_bytes += buffer->capacity;
    if (context->metrics.active_buffer_bytes > context->metrics.peak_active_buffer_bytes) {
        context->metrics.peak_active_buffer_bytes = context->metrics.active_buffer_bytes;
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);
    *out_memory = buffer;
    return LLM_OK;
}

void llm_cuda_deallocate(void *opaque_context, void *memory) {
    if (opaque_context == NULL || memory == NULL) {
        return;
    }
    llm_cuda_context *context = static_cast<llm_cuda_context *>(opaque_context);
    llm_cuda_buffer *target = static_cast<llm_cuda_buffer *>(memory);

    (void)pthread_mutex_lock(&context->buffer_mutex);
    llm_cuda_buffer **link = &context->buffers;
    while (*link != NULL && *link != target) {
        link = &(*link)->next;
    }
    if (*link == target) {
        *link = target->next;
        --context->metrics.active_buffer_count;
        context->metrics.active_buffer_bytes -= target->capacity;
    } else {
        target = NULL;
    }
    if (target != NULL && context->metrics.cached_buffer_count < LLM_CUDA_POOL_MAX_BUFFERS &&
        context->metrics.cached_buffer_bytes <= LLM_CUDA_POOL_MAX_BYTES &&
        target->capacity <= LLM_CUDA_POOL_MAX_BYTES - context->metrics.cached_buffer_bytes) {
        target->byte_count = target->capacity;
        target->next = context->cached_buffers;
        context->cached_buffers = target;
        ++context->metrics.cached_buffer_count;
        context->metrics.cached_buffer_bytes += target->capacity;
        target = NULL;
    }
    (void)pthread_mutex_unlock(&context->buffer_mutex);

    if (target != NULL) {
        (void)cudaFree(target->pointer);
        free(target);
    }
}

llm_status llm_cuda_zero(void *opaque_context, void *memory, size_t byte_count) {
    if (opaque_context == NULL || memory == NULL || byte_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = static_cast<llm_cuda_context *>(opaque_context);
    llm_cuda_buffer *buffer = static_cast<llm_cuda_buffer *>(memory);
    if (buffer->byte_count != byte_count) {
        return LLM_INVALID_ARGUMENT;
    }
    const cudaError_t error =
        cudaMemsetAsync(buffer->pointer, 0, byte_count, context->stream);
    if (error != cudaSuccess) {
        return cuda_report(error, "device memset");
    }
    return context->batch_active != 0 ? LLM_OK : llm_cuda_finish(context);
}

llm_status llm_cuda_copy(void *opaque_context, const void *source, void *destination,
                         size_t byte_count) {
    if (opaque_context == NULL || source == NULL || destination == NULL || byte_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = static_cast<llm_cuda_context *>(opaque_context);
    const int source_is_buffer = llm_cuda_context_contains_buffer(context, source);
    const int destination_is_buffer = llm_cuda_context_contains_buffer(context, destination);
    if (source_is_buffer == 0 && destination_is_buffer == 0) {
        return LLM_INVALID_ARGUMENT;
    }

    if (source_is_buffer != 0 && destination_is_buffer != 0) {
        const llm_cuda_buffer *source_buffer = static_cast<const llm_cuda_buffer *>(source);
        llm_cuda_buffer *destination_buffer = static_cast<llm_cuda_buffer *>(destination);
        if (source_buffer->byte_count < byte_count || destination_buffer->byte_count < byte_count) {
            return LLM_INVALID_ARGUMENT;
        }
        const cudaError_t error =
            cudaMemcpyAsync(destination_buffer->pointer, source_buffer->pointer, byte_count,
                            cudaMemcpyDeviceToDevice, context->stream);
        if (error != cudaSuccess) {
            return cuda_report(error, "device to device copy");
        }
        return context->batch_active != 0 ? LLM_OK : llm_cuda_finish(context);
    }

    /* A transfer that crosses the PCIe bus is a synchronization point in every
       case, so an open batch is flushed first rather than reordered around it. */
    const llm_status flush_status = llm_cuda_flush(context);
    if (flush_status != LLM_OK) {
        return flush_status;
    }

    if (source_is_buffer != 0) {
        const llm_cuda_buffer *source_buffer = static_cast<const llm_cuda_buffer *>(source);
        if (source_buffer->byte_count < byte_count) {
            return LLM_INVALID_ARGUMENT;
        }
        return cuda_report(cudaMemcpy(destination, source_buffer->pointer, byte_count,
                                      cudaMemcpyDeviceToHost),
                           "device to host copy");
    }

    llm_cuda_buffer *destination_buffer = static_cast<llm_cuda_buffer *>(destination);
    if (destination_buffer->byte_count < byte_count) {
        return LLM_INVALID_ARGUMENT;
    }
    return cuda_report(
        cudaMemcpy(destination_buffer->pointer, source, byte_count, cudaMemcpyHostToDevice),
        "host to device copy");
}
