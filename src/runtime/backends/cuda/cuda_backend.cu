/*
 * CUDA backend construction and the public device-specific entry points.
 *
 * The backend fills the same llm_backend_ops table as CPU and Metal, so nothing
 * above src/runtime/backends knows it exists beyond the create call.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuda_internal.h"

namespace {

char *copy_device_name(const cudaDeviceProp &properties) {
    const size_t length = strlen(properties.name);
    char *copy = static_cast<char *>(malloc(length + 1U));
    if (copy != NULL) {
        (void)memcpy(copy, properties.name, length + 1U);
    }
    return copy;
}

int environment_flag_is_set(const char *name) {
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

llm_status configured_math_mode(llm_cuda_math_mode *out_mode) {
    if (out_mode == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const char *requested = getenv("LLM_LAB_CUDA_MATH");
    if (requested == NULL || requested[0] == '\0') {
        *out_mode = environment_flag_is_set("LLM_LAB_CUDA_TF32") != 0 ? LLM_CUDA_MATH_TF32
                                                                       : LLM_CUDA_MATH_F32;
        return LLM_OK;
    }
    if (strcmp(requested, "f32") == 0 || strcmp(requested, "0") == 0) {
        *out_mode = LLM_CUDA_MATH_F32;
    } else if (strcmp(requested, "tf32") == 0 || strcmp(requested, "1") == 0) {
        *out_mode = LLM_CUDA_MATH_TF32;
    } else if (strcmp(requested, "bf16-compute") == 0) {
        *out_mode = LLM_CUDA_MATH_BF16_COMPUTE;
    } else {
        fprintf(stderr,
                "cuda: LLM_LAB_CUDA_MATH must be f32, tf32, or bf16-compute (got %s)\n",
                requested);
        return LLM_INVALID_ARGUMENT;
    }
    return LLM_OK;
}

llm_status configured_numerics_mode(llm_cuda_numerics_mode *out_mode) {
    if (out_mode == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const char *requested = getenv("LLM_LAB_CUDA_NUMERICS");
    if (requested == NULL || requested[0] == '\0' || strcmp(requested, "strict") == 0) {
        *out_mode = LLM_CUDA_NUMERICS_STRICT;
    } else if (strcmp(requested, "step") == 0) {
        *out_mode = LLM_CUDA_NUMERICS_STEP;
    } else {
        fprintf(stderr, "cuda: LLM_LAB_CUDA_NUMERICS must be strict or step (got %s)\n",
                requested);
        return LLM_INVALID_ARGUMENT;
    }
    return LLM_OK;
}

const char *math_mode_name(llm_cuda_math_mode mode) {
    return mode == LLM_CUDA_MATH_TF32          ? "tf32"
           : mode == LLM_CUDA_MATH_BF16_COMPUTE ? "bf16-compute"
                                                 : "f32";
}

const char *numerics_mode_name(llm_cuda_numerics_mode mode) {
    return mode == LLM_CUDA_NUMERICS_STEP ? "step" : "strict";
}

void cuda_destroy(void *opaque_context) {
    llm_cuda_context *context = static_cast<llm_cuda_context *>(opaque_context);
    if (context == NULL) {
        return;
    }
    if (context->stream != NULL) {
        (void)cudaStreamSynchronize(context->stream);
    }
    llm_cuda_buffer *buffer = context->buffers;
    while (buffer != NULL) {
        llm_cuda_buffer *next = buffer->next;
        (void)cudaFree(buffer->pointer);
        free(buffer);
        buffer = next;
    }
    buffer = context->cached_buffers;
    while (buffer != NULL) {
        llm_cuda_buffer *next = buffer->next;
        (void)cudaFree(buffer->pointer);
        free(buffer);
        buffer = next;
    }
    if (context->blas != NULL) {
        (void)cublasDestroy(context->blas);
    }
    if (context->start_event != NULL) {
        (void)cudaEventDestroy(context->start_event);
    }
    if (context->stop_event != NULL) {
        (void)cudaEventDestroy(context->stop_event);
    }
    (void)cudaFree(context->device_flags);
    (void)cudaFreeHost(context->host_flags);
    if (context->stream != NULL) {
        (void)cudaStreamDestroy(context->stream);
    }
    free(context->device_name);
    (void)pthread_mutex_destroy(&context->buffer_mutex);
    free(context);
}

int cuda_supports_dtype(const void *opaque_context, llm_dtype dtype) {
    return opaque_context != NULL && (dtype == LLM_DTYPE_F32 || dtype == LLM_DTYPE_U32);
}

llm_status cuda_synchronize(void *opaque_context) {
    return opaque_context == NULL ? LLM_INVALID_ARGUMENT
                                  : llm_cuda_flush(static_cast<llm_cuda_context *>(opaque_context));
}

const llm_backend_ops *cuda_backend_ops() {
    static const llm_backend_ops operations = {
        /* .destroy = */ cuda_destroy,
        /* .supports_dtype = */ cuda_supports_dtype,
        /* .allocate = */ llm_cuda_allocate,
        /* .deallocate = */ llm_cuda_deallocate,
        /* .zero = */ llm_cuda_zero,
        /* .copy = */ llm_cuda_copy,
        /* .fill_f32 = */ llm_cuda_fill_f32,
        /* .add_f32 = */ llm_cuda_add_f32,
        /* .multiply_f32 = */ llm_cuda_multiply_f32,
        /* .scale_f32 = */ llm_cuda_scale_f32,
        /* .reduce_sum_last_f32 = */ llm_cuda_reduce_sum_last_f32,
        /* .reduce_max_last_f32 = */ llm_cuda_reduce_max_last_f32,
        /* .reduce_mean_square_last_f32 = */ llm_cuda_reduce_mean_square_last_f32,
        /* .accumulate_sum_squares_f32 = */ llm_cuda_accumulate_sum_squares_f32,
        /* .matmul_f32 = */ llm_cuda_matmul_f32,
        /* .matmul_ex_f32 = */ llm_cuda_matmul_ex_f32,
        /* .gather_rows_f32 = */ llm_cuda_gather_rows_f32,
        /* .scatter_add_rows_f32 = */ llm_cuda_scatter_add_rows_f32,
        /* .accumulate_f32 = */ llm_cuda_accumulate_f32,
        /* .silu_f32 = */ llm_cuda_silu_f32,
        /* .silu_backward_f32 = */ llm_cuda_silu_backward_f32,
        /* .rms_norm_f32 = */ llm_cuda_rms_norm_f32,
        /* .rms_norm_backward_f32 = */ llm_cuda_rms_norm_backward_f32,
        /* .rope_f32 = */ llm_cuda_rope_f32,
        /* .rope_backward_f32 = */ llm_cuda_rope_backward_f32,
        /* .attention_forward_f32 = */ llm_cuda_attention_forward_f32,
        /* .attention_backward_f32 = */ llm_cuda_attention_backward_f32,
        /* .softmax_last_f32 = */ llm_cuda_softmax_last_f32,
        /* .cross_entropy_forward_f32 = */ llm_cuda_cross_entropy_forward_f32,
        /* .cross_entropy_backward_f32 = */ llm_cuda_cross_entropy_backward_f32,
        /* .adamw_update_f32 = */ llm_cuda_adamw_update_f32,
        /* .synchronize = */ cuda_synchronize,
    };
    return &operations;
}

llm_cuda_context *as_context(llm_backend *backend) {
    if (backend == NULL || backend->device != LLM_DEVICE_CUDA || backend->context == NULL) {
        return NULL;
    }
    return static_cast<llm_cuda_context *>(backend->context);
}

} // namespace

extern "C" int llm_backend_cuda_is_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" llm_status llm_backend_cuda_create(llm_backend **out_backend) {
    if (out_backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_backend = NULL;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    int device_index = 0;
    const char *requested = getenv("LLM_LAB_CUDA_DEVICE");
    if (requested != NULL) {
        device_index = atoi(requested);
        if (device_index < 0 || device_index >= device_count) {
            return LLM_UNSUPPORTED_DEVICE;
        }
    }
    if (cudaSetDevice(device_index) != cudaSuccess) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, device_index) != cudaSuccess) {
        return LLM_UNSUPPORTED_DEVICE;
    }

    llm_backend *backend = static_cast<llm_backend *>(calloc(1U, sizeof(*backend)));
    llm_cuda_context *context = static_cast<llm_cuda_context *>(calloc(1U, sizeof(*context)));
    if (backend == NULL || context == NULL) {
        free(backend);
        free(context);
        return LLM_ALLOCATION_FAILED;
    }
    if (pthread_mutex_init(&context->buffer_mutex, NULL) != 0) {
        free(context);
        free(backend);
        return LLM_BACKEND_ERROR;
    }
    context->device_index = device_index;
    context->device_name = copy_device_name(properties);
    if (context->device_name == NULL) {
        cuda_destroy(context);
        free(backend);
        return LLM_ALLOCATION_FAILED;
    }
    if (cudaStreamCreate(&context->stream) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&context->device_flags),
                   LLM_CUDA_FLAG_COUNT * sizeof(int)) != cudaSuccess ||
        cudaMemset(context->device_flags, 0, LLM_CUDA_FLAG_COUNT * sizeof(int)) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&context->host_flags),
                      LLM_CUDA_FLAG_COUNT * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
        cuda_destroy(context);
        free(backend);
        return LLM_BACKEND_ERROR;
    }
    if (cublasCreate(&context->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(context->blas, context->stream) != CUBLAS_STATUS_SUCCESS) {
        cuda_destroy(context);
        free(backend);
        return LLM_BACKEND_ERROR;
    }
    llm_status configuration_status = configured_math_mode(&context->math_mode);
    if (configuration_status == LLM_OK) {
        configuration_status = configured_numerics_mode(&context->numerics_mode);
    }
    const cublasMath_t blas_math = context->math_mode == LLM_CUDA_MATH_F32
                                       ? CUBLAS_PEDANTIC_MATH
                                       : context->math_mode == LLM_CUDA_MATH_TF32
                                             ? CUBLAS_TF32_TENSOR_OP_MATH
                                             : CUBLAS_DEFAULT_MATH;
    if (configuration_status != LLM_OK ||
        cublasSetMathMode(context->blas, blas_math) != CUBLAS_STATUS_SUCCESS) {
        cuda_destroy(context);
        free(backend);
        return configuration_status != LLM_OK ? configuration_status : LLM_BACKEND_ERROR;
    }
    cublasMath_t effective_math = CUBLAS_DEFAULT_MATH;
    if (cublasGetMathMode(context->blas, &effective_math) != CUBLAS_STATUS_SUCCESS ||
        effective_math != blas_math) {
        fprintf(stderr, "cuda: cuBLAS did not retain requested math mode %s\n",
                math_mode_name(context->math_mode));
        cuda_destroy(context);
        free(backend);
        return LLM_BACKEND_ERROR;
    }
    fprintf(stderr, "cuda: math=%s, numerics=%s\n", math_mode_name(context->math_mode),
            numerics_mode_name(context->numerics_mode));
    if (cudaEventCreate(&context->start_event) == cudaSuccess &&
        cudaEventCreate(&context->stop_event) == cudaSuccess) {
        context->events_enabled = 1;
    }

    backend->device = LLM_DEVICE_CUDA;
    backend->ops = cuda_backend_ops();
    backend->context = context;
    *out_backend = backend;
    return LLM_OK;
}

extern "C" const char *llm_backend_cuda_device_name(const llm_backend *backend) {
    const llm_cuda_context *context = as_context(const_cast<llm_backend *>(backend));
    return context == NULL ? NULL : context->device_name;
}

extern "C" llm_status llm_backend_cuda_begin_batch(llm_backend *backend) {
    if (backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(backend);
    if (context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    if (context->batch_active != 0) {
        return LLM_INVALID_ARGUMENT;
    }
    context->batch_active = 1;
    llm_cuda_start_timing(context);
    return LLM_OK;
}

extern "C" llm_status llm_backend_cuda_end_batch(llm_backend *backend) {
    if (backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(backend);
    if (context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    if (context->batch_active == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    const llm_status status = llm_cuda_flush(context);
    context->batch_active = 0;
    return status;
}

extern "C" llm_status llm_backend_cuda_get_metrics(const llm_backend *backend,
                                                   llm_cuda_backend_metrics *out_metrics) {
    if (backend == NULL || out_metrics == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const llm_cuda_context *context = as_context(const_cast<llm_backend *>(backend));
    if (context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    *out_metrics = context->metrics;
    return LLM_OK;
}

extern "C" llm_status llm_backend_cuda_reset_metrics(llm_backend *backend) {
    if (backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(backend);
    if (context == NULL) {
        return LLM_UNSUPPORTED_DEVICE;
    }
    const size_t active_count = context->metrics.active_buffer_count;
    const size_t active_bytes = context->metrics.active_buffer_bytes;
    const size_t cached_count = context->metrics.cached_buffer_count;
    const size_t cached_bytes = context->metrics.cached_buffer_bytes;
    context->metrics = llm_cuda_backend_metrics{};
    context->metrics.active_buffer_count = active_count;
    context->metrics.active_buffer_bytes = active_bytes;
    context->metrics.peak_active_buffer_bytes = active_bytes;
    context->metrics.cached_buffer_count = cached_count;
    context->metrics.cached_buffer_bytes = cached_bytes;
    return LLM_OK;
}
