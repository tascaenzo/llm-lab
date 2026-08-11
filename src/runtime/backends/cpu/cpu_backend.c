#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#include <malloc.h>
#endif

#include "cpu_internal.h"

#define LLM_CPU_ALIGNMENT 64U

static void cpu_destroy(void *context) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL) {
        return;
    }
    llm_cpu_executor_destroy(cpu->executor);
    free(cpu);
}

static int cpu_supports_dtype(const void *context, llm_dtype dtype) {
    if (context == NULL) {
        return 0;
    }
    return dtype == LLM_DTYPE_F32 || dtype == LLM_DTYPE_U32 || dtype == LLM_DTYPE_F16 ||
           dtype == LLM_DTYPE_BF16;
}

static llm_status cpu_allocate(void *context, size_t byte_count, void **out_memory) {
    if (context == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (byte_count == 0U || out_memory == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_memory = NULL;
    if (byte_count > SIZE_MAX - (LLM_CPU_ALIGNMENT - 1U)) {
        return LLM_OVERFLOW;
    }
    const size_t allocation_size =
        (byte_count + (LLM_CPU_ALIGNMENT - 1U)) & ~(LLM_CPU_ALIGNMENT - 1U);
#ifdef _WIN32
    void *memory = _aligned_malloc(allocation_size, LLM_CPU_ALIGNMENT);
#else
    void *memory = aligned_alloc(LLM_CPU_ALIGNMENT, allocation_size);
#endif
    if (memory == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    *out_memory = memory;
    return LLM_OK;
}

static void cpu_deallocate(void *context, void *memory) {
    (void)context;
#ifdef _WIN32
    _aligned_free(memory);
#else
    free(memory);
#endif
}

static llm_status cpu_synchronize(void *context) {
    return context == NULL ? LLM_INVALID_ARGUMENT : LLM_OK;
}

static const llm_backend_ops *cpu_backend_ops(void) {
    static const llm_backend_ops operations = {
        .destroy = cpu_destroy,
        .supports_dtype = cpu_supports_dtype,
        .allocate = cpu_allocate,
        .deallocate = cpu_deallocate,
        .zero = llm_cpu_execute_zero,
        .copy = llm_cpu_execute_copy,
        .cast = llm_cpu_execute_cast,
        .fill_f32 = llm_cpu_execute_fill_f32,
        .add_f32 = llm_cpu_execute_add_f32,
        .multiply_f32 = llm_cpu_execute_multiply_f32,
        .scale_f32 = llm_cpu_execute_scale_f32,
        .reduce_sum_last_f32 = llm_cpu_execute_reduce_sum_last_f32,
        .reduce_max_last_f32 = llm_cpu_execute_reduce_max_last_f32,
        .reduce_mean_square_last_f32 = llm_cpu_execute_reduce_mean_square_last_f32,
        .matmul_f32 = llm_cpu_execute_matmul_f32,
        .matmul_mixed_f32 = llm_cpu_execute_matmul_mixed_f32,
        .gather_rows_f32 = llm_cpu_execute_gather_rows_f32,
        .scatter_add_rows_f32 = llm_cpu_execute_scatter_add_rows_f32,
        .softmax_last_f32 = llm_cpu_execute_softmax_last_f32,
        .cross_entropy_forward_f32 = llm_cpu_execute_cross_entropy_forward_f32,
        .cross_entropy_backward_f32 = llm_cpu_execute_cross_entropy_backward_f32,
        .synchronize = cpu_synchronize,
    };
    return &operations;
}

llm_status llm_backend_cpu_create(llm_backend **out_backend) {
    const llm_cpu_backend_config config = {
        .thread_count = 0U,
        .deterministic = 1,
    };
    return llm_backend_cpu_create_with_config(&config, out_backend);
}

llm_status llm_backend_cpu_create_with_config(const llm_cpu_backend_config *config,
                                              llm_backend **out_backend) {
    if (out_backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_backend = NULL;
    if (config == NULL || (config->deterministic != 0 && config->deterministic != 1) ||
        config->thread_count > LLM_CPU_MAX_THREADS) {
        return LLM_INVALID_ARGUMENT;
    }

    const size_t thread_count =
        config->thread_count == 0U ? llm_cpu_detect_thread_count() : config->thread_count;

    llm_backend *backend = calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    llm_cpu_context *context = calloc(1U, sizeof(*context));
    if (context == NULL) {
        free(backend);
        return LLM_ALLOCATION_FAILED;
    }
    context->thread_count = thread_count;
    context->deterministic = config->deterministic;
    const llm_status executor_status = llm_cpu_executor_create(thread_count, &context->executor);
    if (executor_status != LLM_OK) {
        free(context);
        free(backend);
        return executor_status;
    }
    backend->device = LLM_DEVICE_CPU;
    backend->ops = cpu_backend_ops();
    backend->context = context;
    *out_backend = backend;
    return LLM_OK;
}

size_t llm_backend_cpu_thread_count(const llm_backend *backend) {
    if (backend == NULL || backend->device != LLM_DEVICE_CPU || backend->context == NULL) {
        return 0U;
    }
    const llm_cpu_context *context = backend->context;
    return context->thread_count;
}
