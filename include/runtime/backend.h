#ifndef LLM_LAB_RUNTIME_BACKEND_H
#define LLM_LAB_RUNTIME_BACKEND_H

#include <stddef.h>

#include "runtime/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Configuration for a synchronous CPU backend with a persistent thread pool. */
typedef struct llm_cpu_backend_config {
    /** Total participating threads, including the caller. Zero selects automatically. */
    size_t thread_count;
} llm_cpu_backend_config;

/** Observable counters for profiling a Metal backend without exposing native objects. */
typedef struct llm_metal_backend_metrics {
    size_t active_buffer_count;
    size_t active_buffer_bytes;
    size_t peak_active_buffer_bytes;
    size_t cached_buffer_count;
    size_t cached_buffer_bytes;
    unsigned long long submitted_command_buffers;
    unsigned long long kernel_dispatches;
    unsigned long long reused_buffer_allocations;
    double pipeline_compilation_seconds;
    double total_gpu_seconds;
    double last_gpu_seconds;
} llm_metal_backend_metrics;

/** Observable counters for profiling a CUDA backend without exposing native objects. */
typedef struct llm_cuda_backend_metrics {
    size_t active_buffer_count;
    size_t active_buffer_bytes;
    size_t peak_active_buffer_bytes;
    size_t cached_buffer_count;
    size_t cached_buffer_bytes;
    unsigned long long kernel_launches;
    unsigned long long synchronizations;
    unsigned long long reused_buffer_allocations;
    double total_gpu_seconds;
    double last_gpu_seconds;
} llm_cuda_backend_metrics;

/** Creates a synchronous CPU backend. */
llm_status llm_backend_cpu_create(llm_backend **out_backend);

/** Creates a CPU backend with explicit execution settings. */
llm_status llm_backend_cpu_create_with_config(const llm_cpu_backend_config *config,
                                              llm_backend **out_backend);

/** Returns the configured CPU thread count, or zero for a non-CPU/NULL backend. */
size_t llm_backend_cpu_thread_count(const llm_backend *backend);

/** Returns non-zero when a usable Metal device is available on this machine. */
int llm_backend_metal_is_available(void);

/** Creates a synchronous Metal backend using the system default GPU. */
llm_status llm_backend_metal_create(llm_backend **out_backend);

/** Returns the Metal device name, or NULL for a non-Metal/NULL backend. */
const char *llm_backend_metal_device_name(const llm_backend *backend);

/** Begins an explicit asynchronous batch. Nested batches are rejected. */
llm_status llm_backend_metal_begin_batch(llm_backend *backend);

/** Waits for every command in the current batch and closes it. */
llm_status llm_backend_metal_end_batch(llm_backend *backend);

/** Reads Metal profiling and buffer-pool counters. */
llm_status llm_backend_metal_get_metrics(const llm_backend *backend,
                                         llm_metal_backend_metrics *out_metrics);

/** Resets cumulative Metal counters without changing live or cached buffers. */
llm_status llm_backend_metal_reset_metrics(llm_backend *backend);

/** Returns non-zero when a usable CUDA device is available on this machine. */
int llm_backend_cuda_is_available(void);

/** Creates a synchronous CUDA backend. LLM_LAB_CUDA_DEVICE selects the device index. */
llm_status llm_backend_cuda_create(llm_backend **out_backend);

/** Returns the CUDA device name, or NULL for a non-CUDA/NULL backend. */
const char *llm_backend_cuda_device_name(const llm_backend *backend);

/** Begins an explicit asynchronous batch. Nested batches are rejected. */
llm_status llm_backend_cuda_begin_batch(llm_backend *backend);

/** Waits for every command in the current batch and closes it. */
llm_status llm_backend_cuda_end_batch(llm_backend *backend);

/** Reads CUDA profiling and buffer-pool counters. */
llm_status llm_backend_cuda_get_metrics(const llm_backend *backend,
                                        llm_cuda_backend_metrics *out_metrics);

/** Resets cumulative CUDA counters without changing live or cached buffers. */
llm_status llm_backend_cuda_reset_metrics(llm_backend *backend);

/**
 * Opens an asynchronous batch on whichever accelerator the backend represents,
 * so callers can group a training step without naming a device. The CPU backend
 * is synchronous and accepts both calls as a no-op.
 */
llm_status llm_backend_begin_batch(llm_backend *backend);

/** Closes the batch opened by llm_backend_begin_batch and waits for its work. */
llm_status llm_backend_end_batch(llm_backend *backend);

/** Destroys a backend. All tensors created by it must have already been destroyed. */
void llm_backend_destroy(llm_backend *backend);

/** Returns the device represented by a backend, or LLM_DEVICE_NONE for NULL. */
llm_device_type llm_backend_device(const llm_backend *backend);

/** Waits for all submitted work. The synchronous CPU backend returns immediately. */
llm_status llm_backend_synchronize(llm_backend *backend);

#ifdef __cplusplus
}
#endif

#endif
