#ifndef LLM_LAB_RUNTIME_RUNTIME_H
#define LLM_LAB_RUNTIME_RUNTIME_H

#include <stddef.h>

#define LLM_TENSOR_MAX_RANK 4U

typedef enum llm_status {
    LLM_OK = 0,
    LLM_INVALID_ARGUMENT,
    LLM_INVALID_SHAPE,
    LLM_UNSUPPORTED_DTYPE,
    LLM_UNSUPPORTED_DEVICE,
    LLM_UNSUPPORTED_LAYOUT,
    LLM_DEVICE_MISMATCH,
    LLM_INVALID_INDEX,
    LLM_ALLOCATION_FAILED,
    LLM_OVERFLOW,
    LLM_NUMERICAL_ERROR,
    LLM_BACKEND_ERROR
} llm_status;

typedef enum llm_dtype {
    LLM_DTYPE_F32 = 0,
    LLM_DTYPE_U32,
    LLM_DTYPE_BF16,
    LLM_DTYPE_F16,
    LLM_DTYPE_F8_E4M3
} llm_dtype;

typedef enum llm_device_type {
    LLM_DEVICE_NONE = 0,
    LLM_DEVICE_CPU,
    LLM_DEVICE_METAL,
    LLM_DEVICE_CUDA
} llm_device_type;

typedef struct llm_backend llm_backend;
typedef struct llm_storage llm_storage;

/** Configuration for a synchronous CPU backend with a persistent thread pool. */
typedef struct llm_cpu_backend_config {
    /** Total participating threads, including the caller. Zero selects automatically. */
    size_t thread_count;
    /** Requests stable scheduling and reduction order when non-zero. */
    int deterministic;
} llm_cpu_backend_config;

/** Observable counters for profiling a Metal backend without exposing native objects. */
typedef struct llm_metal_backend_metrics {
    size_t active_buffer_count;
    size_t cached_buffer_count;
    size_t cached_buffer_bytes;
    unsigned long long submitted_command_buffers;
    unsigned long long kernel_dispatches;
    unsigned long long reused_buffer_allocations;
    double pipeline_compilation_seconds;
    double total_gpu_seconds;
    double last_gpu_seconds;
} llm_metal_backend_metrics;

/** A row-major tensor descriptor owning its storage. Do not copy it by assignment. */
typedef struct llm_tensor {
    llm_storage *storage;
    size_t offset;
    size_t rank;
    size_t shape[LLM_TENSOR_MAX_RANK];
    size_t strides[LLM_TENSOR_MAX_RANK];
    size_t element_count;
    llm_dtype dtype;
} llm_tensor;

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

/** Destroys a backend. All tensors created by it must have already been destroyed. */
void llm_backend_destroy(llm_backend *backend);

/** Returns the device represented by a backend, or LLM_DEVICE_NONE for NULL. */
llm_device_type llm_backend_device(const llm_backend *backend);

/** Waits for all submitted work. The synchronous CPU backend returns immediately. */
llm_status llm_backend_synchronize(llm_backend *backend);

/** Creates a contiguous row-major tensor in an empty output. Rank zero creates a scalar. */
llm_status llm_tensor_create(llm_backend *backend, llm_dtype dtype, size_t rank,
                             const size_t *shape, llm_tensor *out_tensor);

/** Releases tensor storage and resets the descriptor to an empty value. */
void llm_tensor_destroy(llm_tensor *tensor);

/** Moves ownership from source into an empty destination tensor. */
llm_status llm_tensor_move(llm_tensor *source, llm_tensor *destination);

/** Returns non-zero when the tensor uses a contiguous row-major layout. */
int llm_tensor_is_contiguous(const llm_tensor *tensor);

/** Returns the tensor device, or LLM_DEVICE_NONE for an empty tensor. */
llm_device_type llm_tensor_device(const llm_tensor *tensor);

/** Sets every byte in a tensor payload to zero. */
llm_status llm_tensor_zero(llm_backend *backend, llm_tensor *tensor);

/** Fills an FP32 tensor with one value. */
llm_status llm_tensor_fill_f32(llm_backend *backend, llm_tensor *tensor, float value);

/** Copies between distinct tensors with identical shape and dtype. */
llm_status llm_tensor_copy(llm_backend *backend, const llm_tensor *source, llm_tensor *destination);

/** Copies one complete payload from host memory into a tensor. */
llm_status llm_tensor_write(llm_backend *backend, llm_tensor *destination, const void *source,
                            size_t byte_count);

/** Copies one complete tensor payload into host memory. */
llm_status llm_tensor_read(llm_backend *backend, const llm_tensor *source, void *destination,
                           size_t byte_count);

/** Converts between FP32 and FP16/BF16 tensors with identical shapes. */
llm_status llm_cast(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Adds two distinct FP32 tensors with identical shapes. */
llm_status llm_add(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                   llm_tensor *output);

/** Multiplies two distinct FP32 tensors element by element. */
llm_status llm_multiply(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                        llm_tensor *output);

/** Multiplies every FP32 input value by a scalar. */
llm_status llm_scale(llm_backend *backend, const llm_tensor *input, float scale,
                     llm_tensor *output);

/** Reduces the last FP32 dimension by summation. */
llm_status llm_reduce_sum_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Reduces the last FP32 dimension by selecting its maximum. */
llm_status llm_reduce_max_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Reduces the last FP32 dimension to the mean of its squared values. */
llm_status llm_reduce_mean_square_last(llm_backend *backend, const llm_tensor *input,
                                       llm_tensor *output);

/** Computes a two-dimensional FP32 matrix product: [M,K] x [K,N] -> [M,N]. */
llm_status llm_matmul(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                      llm_tensor *output);

/** Multiplies FP16 or BF16 matrices while accumulating into an FP32 output. */
llm_status llm_matmul_mixed_f32(llm_backend *backend, const llm_tensor *left,
                                const llm_tensor *right, llm_tensor *output);

/** Selects rows from an FP32 [V,C] table using a U32 tensor of row indices. */
llm_status llm_gather_rows(llm_backend *backend, const llm_tensor *table, const llm_tensor *indices,
                           llm_tensor *output);

/** Accumulates FP32 source rows into an FP32 [V,C] table using U32 indices. */
llm_status llm_scatter_add_rows(llm_backend *backend, const llm_tensor *source,
                                const llm_tensor *indices, llm_tensor *table);

/** Computes a numerically stable FP32 softmax over the last dimension. */
llm_status llm_softmax_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Computes mean next-token cross-entropy from FP32 [N,V] logits and U32 [N] targets. */
llm_status llm_cross_entropy_forward(llm_backend *backend, const llm_tensor *logits,
                                     const llm_tensor *targets, llm_tensor *loss);

/** Computes the FP32 [N,V] gradient of mean cross-entropy with respect to logits. */
llm_status llm_cross_entropy_backward(llm_backend *backend, const llm_tensor *logits,
                                      const llm_tensor *targets, llm_tensor *logits_gradient);

/** Returns a stable human-readable description of a runtime status. */
const char *llm_status_string(llm_status status);

#endif
