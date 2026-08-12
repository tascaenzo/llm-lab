#ifndef LLM_LAB_RUNTIME_TENSOR_H
#define LLM_LAB_RUNTIME_TENSOR_H

#include <stddef.h>

#include "runtime/types.h"

#define LLM_TENSOR_MAX_RANK 4U

/** A row-major tensor descriptor owning one storage reference. Do not copy it by assignment. */
struct llm_tensor {
    llm_storage *storage;
    size_t offset;
    size_t rank;
    size_t shape[LLM_TENSOR_MAX_RANK];
    size_t strides[LLM_TENSOR_MAX_RANK];
    size_t element_count;
    llm_dtype dtype;
};

/** Creates a contiguous row-major tensor in an empty output. Rank zero creates a scalar. */
llm_status llm_tensor_create(llm_backend *backend, llm_dtype dtype, size_t rank,
                             const size_t *shape, llm_tensor *out_tensor);

/** Releases tensor storage and resets the descriptor to an empty value. */
void llm_tensor_destroy(llm_tensor *tensor);

/** Moves ownership from source into an empty destination tensor. */
llm_status llm_tensor_move(llm_tensor *source, llm_tensor *destination);

/** Creates a contiguous view with a new shape and shared storage. */
llm_status llm_tensor_reshape(const llm_tensor *input, size_t rank, const size_t *shape,
                              llm_tensor *out_view);

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

#endif
