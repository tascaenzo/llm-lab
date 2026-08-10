#include <stdint.h>

#include "runtime_internal.h"

static llm_status calculate_layout(size_t rank, const size_t *shape, size_t *out_element_count,
                                   size_t out_strides[LLM_TENSOR_MAX_RANK]) {
    if (rank > LLM_TENSOR_MAX_RANK || (rank != 0U && shape == NULL) || out_element_count == NULL ||
        out_strides == NULL) {
        return rank > LLM_TENSOR_MAX_RANK ? LLM_INVALID_SHAPE : LLM_INVALID_ARGUMENT;
    }

    size_t element_count = 1U;
    for (size_t remaining = rank; remaining > 0U; --remaining) {
        const size_t index = remaining - 1U;
        if (shape[index] == 0U) {
            return LLM_INVALID_SHAPE;
        }
        out_strides[index] = element_count;
        if (shape[index] > SIZE_MAX / element_count) {
            return LLM_OVERFLOW;
        }
        element_count *= shape[index];
    }
    *out_element_count = element_count;
    return LLM_OK;
}

llm_status llm_tensor_create(llm_backend *backend, llm_dtype dtype, size_t rank,
                             const size_t *shape, llm_tensor *out_tensor) {
    if (backend == NULL || out_tensor == NULL || out_tensor->storage != NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_tensor = (llm_tensor){0};
    if (llm_backend_supports_dtype(backend, dtype) == 0) {
        return LLM_UNSUPPORTED_DTYPE;
    }

    size_t strides[LLM_TENSOR_MAX_RANK] = {0};
    size_t element_count = 0U;
    llm_status status = calculate_layout(rank, shape, &element_count, strides);
    if (status != LLM_OK) {
        return status;
    }

    const size_t element_size = llm_dtype_size(dtype);
    if (element_size == 0U || element_count > SIZE_MAX / element_size) {
        return LLM_OVERFLOW;
    }

    llm_storage *storage = NULL;
    status = llm_storage_create(backend, element_count * element_size, &storage);
    if (status != LLM_OK) {
        return status;
    }

    out_tensor->storage = storage;
    out_tensor->rank = rank;
    out_tensor->element_count = element_count;
    out_tensor->dtype = dtype;
    for (size_t index = 0U; index < rank; ++index) {
        out_tensor->shape[index] = shape[index];
        out_tensor->strides[index] = strides[index];
    }
    return LLM_OK;
}

void llm_tensor_destroy(llm_tensor *tensor) {
    if (tensor == NULL) {
        return;
    }
    llm_storage_destroy(tensor->storage);
    *tensor = (llm_tensor){0};
}

llm_status llm_tensor_move(llm_tensor *source, llm_tensor *destination) {
    if (source == NULL || destination == NULL || source == destination || source->storage == NULL ||
        destination->storage != NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *destination = *source;
    *source = (llm_tensor){0};
    return LLM_OK;
}

int llm_tensor_is_contiguous(const llm_tensor *tensor) {
    if (tensor == NULL || tensor->storage == NULL || tensor->rank > LLM_TENSOR_MAX_RANK) {
        return 0;
    }
    size_t expected_stride = 1U;
    for (size_t remaining = tensor->rank; remaining > 0U; --remaining) {
        const size_t index = remaining - 1U;
        if (tensor->shape[index] == 0U || tensor->strides[index] != expected_stride ||
            tensor->shape[index] > SIZE_MAX / expected_stride) {
            return 0;
        }
        expected_stride *= tensor->shape[index];
    }
    return expected_stride == tensor->element_count;
}

llm_device_type llm_tensor_device(const llm_tensor *tensor) {
    if (tensor == NULL || tensor->storage == NULL || tensor->storage->backend == NULL) {
        return LLM_DEVICE_NONE;
    }
    return tensor->storage->backend->device;
}

llm_status llm_tensor_validate(const llm_backend *backend, const llm_tensor *tensor,
                               size_t *out_payload_bytes) {
    if (backend == NULL || tensor == NULL || tensor->storage == NULL || out_payload_bytes == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (tensor->storage->backend != backend) {
        return LLM_DEVICE_MISMATCH;
    }
    if (llm_backend_supports_dtype(backend, tensor->dtype) == 0) {
        return LLM_UNSUPPORTED_DTYPE;
    }
    if (llm_tensor_is_contiguous(tensor) == 0 || tensor->offset != 0U) {
        return LLM_UNSUPPORTED_LAYOUT;
    }

    const size_t element_size = llm_dtype_size(tensor->dtype);
    if (element_size == 0U || tensor->element_count > SIZE_MAX / element_size) {
        return LLM_OVERFLOW;
    }
    const size_t payload_bytes = tensor->element_count * element_size;
    if (payload_bytes == 0U || payload_bytes != tensor->storage->byte_count) {
        return LLM_INVALID_SHAPE;
    }
    *out_payload_bytes = payload_bytes;
    return LLM_OK;
}
