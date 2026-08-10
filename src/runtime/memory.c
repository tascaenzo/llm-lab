#include <stdlib.h>

#include "runtime_internal.h"

llm_status llm_storage_create(llm_backend *backend, size_t byte_count, llm_storage **out_storage) {
    if (backend == NULL || backend->ops == NULL || backend->ops->allocate == NULL ||
        byte_count == 0U || out_storage == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_storage = NULL;

    llm_storage *storage = calloc(1U, sizeof(*storage));
    if (storage == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    llm_status status = backend->ops->allocate(backend->context, byte_count, &storage->memory);
    if (status != LLM_OK) {
        free(storage);
        return status;
    }
    storage->backend = backend;
    storage->byte_count = byte_count;
    *out_storage = storage;
    return LLM_OK;
}

void llm_storage_destroy(llm_storage *storage) {
    if (storage == NULL) {
        return;
    }
    if (storage->backend != NULL && storage->backend->ops != NULL &&
        storage->backend->ops->deallocate != NULL) {
        storage->backend->ops->deallocate(storage->backend->context, storage->memory);
    }
    free(storage);
}

llm_status llm_tensor_zero(llm_backend *backend, llm_tensor *tensor) {
    size_t payload_bytes = 0U;
    const llm_status status = llm_tensor_validate(backend, tensor, &payload_bytes);
    if (status != LLM_OK) {
        return status;
    }
    return backend->ops->zero(backend->context, tensor->storage->memory, payload_bytes);
}

llm_status llm_tensor_fill_f32(llm_backend *backend, llm_tensor *tensor, float value) {
    size_t payload_bytes = 0U;
    const llm_status status = llm_tensor_validate(backend, tensor, &payload_bytes);
    if (status != LLM_OK) {
        return status;
    }
    if (tensor->dtype != LLM_DTYPE_F32) {
        return LLM_UNSUPPORTED_DTYPE;
    }
    (void)payload_bytes;
    return backend->ops->fill_f32(backend->context, (float *)tensor->storage->memory,
                                  tensor->element_count, value);
}

static int tensors_have_same_shape(const llm_tensor *left, const llm_tensor *right) {
    if (left->rank != right->rank) {
        return 0;
    }
    for (size_t index = 0U; index < left->rank; ++index) {
        if (left->shape[index] != right->shape[index]) {
            return 0;
        }
    }
    return 1;
}

llm_status llm_tensor_copy(llm_backend *backend, const llm_tensor *source,
                           llm_tensor *destination) {
    size_t source_bytes = 0U;
    size_t destination_bytes = 0U;
    llm_status status = llm_tensor_validate(backend, source, &source_bytes);
    if (status != LLM_OK) {
        return status;
    }
    status = llm_tensor_validate(backend, destination, &destination_bytes);
    if (status != LLM_OK) {
        return status;
    }
    if (source->dtype != destination->dtype || tensors_have_same_shape(source, destination) == 0) {
        return LLM_INVALID_SHAPE;
    }
    if (source->storage == destination->storage || source_bytes != destination_bytes) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->copy(backend->context, source->storage->memory,
                              destination->storage->memory, source_bytes);
}

llm_status llm_tensor_write(llm_backend *backend, llm_tensor *destination, const void *source,
                            size_t byte_count) {
    if (source == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    size_t payload_bytes = 0U;
    const llm_status status = llm_tensor_validate(backend, destination, &payload_bytes);
    if (status != LLM_OK) {
        return status;
    }
    if (byte_count != payload_bytes) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->copy(backend->context, source, destination->storage->memory,
                              payload_bytes);
}

llm_status llm_tensor_read(llm_backend *backend, const llm_tensor *source, void *destination,
                           size_t byte_count) {
    if (destination == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    size_t payload_bytes = 0U;
    const llm_status status = llm_tensor_validate(backend, source, &payload_bytes);
    if (status != LLM_OK) {
        return status;
    }
    if (byte_count != payload_bytes) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->copy(backend->context, source->storage->memory, destination,
                              payload_bytes);
}
