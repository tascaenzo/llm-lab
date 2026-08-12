#include <stdint.h>
#include <stdlib.h>

#include "backend_internal.h"

void llm_backend_destroy(llm_backend *backend) {
    if (backend == NULL) {
        return;
    }
    if (backend->ops != NULL && backend->ops->destroy != NULL) {
        backend->ops->destroy(backend->context);
    }
    free(backend);
}

llm_device_type llm_backend_device(const llm_backend *backend) {
    return backend == NULL ? LLM_DEVICE_NONE : backend->device;
}

llm_status llm_backend_synchronize(llm_backend *backend) {
    if (backend == NULL || backend->ops == NULL || backend->ops->synchronize == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->synchronize(backend->context);
}

size_t llm_dtype_size(llm_dtype dtype) {
    switch (dtype) {
    case LLM_DTYPE_F32:
        return sizeof(float);
    case LLM_DTYPE_U32:
        return sizeof(uint32_t);
    case LLM_DTYPE_BF16:
    case LLM_DTYPE_F16:
        return 2U;
    case LLM_DTYPE_F8_E4M3:
        return 1U;
    }
    return 0U;
}

int llm_backend_supports_dtype(const llm_backend *backend, llm_dtype dtype) {
    if (backend == NULL || backend->ops == NULL || backend->ops->supports_dtype == NULL) {
        return 0;
    }
    return backend->ops->supports_dtype(backend->context, dtype);
}

const char *llm_status_string(llm_status status) {
    switch (status) {
    case LLM_OK:
        return "ok";
    case LLM_INVALID_ARGUMENT:
        return "invalid argument";
    case LLM_INVALID_SHAPE:
        return "invalid tensor shape";
    case LLM_UNSUPPORTED_DTYPE:
        return "unsupported data type";
    case LLM_UNSUPPORTED_DEVICE:
        return "unsupported device";
    case LLM_UNSUPPORTED_OPERATION:
        return "unsupported operation";
    case LLM_UNSUPPORTED_LAYOUT:
        return "unsupported tensor layout";
    case LLM_DEVICE_MISMATCH:
        return "tensor and backend do not match";
    case LLM_INVALID_INDEX:
        return "tensor index outside valid range";
    case LLM_ALLOCATION_FAILED:
        return "allocation failed";
    case LLM_OVERFLOW:
        return "size overflow";
    case LLM_NUMERICAL_ERROR:
        return "numerical error";
    case LLM_BACKEND_ERROR:
        return "backend error";
    }
    return "unknown runtime error";
}
