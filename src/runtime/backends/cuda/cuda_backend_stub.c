#include "runtime/backend.h"

int llm_backend_cuda_is_available(void) { return 0; }

llm_status llm_backend_cuda_create(llm_backend **out_backend) {
    if (out_backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_backend = NULL;
    return LLM_UNSUPPORTED_DEVICE;
}

const char *llm_backend_cuda_device_name(const llm_backend *backend) {
    (void)backend;
    return NULL;
}

llm_status llm_backend_cuda_begin_batch(llm_backend *backend) {
    return backend == NULL ? LLM_INVALID_ARGUMENT : LLM_UNSUPPORTED_DEVICE;
}

llm_status llm_backend_cuda_end_batch(llm_backend *backend) {
    return backend == NULL ? LLM_INVALID_ARGUMENT : LLM_UNSUPPORTED_DEVICE;
}

llm_status llm_backend_cuda_get_metrics(const llm_backend *backend,
                                        llm_cuda_backend_metrics *out_metrics) {
    if (backend == NULL || out_metrics == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_metrics = (llm_cuda_backend_metrics){0};
    return LLM_UNSUPPORTED_DEVICE;
}

llm_status llm_backend_cuda_reset_metrics(llm_backend *backend) {
    return backend == NULL ? LLM_INVALID_ARGUMENT : LLM_UNSUPPORTED_DEVICE;
}
