#include "runtime/runtime.h"

int llm_backend_metal_is_available(void) { return 0; }

llm_status llm_backend_metal_create(llm_backend **out_backend) {
    if (out_backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_backend = NULL;
    return LLM_UNSUPPORTED_DEVICE;
}

const char *llm_backend_metal_device_name(const llm_backend *backend) {
    (void)backend;
    return NULL;
}

llm_status llm_backend_metal_begin_batch(llm_backend *backend) {
    return backend == NULL ? LLM_INVALID_ARGUMENT : LLM_UNSUPPORTED_DEVICE;
}

llm_status llm_backend_metal_end_batch(llm_backend *backend) {
    return backend == NULL ? LLM_INVALID_ARGUMENT : LLM_UNSUPPORTED_DEVICE;
}

llm_status llm_backend_metal_get_metrics(const llm_backend *backend,
                                         llm_metal_backend_metrics *out_metrics) {
    if (backend == NULL || out_metrics == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_metrics = (llm_metal_backend_metrics){0};
    return LLM_UNSUPPORTED_DEVICE;
}

llm_status llm_backend_metal_reset_metrics(llm_backend *backend) {
    return backend == NULL ? LLM_INVALID_ARGUMENT : LLM_UNSUPPORTED_DEVICE;
}
