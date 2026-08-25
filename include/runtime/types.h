#ifndef LLM_LAB_RUNTIME_TYPES_H
#define LLM_LAB_RUNTIME_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/** Status returned by every fallible runtime operation. */
typedef enum llm_status {
    LLM_OK = 0,
    LLM_INVALID_ARGUMENT,
    LLM_INVALID_SHAPE,
    LLM_UNSUPPORTED_DTYPE,
    LLM_UNSUPPORTED_DEVICE,
    LLM_UNSUPPORTED_OPERATION,
    LLM_UNSUPPORTED_LAYOUT,
    LLM_DEVICE_MISMATCH,
    LLM_INVALID_INDEX,
    LLM_ALLOCATION_FAILED,
    LLM_OVERFLOW,
    LLM_NUMERICAL_ERROR,
    LLM_BACKEND_ERROR
} llm_status;

/** Scalar formats named by the ABI. Runtime v1 backends support only F32 and U32. */
typedef enum llm_dtype {
    LLM_DTYPE_F32 = 0,
    LLM_DTYPE_U32,
    /* Reserved for a future contract revision; current backends must reject them. */
    LLM_DTYPE_BF16,
    LLM_DTYPE_F16,
    LLM_DTYPE_F8_E4M3
} llm_dtype;

/** Hardware family that owns a backend or tensor storage. */
typedef enum llm_device_type {
    LLM_DEVICE_NONE = 0,
    LLM_DEVICE_CPU,
    LLM_DEVICE_METAL,
    LLM_DEVICE_CUDA
} llm_device_type;

typedef struct llm_backend llm_backend;
typedef struct llm_storage llm_storage;
typedef struct llm_tensor llm_tensor;

/** Returns a stable human-readable description of a runtime status. */
const char *llm_status_string(llm_status status);

#ifdef __cplusplus
}
#endif

#endif
