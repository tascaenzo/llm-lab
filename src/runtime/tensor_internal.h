#ifndef LLM_LAB_TENSOR_INTERNAL_H
#define LLM_LAB_TENSOR_INTERNAL_H

#include <stddef.h>

#include "runtime/tensor.h"
#include "storage_internal.h"

/** Validates a materialized tensor before an operation reaches a backend. */
llm_status llm_tensor_validate(const llm_backend *backend, const llm_tensor *tensor,
                               size_t *out_payload_bytes);

#endif
