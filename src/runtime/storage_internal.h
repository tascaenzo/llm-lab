#ifndef LLM_LAB_STORAGE_INTERNAL_H
#define LLM_LAB_STORAGE_INTERNAL_H

#include <stddef.h>

#include "backend_internal.h"

/** Backend-owned allocation shared by one or more tensor descriptors. */
struct llm_storage {
    llm_backend *backend;
    void *memory;
    size_t byte_count;
    size_t reference_count;
};

llm_status llm_storage_create(llm_backend *backend, size_t byte_count, llm_storage **out_storage);
llm_status llm_storage_retain(llm_storage *storage);
void llm_storage_release(llm_storage *storage);

#endif
