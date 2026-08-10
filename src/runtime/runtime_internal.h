#ifndef LLM_LAB_RUNTIME_INTERNAL_H
#define LLM_LAB_RUNTIME_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "runtime/runtime.h"

typedef struct llm_backend_ops {
    void (*destroy)(void *context);
    int (*supports_dtype)(const void *context, llm_dtype dtype);
    llm_status (*allocate)(void *context, size_t byte_count, void **out_memory);
    void (*deallocate)(void *context, void *memory);
    llm_status (*zero)(void *context, void *memory, size_t byte_count);
    llm_status (*copy)(void *context, const void *source, void *destination, size_t byte_count);
    llm_status (*fill_f32)(void *context, float *values, size_t value_count, float value);
    llm_status (*add_f32)(void *context, const float *left, const float *right, float *output,
                          size_t value_count);
    llm_status (*multiply_f32)(void *context, const float *left, const float *right, float *output,
                               size_t value_count);
    llm_status (*scale_f32)(void *context, const float *input, float scale, float *output,
                            size_t value_count);
    llm_status (*reduce_sum_last_f32)(void *context, const float *input, float *output,
                                      size_t outer_count, size_t reduction_size);
    llm_status (*reduce_max_last_f32)(void *context, const float *input, float *output,
                                      size_t outer_count, size_t reduction_size);
    llm_status (*reduce_mean_square_last_f32)(void *context, const float *input, float *output,
                                              size_t outer_count, size_t reduction_size);
    llm_status (*matmul_f32)(void *context, const float *left, const float *right, float *output,
                             size_t rows, size_t inner_size, size_t columns);
    llm_status (*gather_rows_f32)(void *context, const float *table, size_t row_count,
                                  size_t row_width, const uint32_t *indices, size_t index_count,
                                  float *output);
    llm_status (*scatter_add_rows_f32)(void *context, const float *source, const uint32_t *indices,
                                       size_t index_count, size_t row_width, size_t row_count,
                                       float *table);
    llm_status (*softmax_last_f32)(void *context, const float *input, float *output,
                                   size_t outer_count, size_t row_width);
    llm_status (*cross_entropy_forward_f32)(void *context, const float *logits,
                                            const uint32_t *targets, size_t row_count,
                                            size_t vocabulary_size, float *loss);
    llm_status (*cross_entropy_backward_f32)(void *context, const float *logits,
                                             const uint32_t *targets, size_t row_count,
                                             size_t vocabulary_size, float *gradient);
    llm_status (*synchronize)(void *context);
} llm_backend_ops;

struct llm_backend {
    llm_device_type device;
    const llm_backend_ops *ops;
    void *context;
};

struct llm_storage {
    llm_backend *backend;
    void *memory;
    size_t byte_count;
};

size_t llm_dtype_size(llm_dtype dtype);
int llm_backend_supports_dtype(const llm_backend *backend, llm_dtype dtype);

llm_status llm_storage_create(llm_backend *backend, size_t byte_count, llm_storage **out_storage);
void llm_storage_destroy(llm_storage *storage);

llm_status llm_tensor_validate(const llm_backend *backend, const llm_tensor *tensor,
                               size_t *out_payload_bytes);

#endif
