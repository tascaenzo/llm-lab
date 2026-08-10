#include "runtime_internal.h"

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

static llm_status validate_f32_tensor(const llm_backend *backend, const llm_tensor *tensor) {
    size_t payload_bytes = 0U;
    const llm_status status = llm_tensor_validate(backend, tensor, &payload_bytes);
    if (status != LLM_OK) {
        return status;
    }
    if (tensor->dtype != LLM_DTYPE_F32) {
        return LLM_UNSUPPORTED_DTYPE;
    }
    (void)payload_bytes;
    return LLM_OK;
}

static llm_status validate_u32_tensor(const llm_backend *backend, const llm_tensor *tensor) {
    size_t payload_bytes = 0U;
    const llm_status status = llm_tensor_validate(backend, tensor, &payload_bytes);
    if (status != LLM_OK) {
        return status;
    }
    if (tensor->dtype != LLM_DTYPE_U32) {
        return LLM_UNSUPPORTED_DTYPE;
    }
    (void)payload_bytes;
    return LLM_OK;
}

static llm_status validate_binary_elementwise(llm_backend *backend, const llm_tensor *left,
                                              const llm_tensor *right, llm_tensor *output) {
    llm_status status = validate_f32_tensor(backend, left);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, right);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(left, right) == 0 || tensors_have_same_shape(left, output) == 0) {
        return LLM_INVALID_SHAPE;
    }
    if (output->storage == left->storage || output->storage == right->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return LLM_OK;
}

llm_status llm_add(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                   llm_tensor *output) {
    const llm_status status = validate_binary_elementwise(backend, left, right, output);
    if (status != LLM_OK) {
        return status;
    }
    return backend->ops->add_f32(backend->context, (const float *)left->storage->memory,
                                 (const float *)right->storage->memory,
                                 (float *)output->storage->memory, output->element_count);
}

llm_status llm_multiply(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                        llm_tensor *output) {
    const llm_status status = validate_binary_elementwise(backend, left, right, output);
    if (status != LLM_OK) {
        return status;
    }
    return backend->ops->multiply_f32(backend->context, (const float *)left->storage->memory,
                                      (const float *)right->storage->memory,
                                      (float *)output->storage->memory, output->element_count);
}

llm_status llm_scale(llm_backend *backend, const llm_tensor *input, float scale,
                     llm_tensor *output) {
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(input, output) == 0) {
        return LLM_INVALID_SHAPE;
    }
    if (input->storage == output->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->scale_f32(backend->context, (const float *)input->storage->memory, scale,
                                   (float *)output->storage->memory, output->element_count);
}

static llm_status validate_reduction(llm_backend *backend, const llm_tensor *input,
                                     llm_tensor *output, size_t *out_outer_count,
                                     size_t *out_reduction_size) {
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (input->rank == 0U || output->rank + 1U != input->rank) {
        return LLM_INVALID_SHAPE;
    }
    for (size_t index = 0U; index < output->rank; ++index) {
        if (output->shape[index] != input->shape[index]) {
            return LLM_INVALID_SHAPE;
        }
    }
    if (input->storage == output->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t reduction_size = input->shape[input->rank - 1U];
    if (output->element_count != input->element_count / reduction_size) {
        return LLM_INVALID_SHAPE;
    }
    *out_outer_count = output->element_count;
    *out_reduction_size = reduction_size;
    return LLM_OK;
}

llm_status llm_reduce_sum_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output) {
    size_t outer_count = 0U;
    size_t reduction_size = 0U;
    const llm_status status =
        validate_reduction(backend, input, output, &outer_count, &reduction_size);
    if (status != LLM_OK) {
        return status;
    }
    return backend->ops->reduce_sum_last_f32(
        backend->context, (const float *)input->storage->memory, (float *)output->storage->memory,
        outer_count, reduction_size);
}

llm_status llm_reduce_max_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output) {
    size_t outer_count = 0U;
    size_t reduction_size = 0U;
    const llm_status status =
        validate_reduction(backend, input, output, &outer_count, &reduction_size);
    if (status != LLM_OK) {
        return status;
    }
    return backend->ops->reduce_max_last_f32(
        backend->context, (const float *)input->storage->memory, (float *)output->storage->memory,
        outer_count, reduction_size);
}

llm_status llm_reduce_mean_square_last(llm_backend *backend, const llm_tensor *input,
                                       llm_tensor *output) {
    size_t outer_count = 0U;
    size_t reduction_size = 0U;
    const llm_status status =
        validate_reduction(backend, input, output, &outer_count, &reduction_size);
    if (status != LLM_OK) {
        return status;
    }
    return backend->ops->reduce_mean_square_last_f32(
        backend->context, (const float *)input->storage->memory, (float *)output->storage->memory,
        outer_count, reduction_size);
}

llm_status llm_matmul(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                      llm_tensor *output) {
    llm_status status = validate_f32_tensor(backend, left);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, right);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (left->rank != 2U || right->rank != 2U || output->rank != 2U ||
        left->shape[1] != right->shape[0] || output->shape[0] != left->shape[0] ||
        output->shape[1] != right->shape[1]) {
        return LLM_INVALID_SHAPE;
    }
    if (output->storage == left->storage || output->storage == right->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->matmul_f32(backend->context, (const float *)left->storage->memory,
                                    (const float *)right->storage->memory,
                                    (float *)output->storage->memory, left->shape[0],
                                    left->shape[1], right->shape[1]);
}

static llm_status validate_gather_shapes(const llm_tensor *table, const llm_tensor *indices,
                                         const llm_tensor *output) {
    if (table->rank != 2U || indices->rank >= LLM_TENSOR_MAX_RANK ||
        output->rank != indices->rank + 1U) {
        return LLM_INVALID_SHAPE;
    }
    for (size_t index = 0U; index < indices->rank; ++index) {
        if (output->shape[index] != indices->shape[index]) {
            return LLM_INVALID_SHAPE;
        }
    }
    return output->shape[output->rank - 1U] == table->shape[1] ? LLM_OK : LLM_INVALID_SHAPE;
}

llm_status llm_gather_rows(llm_backend *backend, const llm_tensor *table, const llm_tensor *indices,
                           llm_tensor *output) {
    llm_status status = validate_f32_tensor(backend, table);
    if (status == LLM_OK) {
        status = validate_u32_tensor(backend, indices);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    status = validate_gather_shapes(table, indices, output);
    if (status != LLM_OK) {
        return status;
    }
    if (output->storage == table->storage || output->storage == indices->storage ||
        table->storage == indices->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->gather_rows_f32(backend->context, (const float *)table->storage->memory,
                                         table->shape[0], table->shape[1],
                                         (const uint32_t *)indices->storage->memory,
                                         indices->element_count, (float *)output->storage->memory);
}

llm_status llm_scatter_add_rows(llm_backend *backend, const llm_tensor *source,
                                const llm_tensor *indices, llm_tensor *table) {
    llm_status status = validate_f32_tensor(backend, source);
    if (status == LLM_OK) {
        status = validate_u32_tensor(backend, indices);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, table);
    }
    if (status != LLM_OK) {
        return status;
    }
    status = validate_gather_shapes(table, indices, source);
    if (status != LLM_OK) {
        return status;
    }
    if (table->storage == source->storage || table->storage == indices->storage ||
        source->storage == indices->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->scatter_add_rows_f32(
        backend->context, (const float *)source->storage->memory,
        (const uint32_t *)indices->storage->memory, indices->element_count, table->shape[1],
        table->shape[0], (float *)table->storage->memory);
}

llm_status llm_softmax_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output) {
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (input->rank == 0U || tensors_have_same_shape(input, output) == 0) {
        return LLM_INVALID_SHAPE;
    }
    if (input->storage == output->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t row_width = input->shape[input->rank - 1U];
    return backend->ops->softmax_last_f32(backend->context, (const float *)input->storage->memory,
                                          (float *)output->storage->memory,
                                          input->element_count / row_width, row_width);
}

static llm_status validate_cross_entropy_inputs(llm_backend *backend, const llm_tensor *logits,
                                                const llm_tensor *targets) {
    llm_status status = validate_f32_tensor(backend, logits);
    if (status == LLM_OK) {
        status = validate_u32_tensor(backend, targets);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (logits->rank != 2U || targets->rank != 1U || logits->shape[0] != targets->shape[0]) {
        return LLM_INVALID_SHAPE;
    }
    return logits->storage == targets->storage ? LLM_INVALID_ARGUMENT : LLM_OK;
}

llm_status llm_cross_entropy_forward(llm_backend *backend, const llm_tensor *logits,
                                     const llm_tensor *targets, llm_tensor *loss) {
    llm_status status = validate_cross_entropy_inputs(backend, logits, targets);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, loss);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (loss->rank != 0U || loss->element_count != 1U) {
        return LLM_INVALID_SHAPE;
    }
    if (loss->storage == logits->storage || loss->storage == targets->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->cross_entropy_forward_f32(
        backend->context, (const float *)logits->storage->memory,
        (const uint32_t *)targets->storage->memory, logits->shape[0], logits->shape[1],
        (float *)loss->storage->memory);
}

llm_status llm_cross_entropy_backward(llm_backend *backend, const llm_tensor *logits,
                                      const llm_tensor *targets, llm_tensor *logits_gradient) {
    llm_status status = validate_cross_entropy_inputs(backend, logits, targets);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, logits_gradient);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(logits, logits_gradient) == 0) {
        return LLM_INVALID_SHAPE;
    }
    if (logits_gradient->storage == logits->storage ||
        logits_gradient->storage == targets->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->cross_entropy_backward_f32(
        backend->context, (const float *)logits->storage->memory,
        (const uint32_t *)targets->storage->memory, logits->shape[0], logits->shape[1],
        (float *)logits_gradient->storage->memory);
}
