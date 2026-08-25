#include <math.h>
#include <stdint.h>

#include "runtime/operations.h"

#include "tensor_internal.h"

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

static int tensors_have_distinct_storage(const llm_tensor *const *tensors, size_t tensor_count) {
    for (size_t left = 0U; left < tensor_count; ++left) {
        for (size_t right = left + 1U; right < tensor_count; ++right) {
            if (tensors[left]->storage == tensors[right]->storage) {
                return 0;
            }
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

llm_status llm_accumulate_sum_squares(llm_backend *backend, const llm_tensor *input,
                                      llm_tensor *accumulator) {
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, accumulator);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (input->element_count == 0U || accumulator->rank != 0U ||
        input->storage == accumulator->storage ||
        backend->ops->accumulate_sum_squares_f32 == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    return backend->ops->accumulate_sum_squares_f32(
        backend->context, (const float *)input->storage->memory,
        (float *)accumulator->storage->memory, input->element_count);
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

llm_status llm_matmul_ex(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                         const llm_matmul_options *options, llm_tensor *output) {
    if (options == NULL || (options->transpose_left != 0 && options->transpose_left != 1) ||
        (options->transpose_right != 0 && options->transpose_right != 1)) {
        return LLM_INVALID_ARGUMENT;
    }
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
    if (left->rank != 2U || right->rank != 2U || output->rank != 2U) {
        return LLM_INVALID_SHAPE;
    }
    const size_t output_rows = options->transpose_left != 0 ? left->shape[1] : left->shape[0];
    const size_t inner_size = options->transpose_left != 0 ? left->shape[0] : left->shape[1];
    const size_t right_inner = options->transpose_right != 0 ? right->shape[1] : right->shape[0];
    const size_t output_columns = options->transpose_right != 0 ? right->shape[0] : right->shape[1];
    if (inner_size != right_inner || output->shape[0] != output_rows ||
        output->shape[1] != output_columns) {
        return LLM_INVALID_SHAPE;
    }
    const llm_tensor *tensors[] = {left, right, output};
    if (tensors_have_distinct_storage(tensors, 3U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->matmul_ex_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->matmul_ex_f32(backend->context, (const float *)left->storage->memory,
                                       (const float *)right->storage->memory,
                                       (float *)output->storage->memory, left->shape[0],
                                       left->shape[1], right->shape[0], right->shape[1],
                                       options->transpose_left, options->transpose_right);
}

llm_status llm_accumulate(llm_backend *backend, const llm_tensor *source, llm_tensor *destination) {
    llm_status status = validate_f32_tensor(backend, source);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, destination);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(source, destination) == 0) {
        return LLM_INVALID_SHAPE;
    }
    if (source->storage == destination->storage) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->accumulate_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->accumulate_f32(backend->context, (const float *)source->storage->memory,
                                        (float *)destination->storage->memory,
                                        destination->element_count);
}

llm_status llm_silu(llm_backend *backend, const llm_tensor *input, llm_tensor *output) {
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
    if (backend->ops->silu_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->silu_f32(backend->context, (const float *)input->storage->memory,
                                  (float *)output->storage->memory, input->element_count);
}

llm_status llm_silu_backward(llm_backend *backend, const llm_tensor *input,
                             const llm_tensor *output_gradient, llm_tensor *input_gradient) {
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output_gradient);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, input_gradient);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(input, output_gradient) == 0 ||
        tensors_have_same_shape(input, input_gradient) == 0) {
        return LLM_INVALID_SHAPE;
    }
    const llm_tensor *tensors[] = {input, output_gradient, input_gradient};
    if (tensors_have_distinct_storage(tensors, 3U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->silu_backward_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->silu_backward_f32(backend->context, (const float *)input->storage->memory,
                                           (const float *)output_gradient->storage->memory,
                                           (float *)input_gradient->storage->memory,
                                           input->element_count);
}

static llm_status validate_rms_norm_shapes(const llm_tensor *input, const llm_tensor *weight,
                                           const llm_tensor *output) {
    if (input->rank == 0U || weight->rank != 1U || tensors_have_same_shape(input, output) == 0 ||
        weight->shape[0] != input->shape[input->rank - 1U]) {
        return LLM_INVALID_SHAPE;
    }
    return LLM_OK;
}

llm_status llm_rms_norm(llm_backend *backend, const llm_tensor *input, const llm_tensor *weight,
                        float epsilon, llm_tensor *output) {
    if (!isfinite(epsilon) || epsilon <= 0.0F) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, weight);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status == LLM_OK) {
        status = validate_rms_norm_shapes(input, weight, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    const llm_tensor *tensors[] = {input, weight, output};
    if (tensors_have_distinct_storage(tensors, 3U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->rms_norm_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    const size_t row_width = weight->shape[0];
    return backend->ops->rms_norm_f32(backend->context, (const float *)input->storage->memory,
                                      (const float *)weight->storage->memory, epsilon,
                                      (float *)output->storage->memory,
                                      input->element_count / row_width, row_width);
}

llm_status llm_rms_norm_backward(llm_backend *backend, const llm_tensor *input,
                                 const llm_tensor *weight, const llm_tensor *output_gradient,
                                 float epsilon, llm_tensor *input_gradient,
                                 llm_tensor *weight_gradient) {
    if (!isfinite(epsilon) || epsilon <= 0.0F) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, weight);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output_gradient);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, input_gradient);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, weight_gradient);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (validate_rms_norm_shapes(input, weight, output_gradient) != LLM_OK ||
        tensors_have_same_shape(input, input_gradient) == 0 ||
        tensors_have_same_shape(weight, weight_gradient) == 0) {
        return LLM_INVALID_SHAPE;
    }
    const llm_tensor *tensors[] = {input, weight, output_gradient, input_gradient, weight_gradient};
    if (tensors_have_distinct_storage(tensors, 5U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->rms_norm_backward_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    const size_t row_width = weight->shape[0];
    return backend->ops->rms_norm_backward_f32(
        backend->context, (const float *)input->storage->memory,
        (const float *)weight->storage->memory, (const float *)output_gradient->storage->memory,
        epsilon, (float *)input_gradient->storage->memory,
        (float *)weight_gradient->storage->memory, input->element_count / row_width, row_width);
}

static llm_status validate_rope_tensors(llm_backend *backend, const llm_tensor *input,
                                        const llm_tensor *cos_table, const llm_tensor *sin_table,
                                        llm_tensor *output) {
    llm_status status = validate_f32_tensor(backend, input);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, cos_table);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, sin_table);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (input->rank != 4U || cos_table->rank != 2U ||
        tensors_have_same_shape(cos_table, sin_table) == 0 ||
        tensors_have_same_shape(input, output) == 0 || input->shape[3] % 2U != 0U ||
        cos_table->shape[0] != input->shape[1] || cos_table->shape[1] != input->shape[3] / 2U) {
        return LLM_INVALID_SHAPE;
    }
    const llm_tensor *tensors[] = {input, cos_table, sin_table, output};
    return tensors_have_distinct_storage(tensors, 4U) != 0 ? LLM_OK : LLM_INVALID_ARGUMENT;
}

static llm_status dispatch_rope(llm_backend *backend, const llm_tensor *input,
                                const llm_tensor *cos_table, const llm_tensor *sin_table,
                                llm_tensor *output, int backward) {
    const llm_status status = validate_rope_tensors(backend, input, cos_table, sin_table, output);
    if (status != LLM_OK) {
        return status;
    }
    if (backward == 0) {
        if (backend->ops->rope_f32 == NULL) {
            return LLM_UNSUPPORTED_OPERATION;
        }
        return backend->ops->rope_f32(backend->context, (const float *)input->storage->memory,
                                      (const float *)cos_table->storage->memory,
                                      (const float *)sin_table->storage->memory, input->shape[0],
                                      input->shape[1], input->shape[2], input->shape[3],
                                      (float *)output->storage->memory);
    }
    if (backend->ops->rope_backward_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->rope_backward_f32(backend->context, (const float *)input->storage->memory,
                                           (const float *)cos_table->storage->memory,
                                           (const float *)sin_table->storage->memory,
                                           input->shape[0], input->shape[1], input->shape[2],
                                           input->shape[3], (float *)output->storage->memory);
}

llm_status llm_rope(llm_backend *backend, const llm_tensor *input, const llm_tensor *cos_table,
                    const llm_tensor *sin_table, llm_tensor *output) {
    return dispatch_rope(backend, input, cos_table, sin_table, output, 0);
}

llm_status llm_rope_backward(llm_backend *backend, const llm_tensor *output_gradient,
                             const llm_tensor *cos_table, const llm_tensor *sin_table,
                             llm_tensor *input_gradient) {
    return dispatch_rope(backend, output_gradient, cos_table, sin_table, input_gradient, 1);
}

static llm_status validate_attention_tensors(llm_backend *backend, const llm_tensor *query,
                                             const llm_tensor *key, const llm_tensor *value,
                                             const llm_attention_options *options,
                                             const llm_tensor *output) {
    if (options == NULL || !isfinite(options->scale) || options->scale <= 0.0F) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_status status = validate_f32_tensor(backend, query);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, key);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, value);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, output);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (query->rank != 4U || key->rank != 4U || value->rank != 4U ||
        tensors_have_same_shape(key, value) == 0 || tensors_have_same_shape(query, output) == 0 ||
        query->shape[0] != key->shape[0] || query->shape[1] != key->shape[1] ||
        query->shape[3] != key->shape[3] || query->shape[2] % key->shape[2] != 0U) {
        return LLM_INVALID_SHAPE;
    }
    return LLM_OK;
}

llm_status llm_attention_forward(llm_backend *backend, const llm_tensor *query,
                                 const llm_tensor *key, const llm_tensor *value,
                                 const llm_attention_options *options, llm_tensor *output) {
    const llm_status status =
        validate_attention_tensors(backend, query, key, value, options, output);
    if (status != LLM_OK) {
        return status;
    }
    const llm_tensor *tensors[] = {query, key, value, output};
    if (tensors_have_distinct_storage(tensors, 4U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->attention_forward_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->attention_forward_f32(
        backend->context, (const float *)query->storage->memory,
        (const float *)key->storage->memory, (const float *)value->storage->memory, options->scale,
        query->shape[0], query->shape[1], query->shape[2], key->shape[2], query->shape[3],
        (float *)output->storage->memory);
}

llm_status llm_attention_backward(llm_backend *backend, const llm_tensor *query,
                                  const llm_tensor *key, const llm_tensor *value,
                                  const llm_tensor *output_gradient,
                                  const llm_attention_options *options, llm_tensor *query_gradient,
                                  llm_tensor *key_gradient, llm_tensor *value_gradient) {
    llm_status status =
        validate_attention_tensors(backend, query, key, value, options, output_gradient);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, query_gradient);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, key_gradient);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, value_gradient);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(query, query_gradient) == 0 ||
        tensors_have_same_shape(key, key_gradient) == 0 ||
        tensors_have_same_shape(value, value_gradient) == 0) {
        return LLM_INVALID_SHAPE;
    }
    const llm_tensor *tensors[] = {query,          key,          value,         output_gradient,
                                   query_gradient, key_gradient, value_gradient};
    if (tensors_have_distinct_storage(tensors, 7U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->attention_backward_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->attention_backward_f32(
        backend->context, (const float *)query->storage->memory,
        (const float *)key->storage->memory, (const float *)value->storage->memory,
        (const float *)output_gradient->storage->memory, options->scale, query->shape[0],
        query->shape[1], query->shape[2], key->shape[2], query->shape[3],
        (float *)query_gradient->storage->memory, (float *)key_gradient->storage->memory,
        (float *)value_gradient->storage->memory);
}

llm_status llm_adamw_update(llm_backend *backend, llm_tensor *parameter, llm_tensor *gradient,
                            llm_tensor *first_moment, llm_tensor *second_moment,
                            const llm_adamw_options *options) {
    if (options == NULL || !isfinite(options->learning_rate) || options->learning_rate < 0.0F ||
        !isfinite(options->beta1) || options->beta1 < 0.0F || options->beta1 >= 1.0F ||
        !isfinite(options->beta2) || options->beta2 < 0.0F || options->beta2 >= 1.0F ||
        !isfinite(options->epsilon) || options->epsilon <= 0.0F ||
        !isfinite(options->weight_decay) || options->weight_decay < 0.0F ||
        !isfinite(options->gradient_scale) || options->step == 0ULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_status status = validate_f32_tensor(backend, parameter);
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, gradient);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, first_moment);
    }
    if (status == LLM_OK) {
        status = validate_f32_tensor(backend, second_moment);
    }
    if (status != LLM_OK) {
        return status;
    }
    if (tensors_have_same_shape(parameter, gradient) == 0 ||
        tensors_have_same_shape(parameter, first_moment) == 0 ||
        tensors_have_same_shape(parameter, second_moment) == 0) {
        return LLM_INVALID_SHAPE;
    }
    const llm_tensor *tensors[] = {parameter, gradient, first_moment, second_moment};
    if (tensors_have_distinct_storage(tensors, 4U) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (backend->ops->adamw_update_f32 == NULL) {
        return LLM_UNSUPPORTED_OPERATION;
    }
    return backend->ops->adamw_update_f32(
        backend->context, (float *)parameter->storage->memory, (float *)gradient->storage->memory,
        (float *)first_moment->storage->memory, (float *)second_moment->storage->memory,
        parameter->element_count, options->learning_rate, options->beta1, options->beta2,
        options->epsilon, options->weight_decay, options->gradient_scale, options->step,
        options->zero_gradient != 0);
}
