/*
 * Host-side dispatch for the CUDA backend.
 *
 * Matrix products go to cuBLAS: it is the vendor BLAS for this hardware and no
 * hand-written tile loop in this repository would come close to it. Everything
 * else is a kernel in cuda_kernels.cu, because NVIDIA ships no equivalent.
 *
 * Each entry point validates its arguments exactly like the CPU and Metal
 * backends do, enqueues work on the context stream, then calls llm_cuda_finish:
 * outside a batch that synchronizes and turns a sticky device flag into
 * LLM_NUMERICAL_ERROR or LLM_INVALID_INDEX; inside a batch it defers both.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_internal.h"

namespace {

llm_cuda_context *as_context(void *opaque_context) {
    return static_cast<llm_cuda_context *>(opaque_context);
}

float *device_float(void *memory) { return static_cast<float *>(llm_cuda_device_pointer(memory)); }

const float *device_const_float(const void *memory) {
    return static_cast<const float *>(llm_cuda_device_const_pointer(memory));
}

const uint32_t *device_const_u32(const void *memory) {
    return static_cast<const uint32_t *>(llm_cuda_device_const_pointer(memory));
}

/* Grid dimension x is bounded by 2^31 - 1; a row count above that cannot launch. */
int row_count_is_launchable(size_t row_count) {
    return row_count != 0U && row_count <= 2147483647U;
}

llm_status blas_report(cublasStatus_t status, const char *stage) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return LLM_OK;
    }
    fprintf(stderr, "cuda: cublas %s failed with status %d\n", stage, (int)status);
    return LLM_BACKEND_ERROR;
}

llm_status finish_with_finite_check(llm_cuda_context *context, const void *output,
                                    size_t value_count) {
    if (context->numerics_mode == LLM_CUDA_NUMERICS_STEP) {
        return llm_cuda_finish(context);
    }
    const llm_status status = llm_cuda_check_finite(context, output, value_count);
    if (status != LLM_OK) {
        return status;
    }
    return llm_cuda_finish(context);
}

llm_status finish_with_required_finite_check(llm_cuda_context *context, const void *output,
                                             size_t value_count) {
    const llm_status status = llm_cuda_check_finite(context, output, value_count);
    return status == LLM_OK ? llm_cuda_finish(context) : status;
}

} // namespace

llm_status llm_cuda_fill_f32(void *opaque_context, float *values, size_t value_count, float value) {
    if (opaque_context == NULL || values == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_fill(context->stream, device_float(values), value_count, value);
    ++context->metrics.kernel_launches;
    return llm_cuda_finish(context);
}

llm_status llm_cuda_add_f32(void *opaque_context, const float *left, const float *right,
                            float *output, size_t value_count) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL ||
        value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_add(context->stream, device_const_float(left), device_const_float(right),
                        device_float(output), value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, value_count);
}

llm_status llm_cuda_multiply_f32(void *opaque_context, const float *left, const float *right,
                                 float *output, size_t value_count) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL ||
        value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_multiply(context->stream, device_const_float(left), device_const_float(right),
                             device_float(output), value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, value_count);
}

llm_status llm_cuda_scale_f32(void *opaque_context, const float *input, float scale, float *output,
                              size_t value_count) {
    if (opaque_context == NULL || input == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_scale(context->stream, device_const_float(input), scale, device_float(output),
                          value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, value_count);
}

llm_status llm_cuda_accumulate_f32(void *opaque_context, const float *source, float *destination,
                                   size_t value_count) {
    if (opaque_context == NULL || source == NULL || destination == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_accumulate(context->stream, device_const_float(source),
                               device_float(destination), value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, destination, value_count);
}

llm_status llm_cuda_silu_f32(void *opaque_context, const float *input, float *output,
                             size_t value_count) {
    if (opaque_context == NULL || input == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_silu(context->stream, device_const_float(input), device_float(output),
                         value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, value_count);
}

llm_status llm_cuda_silu_backward_f32(void *opaque_context, const float *input,
                                      const float *output_gradient, float *input_gradient,
                                      size_t value_count) {
    if (opaque_context == NULL || input == NULL || output_gradient == NULL ||
        input_gradient == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_silu_backward(context->stream, device_const_float(input),
                                  device_const_float(output_gradient), device_float(input_gradient),
                                  value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, input_gradient, value_count);
}

static llm_status cuda_reduce_last(void *opaque_context, const float *input, float *output,
                                   size_t outer_count, size_t reduction_size, int kind) {
    if (opaque_context == NULL || input == NULL || output == NULL || reduction_size == 0U ||
        row_count_is_launchable(outer_count) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (outer_count > SIZE_MAX / reduction_size) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    const float *source = device_const_float(input);
    float *destination = device_float(output);
    if (kind == 0) {
        llm_cuda_launch_reduce_sum_last(context->stream, source, destination, outer_count,
                                        reduction_size);
    } else if (kind == 1) {
        llm_cuda_launch_reduce_max_last(context->stream, source, destination, outer_count,
                                        reduction_size);
    } else {
        llm_cuda_launch_reduce_mean_square_last(context->stream, source, destination, outer_count,
                                                reduction_size);
    }
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, outer_count);
}

llm_status llm_cuda_reduce_sum_last_f32(void *context, const float *input, float *output,
                                        size_t outer_count, size_t reduction_size) {
    return cuda_reduce_last(context, input, output, outer_count, reduction_size, 0);
}

llm_status llm_cuda_reduce_max_last_f32(void *context, const float *input, float *output,
                                        size_t outer_count, size_t reduction_size) {
    return cuda_reduce_last(context, input, output, outer_count, reduction_size, 1);
}

llm_status llm_cuda_reduce_mean_square_last_f32(void *context, const float *input, float *output,
                                                size_t outer_count, size_t reduction_size) {
    return cuda_reduce_last(context, input, output, outer_count, reduction_size, 2);
}

llm_status llm_cuda_accumulate_sum_squares_f32(void *opaque_context, const float *input,
                                               float *accumulator, size_t value_count) {
    if (opaque_context == NULL || input == NULL || accumulator == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_accumulate_sum_squares(context->stream, device_const_float(input),
                                           device_float(accumulator), value_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, accumulator, 1U);
}

llm_status llm_cuda_softmax_last_f32(void *opaque_context, const float *input, float *output,
                                     size_t outer_count, size_t row_width) {
    if (opaque_context == NULL || input == NULL || output == NULL || row_width == 0U ||
        row_count_is_launchable(outer_count) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (outer_count > SIZE_MAX / row_width) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_softmax_last(context->stream, device_const_float(input), device_float(output),
                                 outer_count, row_width);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, outer_count * row_width);
}

/*
 * cuBLAS is column-major, so the row-major product C = op(A) . op(B) is issued as
 * its transpose: C^T = op(B)^T . op(A)^T. Reading a row-major matrix as
 * column-major already transposes it, which is why the operation flags pass
 * through unchanged and only the operand order is swapped.
 */
llm_status llm_cuda_matmul_ex_f32(void *opaque_context, const float *left, const float *right,
                                  float *output, size_t left_rows, size_t left_columns,
                                  size_t right_rows, size_t right_columns, int transpose_left,
                                  int transpose_right) {
    if (opaque_context == NULL || left == NULL || right == NULL || output == NULL ||
        left_rows == 0U || left_columns == 0U || right_rows == 0U || right_columns == 0U ||
        (transpose_left != 0 && transpose_left != 1) ||
        (transpose_right != 0 && transpose_right != 1)) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t result_rows = transpose_left != 0 ? left_columns : left_rows;
    const size_t interior = transpose_left != 0 ? left_rows : left_columns;
    const size_t right_interior = transpose_right != 0 ? right_columns : right_rows;
    const size_t result_columns = transpose_right != 0 ? right_rows : right_columns;
    if (interior != right_interior) {
        return LLM_INVALID_SHAPE;
    }
    if (result_rows > SIZE_MAX / result_columns || result_rows > INT32_MAX ||
        result_columns > INT32_MAX || interior > INT32_MAX || left_columns > INT32_MAX ||
        right_columns > INT32_MAX) {
        return LLM_OVERFLOW;
    }

    llm_cuda_context *context = as_context(opaque_context);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const cublasOperation_t operation_left = transpose_left != 0 ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_right = transpose_right != 0 ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasComputeType_t compute_type =
        context->math_mode == LLM_CUDA_MATH_TF32           ? CUBLAS_COMPUTE_32F_FAST_TF32
        : context->math_mode == LLM_CUDA_MATH_BF16_COMPUTE ? CUBLAS_COMPUTE_32F_FAST_16BF
                                                           : CUBLAS_COMPUTE_32F_PEDANTIC;
    const cublasStatus_t status = cublasGemmEx(
        context->blas, operation_right, operation_left, (int)result_columns, (int)result_rows,
        (int)interior, &alpha, device_const_float(right), CUDA_R_32F, (int)right_columns,
        device_const_float(left), CUDA_R_32F, (int)left_columns, &beta, device_float(output),
        CUDA_R_32F, (int)result_columns, compute_type, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return blas_report(status, "gemm_ex");
    }
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, result_rows * result_columns);
}

llm_status llm_cuda_matmul_f32(void *context, const float *left, const float *right, float *output,
                               size_t rows, size_t inner_size, size_t columns) {
    return llm_cuda_matmul_ex_f32(context, left, right, output, rows, inner_size, inner_size,
                                  columns, 0, 0);
}

llm_status llm_cuda_gather_rows_f32(void *opaque_context, const float *table, size_t row_count,
                                    size_t row_width, const uint32_t *indices, size_t index_count,
                                    float *output) {
    if (opaque_context == NULL || table == NULL || indices == NULL || output == NULL ||
        row_count == 0U || row_width == 0U || index_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    if (index_count > SIZE_MAX / row_width || row_count > SIZE_MAX / row_width) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    const llm_status check = llm_cuda_check_indices(context, indices, index_count, row_count);
    if (check != LLM_OK) {
        return check;
    }
    llm_cuda_launch_gather_rows(context->stream, device_const_float(table),
                                device_const_u32(indices), device_float(output), row_count,
                                row_width, index_count);
    ++context->metrics.kernel_launches;
    return llm_cuda_finish(context);
}

llm_status llm_cuda_scatter_add_rows_f32(void *opaque_context, const float *source,
                                         const uint32_t *indices, size_t index_count,
                                         size_t row_width, size_t row_count, float *table) {
    if (opaque_context == NULL || source == NULL || indices == NULL || table == NULL ||
        row_count == 0U || row_width == 0U || index_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    if (index_count > SIZE_MAX / row_width || row_count > SIZE_MAX / row_width) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    const llm_status check = llm_cuda_check_indices(context, indices, index_count, row_count);
    if (check != LLM_OK) {
        return check;
    }
    llm_cuda_launch_scatter_add_rows(context->stream, device_const_float(source),
                                     device_const_u32(indices), device_float(table), row_count,
                                     row_width, index_count);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, table, row_count * row_width);
}

llm_status llm_cuda_rms_norm_f32(void *opaque_context, const float *input, const float *weight,
                                 float epsilon, float *output, size_t outer_count,
                                 size_t row_width) {
    if (opaque_context == NULL || input == NULL || weight == NULL || output == NULL ||
        row_width == 0U || row_count_is_launchable(outer_count) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (outer_count > SIZE_MAX / row_width) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_rms_norm(context->stream, device_const_float(input), device_const_float(weight),
                             epsilon, device_float(output), outer_count, row_width);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, outer_count * row_width);
}

llm_status llm_cuda_rms_norm_backward_f32(void *opaque_context, const float *input,
                                          const float *weight, const float *output_gradient,
                                          float epsilon, float *input_gradient,
                                          float *weight_gradient, size_t outer_count,
                                          size_t row_width) {
    if (opaque_context == NULL || input == NULL || weight == NULL || output_gradient == NULL ||
        input_gradient == NULL || weight_gradient == NULL || row_width == 0U ||
        row_count_is_launchable(outer_count) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (outer_count > SIZE_MAX / row_width) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    /* The kernel accumulates the weight gradient with atomics, so it must start
       at zero, and the clear has to be queued on the stream to stay ordered
       against any work already enqueued by an open batch. */
    const llm_status clear = llm_cuda_zero(context, weight_gradient, row_width * sizeof(float));
    if (clear != LLM_OK) {
        return clear;
    }
    llm_cuda_launch_rms_norm_backward(
        context->stream, device_const_float(input), device_const_float(weight),
        device_const_float(output_gradient), epsilon, device_float(input_gradient),
        device_float(weight_gradient), outer_count, row_width);
    ++context->metrics.kernel_launches;
    if (context->numerics_mode == LLM_CUDA_NUMERICS_STRICT) {
        const llm_status check = llm_cuda_check_finite(context, weight_gradient, row_width);
        if (check != LLM_OK) {
            return check;
        }
    }
    return finish_with_finite_check(context, input_gradient, outer_count * row_width);
}

static llm_status cuda_rope_dispatch(void *opaque_context, const float *input,
                                     const float *cos_table, const float *sin_table,
                                     size_t batch_count, size_t sequence_length, size_t head_count,
                                     size_t head_dimension, float *output, int backward) {
    if (opaque_context == NULL || input == NULL || cos_table == NULL || sin_table == NULL ||
        output == NULL || batch_count == 0U || sequence_length == 0U || head_count == 0U ||
        head_dimension == 0U || head_dimension % 2U != 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t pairs_per_head = head_dimension / 2U;
    if (batch_count > SIZE_MAX / sequence_length ||
        batch_count * sequence_length > SIZE_MAX / head_count ||
        batch_count * sequence_length * head_count > SIZE_MAX / pairs_per_head) {
        return LLM_OVERFLOW;
    }
    const size_t pair_count = batch_count * sequence_length * head_count * pairs_per_head;
    llm_cuda_context *context = as_context(opaque_context);
    if (backward == 0) {
        llm_cuda_launch_rope(context->stream, device_const_float(input),
                             device_const_float(cos_table), device_const_float(sin_table),
                             device_float(output), pair_count, sequence_length, pairs_per_head,
                             head_count);
    } else {
        llm_cuda_launch_rope_backward(context->stream, device_const_float(input),
                                      device_const_float(cos_table), device_const_float(sin_table),
                                      device_float(output), pair_count, sequence_length,
                                      pairs_per_head, head_count);
    }
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, pair_count * 2U);
}

llm_status llm_cuda_rope_f32(void *context, const float *input, const float *cos_table,
                             const float *sin_table, size_t batch_count, size_t sequence_length,
                             size_t head_count, size_t head_dimension, float *output) {
    return cuda_rope_dispatch(context, input, cos_table, sin_table, batch_count, sequence_length,
                              head_count, head_dimension, output, 0);
}

llm_status llm_cuda_rope_backward_f32(void *context, const float *output_gradient,
                                      const float *cos_table, const float *sin_table,
                                      size_t batch_count, size_t sequence_length, size_t head_count,
                                      size_t head_dimension, float *input_gradient) {
    return cuda_rope_dispatch(context, output_gradient, cos_table, sin_table, batch_count,
                              sequence_length, head_count, head_dimension, input_gradient, 1);
}

/* Shared computation of the attention shapes, with the same validity rules the
   CPU backend applies: heads must divide evenly and every product must fit. */
static llm_status cuda_attention_shapes(size_t batch_count, size_t sequence_length,
                                        size_t query_head_count, size_t key_value_head_count,
                                        size_t head_dimension, size_t *out_query_rows,
                                        size_t *out_query_values, size_t *out_key_value_values) {
    if (batch_count == 0U || sequence_length == 0U || query_head_count == 0U ||
        key_value_head_count == 0U || head_dimension == 0U ||
        query_head_count % key_value_head_count != 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    if (batch_count > SIZE_MAX / sequence_length) {
        return LLM_OVERFLOW;
    }
    const size_t positions = batch_count * sequence_length;
    if (positions > SIZE_MAX / query_head_count) {
        return LLM_OVERFLOW;
    }
    const size_t query_rows = positions * query_head_count;
    if (query_rows > SIZE_MAX / head_dimension || positions > SIZE_MAX / key_value_head_count) {
        return LLM_OVERFLOW;
    }
    const size_t key_value_rows = positions * key_value_head_count;
    if (key_value_rows > SIZE_MAX / head_dimension || row_count_is_launchable(query_rows) == 0) {
        return LLM_OVERFLOW;
    }
    *out_query_rows = query_rows;
    *out_query_values = query_rows * head_dimension;
    *out_key_value_values = key_value_rows * head_dimension;
    return LLM_OK;
}

llm_status llm_cuda_attention_forward_f32(void *opaque_context, const float *query,
                                          const float *key, const float *value, float scale,
                                          size_t batch_count, size_t sequence_length,
                                          size_t query_head_count, size_t key_value_head_count,
                                          size_t head_dimension, float *output) {
    if (opaque_context == NULL || query == NULL || key == NULL || value == NULL || output == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    size_t query_rows = 0U, query_values = 0U, key_value_values = 0U;
    const llm_status shape_status =
        cuda_attention_shapes(batch_count, sequence_length, query_head_count, key_value_head_count,
                              head_dimension, &query_rows, &query_values, &key_value_values);
    if (shape_status != LLM_OK) {
        return shape_status;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_attention_forward(context->stream, device_const_float(query),
                                      device_const_float(key), device_const_float(value),
                                      device_float(output), scale, query_rows, sequence_length,
                                      query_head_count, key_value_head_count, head_dimension);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, output, query_values);
}

llm_status llm_cuda_attention_backward_f32(void *opaque_context, const float *query,
                                           const float *key, const float *value,
                                           const float *output_gradient, float scale,
                                           size_t batch_count, size_t sequence_length,
                                           size_t query_head_count, size_t key_value_head_count,
                                           size_t head_dimension, float *query_gradient,
                                           float *key_gradient, float *value_gradient) {
    if (opaque_context == NULL || query == NULL || key == NULL || value == NULL ||
        output_gradient == NULL || query_gradient == NULL || key_gradient == NULL ||
        value_gradient == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    size_t query_rows = 0U, query_values = 0U, key_value_values = 0U;
    const llm_status shape_status =
        cuda_attention_shapes(batch_count, sequence_length, query_head_count, key_value_head_count,
                              head_dimension, &query_rows, &query_values, &key_value_values);
    if (shape_status != LLM_OK) {
        return shape_status;
    }
    llm_cuda_context *context = as_context(opaque_context);
    /* Key and value gradients are accumulated with atomics from every causal
       query row, so both buffers start from zero. */
    llm_status status = llm_cuda_zero(context, key_gradient, key_value_values * sizeof(float));
    if (status == LLM_OK) {
        status = llm_cuda_zero(context, value_gradient, key_value_values * sizeof(float));
    }
    if (status != LLM_OK) {
        return status;
    }
    llm_cuda_launch_attention_backward(
        context->stream, device_const_float(query), device_const_float(key),
        device_const_float(value), device_const_float(output_gradient),
        device_float(query_gradient), device_float(key_gradient), device_float(value_gradient),
        scale, query_rows, sequence_length, query_head_count, key_value_head_count, head_dimension);
    ++context->metrics.kernel_launches;
    if (context->numerics_mode == LLM_CUDA_NUMERICS_STRICT) {
        status = llm_cuda_check_finite(context, key_gradient, key_value_values);
        if (status == LLM_OK) {
            status = llm_cuda_check_finite(context, value_gradient, key_value_values);
        }
        if (status != LLM_OK) {
            return status;
        }
    }
    return finish_with_finite_check(context, query_gradient, query_values);
}

llm_status llm_cuda_cross_entropy_forward_f32(void *opaque_context, const float *logits,
                                              const uint32_t *targets, size_t row_count,
                                              size_t vocabulary_size, float *loss) {
    if (opaque_context == NULL || logits == NULL || targets == NULL || loss == NULL ||
        vocabulary_size == 0U || row_count_is_launchable(row_count) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (row_count > SIZE_MAX / vocabulary_size) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_status status = llm_cuda_check_indices(context, targets, row_count, vocabulary_size);
    if (status != LLM_OK) {
        return status;
    }
    /* The kernel accumulates into this scalar, so it must start at zero. */
    status = llm_cuda_zero(context, loss, sizeof(float));
    if (status != LLM_OK) {
        return status;
    }
    llm_cuda_launch_cross_entropy_forward(context->stream, device_const_float(logits),
                                          device_const_u32(targets), device_float(loss), row_count,
                                          vocabulary_size);
    ++context->metrics.kernel_launches;
    return finish_with_required_finite_check(context, loss, 1U);
}

llm_status llm_cuda_cross_entropy_backward_f32(void *opaque_context, const float *logits,
                                               const uint32_t *targets, size_t row_count,
                                               size_t vocabulary_size, float *gradient) {
    if (opaque_context == NULL || logits == NULL || targets == NULL || gradient == NULL ||
        vocabulary_size == 0U || row_count_is_launchable(row_count) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    if (row_count > SIZE_MAX / vocabulary_size) {
        return LLM_OVERFLOW;
    }
    llm_cuda_context *context = as_context(opaque_context);
    const llm_status status = llm_cuda_check_indices(context, targets, row_count, vocabulary_size);
    if (status != LLM_OK) {
        return status;
    }
    llm_cuda_launch_cross_entropy_backward(context->stream, device_const_float(logits),
                                           device_const_u32(targets), device_float(gradient),
                                           row_count, vocabulary_size);
    ++context->metrics.kernel_launches;
    return finish_with_finite_check(context, gradient, row_count * vocabulary_size);
}

llm_status llm_cuda_adamw_update_f32(void *opaque_context, float *parameter, float *gradient,
                                     float *first_moment, float *second_moment, size_t value_count,
                                     float learning_rate, float beta1, float beta2, float epsilon,
                                     float weight_decay, float gradient_scale,
                                     unsigned long long step, int zero_gradient) {
    if (opaque_context == NULL || parameter == NULL || gradient == NULL || first_moment == NULL ||
        second_moment == NULL || value_count == 0U || step == 0ULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const double first_bias = 1.0 - pow((double)beta1, (double)step);
    const double second_bias = 1.0 - pow((double)beta2, (double)step);
    if (first_bias <= 0.0 || second_bias <= 0.0) {
        return LLM_NUMERICAL_ERROR;
    }
    llm_cuda_context *context = as_context(opaque_context);
    llm_cuda_launch_adamw(context->stream, device_float(parameter), device_float(gradient),
                          device_float(first_moment), device_float(second_moment), value_count,
                          learning_rate, beta1, beta2, epsilon, weight_decay, gradient_scale,
                          (float)(1.0 / first_bias), (float)(1.0 / second_bias), zero_gradient,
                          context->device_flags);
    ++context->metrics.kernel_launches;
    /* AdamW raises the same sticky non-finite flag while the values are already
       in registers. No second full read of every master parameter is needed. */
    return llm_cuda_finish(context);
}
