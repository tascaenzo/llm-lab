#include <math.h>
#include <string.h>

#include "cpu_kernels.h"
#include "cpu_simd.h"

#define LLM_CPU_MATMUL_INNER_TILE 64U
#define LLM_CPU_MATMUL_COLUMN_TILE 64U

llm_status llm_cpu_add_f32(const float *left, const float *right, float *output,
                           size_t value_count) {
    if (left == NULL || right == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cpu_simd_add_f32(left, right, output, value_count);
    return LLM_OK;
}

llm_status llm_cpu_multiply_f32(const float *left, const float *right, float *output,
                                size_t value_count) {
    if (left == NULL || right == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cpu_simd_multiply_f32(left, right, output, value_count);
    return LLM_OK;
}

llm_status llm_cpu_scale_f32(const float *input, float scale, float *output, size_t value_count) {
    if (input == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_cpu_simd_scale_f32(input, scale, output, value_count);
    return LLM_OK;
}

llm_status llm_cpu_reduce_sum_last_f32(const float *input, float *output, size_t outer_count,
                                       size_t reduction_size) {
    if (input == NULL || output == NULL || outer_count == 0U || reduction_size == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row = 0U; row < outer_count; ++row) {
        float sum = 0.0F;
        const float *values = input + row * reduction_size;
        for (size_t column = 0U; column < reduction_size; ++column) {
            sum += values[column];
        }
        if (isfinite(sum) == 0) {
            return LLM_NUMERICAL_ERROR;
        }
        output[row] = sum;
    }
    return LLM_OK;
}

llm_status llm_cpu_reduce_max_last_f32(const float *input, float *output, size_t outer_count,
                                       size_t reduction_size) {
    if (input == NULL || output == NULL || outer_count == 0U || reduction_size == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row = 0U; row < outer_count; ++row) {
        const float *values = input + row * reduction_size;
        float maximum = values[0];
        if (isfinite(maximum) == 0) {
            return LLM_NUMERICAL_ERROR;
        }
        for (size_t column = 1U; column < reduction_size; ++column) {
            if (isfinite(values[column]) == 0) {
                return LLM_NUMERICAL_ERROR;
            }
            if (values[column] > maximum) {
                maximum = values[column];
            }
        }
        output[row] = maximum;
    }
    return LLM_OK;
}

llm_status llm_cpu_reduce_mean_square_last_f32(const float *input, float *output,
                                               size_t outer_count, size_t reduction_size) {
    if (input == NULL || output == NULL || outer_count == 0U || reduction_size == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row = 0U; row < outer_count; ++row) {
        float square_sum = 0.0F;
        const float *values = input + row * reduction_size;
        for (size_t column = 0U; column < reduction_size; ++column) {
            square_sum += values[column] * values[column];
        }
        const float mean_square = square_sum / (float)reduction_size;
        if (isfinite(mean_square) == 0) {
            return LLM_NUMERICAL_ERROR;
        }
        output[row] = mean_square;
    }
    return LLM_OK;
}

llm_status llm_cpu_matmul_f32(const float *left, const float *right, float *output, size_t rows,
                              size_t inner_size, size_t columns) {
    if (left == NULL || right == NULL || output == NULL || rows == 0U || inner_size == 0U ||
        columns == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row_begin = 0U; row_begin < rows; row_begin += 4U) {
        const size_t block_rows = rows - row_begin < 4U ? rows - row_begin : 4U;
        float *output_rows[4] = {NULL, NULL, NULL, NULL};
        for (size_t block_row = 0U; block_row < block_rows; ++block_row) {
            output_rows[block_row] = output + (row_begin + block_row) * columns;
            (void)memset(output_rows[block_row], 0, columns * sizeof(*output));
        }
        for (size_t inner_begin = 0U; inner_begin < inner_size;
             inner_begin += LLM_CPU_MATMUL_INNER_TILE) {
            const size_t inner_end = inner_size - inner_begin < LLM_CPU_MATMUL_INNER_TILE
                                         ? inner_size
                                         : inner_begin + LLM_CPU_MATMUL_INNER_TILE;
            for (size_t column_begin = 0U; column_begin < columns;
                 column_begin += LLM_CPU_MATMUL_COLUMN_TILE) {
                const size_t column_end = columns - column_begin < LLM_CPU_MATMUL_COLUMN_TILE
                                              ? columns
                                              : column_begin + LLM_CPU_MATMUL_COLUMN_TILE;
                for (size_t inner = inner_begin; inner < inner_end; ++inner) {
                    const float *right_row = right + inner * columns;
                    if (block_rows == 4U) {
                        const float scales[4] = {
                            left[(row_begin + 0U) * inner_size + inner],
                            left[(row_begin + 1U) * inner_size + inner],
                            left[(row_begin + 2U) * inner_size + inner],
                            left[(row_begin + 3U) * inner_size + inner],
                        };
                        float *column_outputs[4] = {
                            output_rows[0] + column_begin,
                            output_rows[1] + column_begin,
                            output_rows[2] + column_begin,
                            output_rows[3] + column_begin,
                        };
                        llm_cpu_simd_axpy4_f32(right_row + column_begin, scales, column_outputs,
                                               column_end - column_begin);
                    } else {
                        for (size_t block_row = 0U; block_row < block_rows; ++block_row) {
                            const float left_value =
                                left[(row_begin + block_row) * inner_size + inner];
                            llm_cpu_simd_axpy_f32(right_row + column_begin, left_value,
                                                  output_rows[block_row] + column_begin,
                                                  column_end - column_begin);
                        }
                    }
                }
            }
        }
        for (size_t block_row = 0U; block_row < block_rows; ++block_row) {
            for (size_t column = 0U; column < columns; ++column) {
                if (isfinite(output_rows[block_row][column]) == 0) {
                    return LLM_NUMERICAL_ERROR;
                }
            }
        }
    }
    return LLM_OK;
}

llm_status llm_cpu_gather_rows_f32(const float *table, size_t row_count, size_t row_width,
                                   const uint32_t *indices, size_t index_count, float *output) {
    if (table == NULL || indices == NULL || output == NULL || row_count == 0U || row_width == 0U ||
        index_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t index = 0U; index < index_count; ++index) {
        if ((size_t)indices[index] >= row_count) {
            return LLM_INVALID_INDEX;
        }
    }
    for (size_t index = 0U; index < index_count; ++index) {
        const size_t row = (size_t)indices[index];
        (void)memcpy(output + index * row_width, table + row * row_width,
                     row_width * sizeof(*output));
    }
    return LLM_OK;
}

llm_status llm_cpu_scatter_add_rows_f32(const float *source, const uint32_t *indices,
                                        size_t index_count, size_t row_width, size_t row_count,
                                        float *table) {
    if (source == NULL || indices == NULL || table == NULL || index_count == 0U ||
        row_width == 0U || row_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t index = 0U; index < index_count; ++index) {
        if ((size_t)indices[index] >= row_count) {
            return LLM_INVALID_INDEX;
        }
    }
    for (size_t index = 0U; index < index_count; ++index) {
        const size_t row = (size_t)indices[index];
        for (size_t column = 0U; column < row_width; ++column) {
            table[row * row_width + column] += source[index * row_width + column];
        }
    }
    return LLM_OK;
}

llm_status llm_cpu_softmax_last_f32(const float *input, float *output, size_t outer_count,
                                    size_t row_width) {
    if (input == NULL || output == NULL || outer_count == 0U || row_width == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row = 0U; row < outer_count; ++row) {
        const float *input_row = input + row * row_width;
        float *output_row = output + row * row_width;
        float maximum = input_row[0];
        if (isfinite(maximum) == 0) {
            return LLM_NUMERICAL_ERROR;
        }
        for (size_t column = 1U; column < row_width; ++column) {
            if (isfinite(input_row[column]) == 0) {
                return LLM_NUMERICAL_ERROR;
            }
            if (input_row[column] > maximum) {
                maximum = input_row[column];
            }
        }

        float sum = 0.0F;
        for (size_t column = 0U; column < row_width; ++column) {
            output_row[column] = expf(input_row[column] - maximum);
            sum += output_row[column];
        }
        if (isfinite(sum) == 0 || sum <= 0.0F) {
            return LLM_NUMERICAL_ERROR;
        }
        const float inverse_sum = 1.0F / sum;
        for (size_t column = 0U; column < row_width; ++column) {
            output_row[column] *= inverse_sum;
        }
    }
    return LLM_OK;
}

static llm_status find_row_softmax_denominator(const float *row, size_t row_width,
                                               float *out_maximum, float *out_sum) {
    float maximum = row[0];
    if (isfinite(maximum) == 0) {
        return LLM_NUMERICAL_ERROR;
    }
    for (size_t column = 1U; column < row_width; ++column) {
        if (isfinite(row[column]) == 0) {
            return LLM_NUMERICAL_ERROR;
        }
        if (row[column] > maximum) {
            maximum = row[column];
        }
    }
    float sum = 0.0F;
    for (size_t column = 0U; column < row_width; ++column) {
        sum += expf(row[column] - maximum);
    }
    if (isfinite(sum) == 0 || sum <= 0.0F) {
        return LLM_NUMERICAL_ERROR;
    }
    *out_maximum = maximum;
    *out_sum = sum;
    return LLM_OK;
}

llm_status llm_cpu_cross_entropy_forward_f32(const float *logits, const uint32_t *targets,
                                             const uint32_t *loss_mask, size_t row_count,
                                             size_t vocabulary_size,
                                             size_t normalization_row_count, float *loss) {
    if (logits == NULL || targets == NULL || loss == NULL || row_count == 0U ||
        vocabulary_size == 0U || normalization_row_count == 0U ||
        normalization_row_count > row_count) {
        return LLM_INVALID_ARGUMENT;
    }
    double loss_sum = 0.0;
    for (size_t row = 0U; row < row_count; ++row) {
        if (loss_mask != NULL && loss_mask[row] > 1U) {
            return LLM_INVALID_ARGUMENT;
        }
        if ((size_t)targets[row] >= vocabulary_size) {
            return LLM_INVALID_INDEX;
        }
    }
    for (size_t row = 0U; row < row_count; ++row) {
        if (loss_mask != NULL && loss_mask[row] == 0U) {
            continue;
        }
        const float *logit_row = logits + row * vocabulary_size;
        float maximum = 0.0F;
        float sum = 0.0F;
        const llm_status status =
            find_row_softmax_denominator(logit_row, vocabulary_size, &maximum, &sum);
        if (status != LLM_OK) {
            return status;
        }
        const float row_loss = maximum + logf(sum) - logit_row[(size_t)targets[row]];
        if (isfinite(row_loss) == 0) {
            return LLM_NUMERICAL_ERROR;
        }
        loss_sum += (double)row_loss;
    }
    const double mean_loss = loss_sum / (double)normalization_row_count;
    if (isfinite(mean_loss) == 0) {
        return LLM_NUMERICAL_ERROR;
    }
    *loss = (float)mean_loss;
    return LLM_OK;
}

llm_status llm_cpu_cross_entropy_backward_f32(const float *logits, const uint32_t *targets,
                                              const uint32_t *loss_mask, size_t row_count,
                                              size_t vocabulary_size,
                                              size_t normalization_row_count, float *gradient) {
    if (logits == NULL || targets == NULL || gradient == NULL || row_count == 0U ||
        vocabulary_size == 0U || normalization_row_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t row = 0U; row < row_count; ++row) {
        if (loss_mask != NULL && loss_mask[row] > 1U) {
            return LLM_INVALID_ARGUMENT;
        }
        if ((size_t)targets[row] >= vocabulary_size) {
            return LLM_INVALID_INDEX;
        }
    }
    const float inverse_rows = 1.0F / (float)normalization_row_count;
    for (size_t row = 0U; row < row_count; ++row) {
        const float *logit_row = logits + row * vocabulary_size;
        float *gradient_row = gradient + row * vocabulary_size;
        if (loss_mask != NULL && loss_mask[row] == 0U) {
            for (size_t column = 0U; column < vocabulary_size; ++column) {
                gradient_row[column] = 0.0F;
            }
            continue;
        }
        float maximum = 0.0F;
        float sum = 0.0F;
        const llm_status status =
            find_row_softmax_denominator(logit_row, vocabulary_size, &maximum, &sum);
        if (status != LLM_OK) {
            return status;
        }
        const float inverse_sum = 1.0F / sum;
        for (size_t column = 0U; column < vocabulary_size; ++column) {
            gradient_row[column] = expf(logit_row[column] - maximum) * inverse_sum * inverse_rows;
        }
        gradient_row[(size_t)targets[row]] -= inverse_rows;
    }
    return LLM_OK;
}
