/*
 * Device kernels for the CUDA backend.
 *
 * Every kernel reproduces the semantics already fixed by the CPU and Metal
 * backends, including how a non-finite value is reported: the affected row
 * writes a quiet NaN, and the host side turns that into LLM_NUMERICAL_ERROR.
 * Matrix products are deliberately absent: cuBLAS owns them.
 */

#include <math.h>

#include "cuda_internal.h"

namespace {

constexpr unsigned int kFullWarpMask = 0xffffffffu;
constexpr int kWarpSize = 32;

__device__ inline float quiet_nan() { return __int_as_float(0x7fc00000); }

/* Butterfly shuffles so every lane leaves with the reduced value, matching the
   broadcast behaviour the Metal kernels rely on. */
__device__ inline float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ inline float warp_max(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(kFullWarpMask, value, offset));
    }
    return value;
}

/* partial must hold at least 32 floats and must not be read by the caller until
   the second barrier below has run. Both reductions broadcast to every thread. */
__device__ inline float block_sum(float value, float *partial) {
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warp_count = static_cast<int>(blockDim.x + kWarpSize - 1) / kWarpSize;
    value = warp_sum(value);
    if (warp_count == 1) {
        return value;
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    float total = static_cast<int>(threadIdx.x) < warp_count ? partial[threadIdx.x] : 0.0f;
    if (warp == 0) {
        total = warp_sum(total);
        if (lane == 0) {
            partial[0] = total;
        }
    }
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}

__device__ inline float block_max(float value, float *partial) {
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warp_count = static_cast<int>(blockDim.x + kWarpSize - 1) / kWarpSize;
    value = warp_max(value);
    if (warp_count == 1) {
        return value;
    }
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    float total = static_cast<int>(threadIdx.x) < warp_count ? partial[threadIdx.x] : -INFINITY;
    if (warp == 0) {
        total = warp_max(total);
        if (lane == 0) {
            partial[0] = total;
        }
    }
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}

__device__ inline size_t grid_stride() {
    return static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
}

__device__ inline size_t grid_index() {
    return static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
           static_cast<size_t>(threadIdx.x);
}

__device__ inline float stable_sigmoid(float value) {
    return value >= 0.0f ? 1.0f / (1.0f + expf(-value)) : expf(value) / (1.0f + expf(value));
}

__device__ inline size_t attention_offset(size_t batch, size_t sequence, size_t head,
                                          size_t sequence_length, size_t head_count,
                                          size_t head_dimension) {
    return ((batch * sequence_length + sequence) * head_count + head) * head_dimension;
}

__global__ void fill_kernel(float *output, size_t count, float value) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        output[index] = value;
    }
}

__global__ void add_kernel(const float *left, const float *right, float *output, size_t count) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        output[index] = left[index] + right[index];
    }
}

__global__ void multiply_kernel(const float *left, const float *right, float *output,
                                size_t count) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        output[index] = left[index] * right[index];
    }
}

__global__ void scale_kernel(const float *input, float scale, float *output, size_t count) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        output[index] = input[index] * scale;
    }
}

__global__ void accumulate_kernel(const float *source, float *destination, size_t count) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        destination[index] += source[index];
    }
}

__global__ void silu_kernel(const float *input, float *output, size_t count) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        output[index] = input[index] * stable_sigmoid(input[index]);
    }
}

__global__ void silu_backward_kernel(const float *input, const float *output_gradient,
                                     float *input_gradient, size_t count) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        const float sigmoid = stable_sigmoid(input[index]);
        input_gradient[index] =
            output_gradient[index] * sigmoid * (1.0f + input[index] * (1.0f - sigmoid));
    }
}

__global__ void reduce_sum_last_kernel(const float *input, float *output, size_t outer_count,
                                       size_t reduction_size) {
    const size_t row = blockIdx.x;
    if (row >= outer_count) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * reduction_size;
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < reduction_size; column += blockDim.x) {
        sum += input[offset + column];
    }
    sum = block_sum(sum, partial);
    if (threadIdx.x == 0U) {
        output[row] = sum;
    }
}

__global__ void reduce_max_last_kernel(const float *input, float *output, size_t outer_count,
                                       size_t reduction_size) {
    const size_t row = blockIdx.x;
    if (row >= outer_count) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * reduction_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (size_t column = threadIdx.x; column < reduction_size; column += blockDim.x) {
        const float value = input[offset + column];
        if (isfinite(value) == 0) {
            invalid = 1.0f;
        } else {
            maximum = fmaxf(maximum, value);
        }
    }
    invalid = block_sum(invalid, partial);
    maximum = block_max(maximum, partial);
    if (threadIdx.x == 0U) {
        output[row] = invalid > 0.0f ? quiet_nan() : maximum;
    }
}

__global__ void reduce_mean_square_last_kernel(const float *input, float *output,
                                               size_t outer_count, size_t reduction_size) {
    const size_t row = blockIdx.x;
    if (row >= outer_count) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * reduction_size;
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < reduction_size; column += blockDim.x) {
        const float value = input[offset + column];
        sum += value * value;
    }
    sum = block_sum(sum, partial);
    if (threadIdx.x == 0U) {
        output[row] = sum / static_cast<float>(reduction_size);
    }
}

__global__ void softmax_last_kernel(const float *input, float *output, size_t outer_count,
                                    size_t row_width) {
    const size_t row = blockIdx.x;
    if (row >= outer_count) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * row_width;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        const float value = input[offset + column];
        if (isfinite(value) == 0) {
            invalid = 1.0f;
        } else {
            maximum = fmaxf(maximum, value);
        }
    }
    invalid = block_sum(invalid, partial);
    maximum = block_max(maximum, partial);
    if (invalid > 0.0f) {
        if (threadIdx.x == 0U) {
            output[offset] = quiet_nan();
        }
        return;
    }
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        const float probability = expf(input[offset + column] - maximum);
        output[offset + column] = probability;
        sum += probability;
    }
    sum = block_sum(sum, partial);
    const float inverse_sum = 1.0f / sum;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        output[offset + column] *= inverse_sum;
    }
}

/* The bounds check is not redundant with the host-side validation scan: it keeps
   an out-of-range index from writing outside the table while the scan result is
   still in flight on the stream. */
__global__ void gather_rows_kernel(const float *table, const uint32_t *indices, float *output,
                                   size_t row_count, size_t row_width, size_t index_count) {
    const size_t count = index_count * row_width;
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        const size_t source_row = indices[index / row_width];
        const size_t column = index % row_width;
        output[index] = source_row < row_count ? table[source_row * row_width + column] : 0.0f;
    }
}

__global__ void scatter_add_rows_kernel(const float *source, const uint32_t *indices, float *table,
                                        size_t row_count, size_t row_width, size_t index_count) {
    const size_t count = index_count * row_width;
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        const size_t destination_row = indices[index / row_width];
        if (destination_row >= row_count) {
            continue;
        }
        const size_t column = index % row_width;
        atomicAdd(&table[destination_row * row_width + column], source[index]);
    }
}

__global__ void rms_norm_kernel(const float *input, const float *weight, float epsilon,
                                float *output, size_t outer_count, size_t row_width) {
    const size_t row = blockIdx.x;
    if (row >= outer_count) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * row_width;
    float square_sum = 0.0f;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        const float value = input[offset + column];
        square_sum += value * value;
    }
    square_sum = block_sum(square_sum, partial);
    const float inverse_rms = rsqrtf(square_sum / static_cast<float>(row_width) + epsilon);
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        output[offset + column] = input[offset + column] * inverse_rms * weight[column];
    }
}

__global__ void rms_norm_backward_kernel(const float *input, const float *weight,
                                         const float *output_gradient, float epsilon,
                                         float *input_gradient, float *weight_gradient,
                                         size_t outer_count, size_t row_width) {
    const size_t row = blockIdx.x;
    if (row >= outer_count) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * row_width;
    float square_sum = 0.0f;
    float projected_gradient = 0.0f;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        const float x = input[offset + column];
        square_sum += x * x;
        projected_gradient += output_gradient[offset + column] * weight[column] * x;
    }
    square_sum = block_sum(square_sum, partial);
    projected_gradient = block_sum(projected_gradient, partial);
    const float inverse_rms = rsqrtf(square_sum / static_cast<float>(row_width) + epsilon);
    const float correction = projected_gradient * inverse_rms * inverse_rms * inverse_rms /
                             static_cast<float>(row_width);
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        const size_t index = offset + column;
        input_gradient[index] =
            output_gradient[index] * weight[column] * inverse_rms - input[index] * correction;
        atomicAdd(&weight_gradient[column], output_gradient[index] * input[index] * inverse_rms);
    }
}

__global__ void rope_kernel(const float *input, const float *cos_table, const float *sin_table,
                            float *output, size_t pair_count, size_t sequence_length,
                            size_t pairs_per_head, size_t head_count) {
    const size_t pairs_per_position = head_count * pairs_per_head;
    for (size_t pair_index = grid_index(); pair_index < pair_count; pair_index += grid_stride()) {
        const size_t position_index = pair_index / pairs_per_position;
        const size_t pair = (pair_index % pairs_per_position) % pairs_per_head;
        const size_t sequence = position_index % sequence_length;
        const size_t value_index = pair_index * 2U;
        const size_t table_index = sequence * pairs_per_head + pair;
        const float first = input[value_index];
        const float second = input[value_index + 1U];
        const float cosine = cos_table[table_index];
        const float sine = sin_table[table_index];
        output[value_index] = first * cosine - second * sine;
        output[value_index + 1U] = first * sine + second * cosine;
    }
}

__global__ void rope_backward_kernel(const float *output_gradient, const float *cos_table,
                                     const float *sin_table, float *input_gradient,
                                     size_t pair_count, size_t sequence_length,
                                     size_t pairs_per_head, size_t head_count) {
    const size_t pairs_per_position = head_count * pairs_per_head;
    for (size_t pair_index = grid_index(); pair_index < pair_count; pair_index += grid_stride()) {
        const size_t position_index = pair_index / pairs_per_position;
        const size_t pair = (pair_index % pairs_per_position) % pairs_per_head;
        const size_t sequence = position_index % sequence_length;
        const size_t value_index = pair_index * 2U;
        const size_t table_index = sequence * pairs_per_head + pair;
        const float first = output_gradient[value_index];
        const float second = output_gradient[value_index + 1U];
        const float cosine = cos_table[table_index];
        const float sine = sin_table[table_index];
        input_gradient[value_index] = first * cosine + second * sine;
        input_gradient[value_index + 1U] = -first * sine + second * cosine;
    }
}

/*
 * One block per (batch, position, query head). Shared memory holds the causal row
 * of probabilities plus the 32 floats the block reduction needs. Softmax runs
 * across the whole block: each thread owns the key positions congruent to its
 * index, so the three passes never share an element.
 */
__global__ void attention_forward_kernel(const float *query, const float *key, const float *value,
                                         float *output, float scale, size_t query_rows,
                                         size_t sequence_length, size_t query_head_count,
                                         size_t key_value_head_count, size_t head_dimension) {
    const size_t query_row = blockIdx.x;
    if (query_row >= query_rows) {
        return;
    }
    extern __shared__ float shared[];
    float *probabilities = shared;
    float *partial = shared + sequence_length;

    const size_t rows_per_batch = sequence_length * query_head_count;
    const size_t batch = query_row / rows_per_batch;
    const size_t within_batch = query_row % rows_per_batch;
    const size_t query_position = within_batch / query_head_count;
    const size_t query_head = within_batch % query_head_count;
    const size_t heads_per_group = query_head_count / key_value_head_count;
    const size_t key_value_head = query_head / heads_per_group;
    const size_t query_index = attention_offset(batch, query_position, query_head, sequence_length,
                                                query_head_count, head_dimension);
    const float *query_row_values = query + query_index;

    for (size_t key_position = 0U; key_position <= query_position; ++key_position) {
        const size_t key_index =
            attention_offset(batch, key_position, key_value_head, sequence_length,
                             key_value_head_count, head_dimension);
        float dot = 0.0f;
        for (size_t dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
            dot += query_row_values[dimension] * key[key_index + dimension];
        }
        dot = block_sum(dot, partial);
        if (threadIdx.x == 0U) {
            probabilities[key_position] = dot * scale;
        }
        __syncthreads();
    }

    float maximum = -INFINITY;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        maximum = fmaxf(maximum, probabilities[key_position]);
    }
    maximum = block_max(maximum, partial);
    float denominator = 0.0f;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        const float probability = expf(probabilities[key_position] - maximum);
        probabilities[key_position] = probability;
        denominator += probability;
    }
    denominator = block_sum(denominator, partial);
    const float inverse_denominator = 1.0f / denominator;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        probabilities[key_position] *= inverse_denominator;
    }
    __syncthreads();

    for (size_t dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
        float result = 0.0f;
        for (size_t key_position = 0U; key_position <= query_position; ++key_position) {
            const size_t key_index =
                attention_offset(batch, key_position, key_value_head, sequence_length,
                                 key_value_head_count, head_dimension);
            result += probabilities[key_position] * value[key_index + dimension];
        }
        output[query_index + dimension] = result;
    }
}

__global__ void attention_backward_kernel(const float *query, const float *key, const float *value,
                                          const float *output_gradient, float *query_gradient,
                                          float *key_gradient, float *value_gradient, float scale,
                                          size_t query_rows, size_t sequence_length,
                                          size_t query_head_count, size_t key_value_head_count,
                                          size_t head_dimension) {
    const size_t query_row = blockIdx.x;
    if (query_row >= query_rows) {
        return;
    }
    extern __shared__ float shared[];
    float *probabilities = shared;
    float *probability_gradients = shared + sequence_length;
    float *partial = shared + 2U * sequence_length;

    const size_t rows_per_batch = sequence_length * query_head_count;
    const size_t batch = query_row / rows_per_batch;
    const size_t within_batch = query_row % rows_per_batch;
    const size_t query_position = within_batch / query_head_count;
    const size_t query_head = within_batch % query_head_count;
    const size_t heads_per_group = query_head_count / key_value_head_count;
    const size_t key_value_head = query_head / heads_per_group;
    const size_t query_index = attention_offset(batch, query_position, query_head, sequence_length,
                                                query_head_count, head_dimension);
    const float *query_row_values = query + query_index;
    const float *output_gradient_row = output_gradient + query_index;

    for (size_t key_position = 0U; key_position <= query_position; ++key_position) {
        const size_t key_index =
            attention_offset(batch, key_position, key_value_head, sequence_length,
                             key_value_head_count, head_dimension);
        float dot = 0.0f;
        for (size_t dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
            dot += query_row_values[dimension] * key[key_index + dimension];
        }
        dot = block_sum(dot, partial);
        if (threadIdx.x == 0U) {
            probabilities[key_position] = dot * scale;
        }
        __syncthreads();
    }

    float maximum = -INFINITY;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        maximum = fmaxf(maximum, probabilities[key_position]);
    }
    maximum = block_max(maximum, partial);
    float denominator = 0.0f;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        const float probability = expf(probabilities[key_position] - maximum);
        probabilities[key_position] = probability;
        denominator += probability;
    }
    denominator = block_sum(denominator, partial);
    const float inverse_denominator = 1.0f / denominator;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        probabilities[key_position] *= inverse_denominator;
    }
    __syncthreads();

    for (size_t key_position = 0U; key_position <= query_position; ++key_position) {
        const size_t key_index =
            attention_offset(batch, key_position, key_value_head, sequence_length,
                             key_value_head_count, head_dimension);
        float probability_gradient = 0.0f;
        for (size_t dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
            probability_gradient += output_gradient_row[dimension] * value[key_index + dimension];
        }
        probability_gradient = block_sum(probability_gradient, partial);
        if (threadIdx.x == 0U) {
            probability_gradients[key_position] = probability_gradient;
        }
        __syncthreads();
    }

    float weighted_probability_gradient = 0.0f;
    for (size_t key_position = threadIdx.x; key_position <= query_position;
         key_position += blockDim.x) {
        weighted_probability_gradient +=
            probabilities[key_position] * probability_gradients[key_position];
    }
    weighted_probability_gradient = block_sum(weighted_probability_gradient, partial);

    for (size_t dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
        float query_value_gradient = 0.0f;
        for (size_t key_position = 0U; key_position <= query_position; ++key_position) {
            const size_t key_index =
                attention_offset(batch, key_position, key_value_head, sequence_length,
                                 key_value_head_count, head_dimension);
            const float score_gradient =
                probabilities[key_position] *
                (probability_gradients[key_position] - weighted_probability_gradient);
            query_value_gradient += scale * score_gradient * key[key_index + dimension];
            atomicAdd(&key_gradient[key_index + dimension],
                      scale * score_gradient * query_row_values[dimension]);
            atomicAdd(&value_gradient[key_index + dimension],
                      probabilities[key_position] * output_gradient_row[dimension]);
        }
        query_gradient[query_index + dimension] = query_value_gradient;
    }
}

__global__ void cross_entropy_forward_kernel(const float *logits, const uint32_t *targets,
                                             float *loss, size_t row_count,
                                             size_t vocabulary_size) {
    const size_t row = blockIdx.x;
    if (row >= row_count) {
        return;
    }
    const uint32_t target = targets[row];
    if (target >= vocabulary_size) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * vocabulary_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        const float value = logits[offset + column];
        if (isfinite(value) == 0) {
            invalid = 1.0f;
        } else {
            maximum = fmaxf(maximum, value);
        }
    }
    invalid = block_sum(invalid, partial);
    maximum = block_max(maximum, partial);
    if (invalid > 0.0f) {
        if (threadIdx.x == 0U) {
            *loss = quiet_nan();
        }
        return;
    }
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        sum += expf(logits[offset + column] - maximum);
    }
    sum = block_sum(sum, partial);
    if (threadIdx.x == 0U) {
        const float row_loss = maximum + logf(sum) - logits[offset + target];
        atomicAdd(loss, row_loss / static_cast<float>(row_count));
    }
}

__global__ void cross_entropy_backward_kernel(const float *logits, const uint32_t *targets,
                                              float *gradient, size_t row_count,
                                              size_t vocabulary_size) {
    const size_t row = blockIdx.x;
    if (row >= row_count) {
        return;
    }
    const uint32_t target = targets[row];
    if (target >= vocabulary_size) {
        return;
    }
    __shared__ float partial[32];
    const size_t offset = row * vocabulary_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        const float value = logits[offset + column];
        if (isfinite(value) == 0) {
            invalid = 1.0f;
        } else {
            maximum = fmaxf(maximum, value);
        }
    }
    invalid = block_sum(invalid, partial);
    maximum = block_max(maximum, partial);
    if (invalid > 0.0f) {
        if (threadIdx.x == 0U) {
            gradient[offset] = quiet_nan();
        }
        return;
    }
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        sum += expf(logits[offset + column] - maximum);
    }
    sum = block_sum(sum, partial);
    const float scale = 1.0f / (sum * static_cast<float>(row_count));
    for (size_t column = threadIdx.x; column < vocabulary_size; column += blockDim.x) {
        float value = expf(logits[offset + column] - maximum) * scale;
        if (column == target) {
            value -= 1.0f / static_cast<float>(row_count);
        }
        gradient[offset + column] = value;
    }
}

__global__ void adamw_kernel(float *parameter, const float *gradient, float *first_moment,
                             float *second_moment, size_t count, float learning_rate, float beta1,
                             float beta2, float epsilon, float weight_decay, float gradient_scale,
                             float inverse_first_bias, float inverse_second_bias) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        const float scaled_gradient = gradient[index] * gradient_scale;
        const float first = beta1 * first_moment[index] + (1.0f - beta1) * scaled_gradient;
        const float second =
            beta2 * second_moment[index] + (1.0f - beta2) * scaled_gradient * scaled_gradient;
        const float corrected_first = first * inverse_first_bias;
        const float corrected_second = second * inverse_second_bias;
        parameter[index] -= learning_rate * (corrected_first / (sqrtf(corrected_second) + epsilon) +
                                             weight_decay * parameter[index]);
        first_moment[index] = first;
        second_moment[index] = second;
    }
}

__global__ void check_finite_kernel(const float *values, size_t count, int *flags) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        if (isfinite(values[index]) == 0) {
            flags[LLM_CUDA_FLAG_NON_FINITE] = 1;
            return;
        }
    }
}

__global__ void check_indices_kernel(const uint32_t *indices, size_t count, uint32_t bound,
                                     int *flags) {
    for (size_t index = grid_index(); index < count; index += grid_stride()) {
        if (indices[index] >= bound) {
            flags[LLM_CUDA_FLAG_INVALID_INDEX] = 1;
            return;
        }
    }
}

/* Caps the grid so a huge tensor does not ask for more blocks than the launch
   configuration allows; the grid-stride loops cover the remainder. */
unsigned int elementwise_blocks(size_t count, unsigned int block_size) {
    const size_t needed = (count + block_size - 1U) / block_size;
    const size_t capped = needed > 65535U ? 65535U : needed;
    return capped == 0U ? 1U : static_cast<unsigned int>(capped);
}

unsigned int row_block_size(size_t row_width) {
    unsigned int threads = kWarpSize;
    while (threads < LLM_CUDA_ROW_BLOCK && static_cast<size_t>(threads) < row_width) {
        threads *= 2U;
    }
    return threads;
}

} // namespace

void llm_cuda_launch_fill(cudaStream_t stream, float *output, size_t count, float value) {
    fill_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK), LLM_CUDA_ELEMENTWISE_BLOCK,
                  0, stream>>>(output, count, value);
}

void llm_cuda_launch_add(cudaStream_t stream, const float *left, const float *right, float *output,
                         size_t count) {
    add_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK), LLM_CUDA_ELEMENTWISE_BLOCK,
                 0, stream>>>(left, right, output, count);
}

void llm_cuda_launch_multiply(cudaStream_t stream, const float *left, const float *right,
                              float *output, size_t count) {
    multiply_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                      LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(left, right, output, count);
}

void llm_cuda_launch_scale(cudaStream_t stream, const float *input, float scale, float *output,
                           size_t count) {
    scale_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                   LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(input, scale, output, count);
}

void llm_cuda_launch_accumulate(cudaStream_t stream, const float *source, float *destination,
                                size_t count) {
    accumulate_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                        LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(source, destination, count);
}

void llm_cuda_launch_silu(cudaStream_t stream, const float *input, float *output, size_t count) {
    silu_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK), LLM_CUDA_ELEMENTWISE_BLOCK,
                  0, stream>>>(input, output, count);
}

void llm_cuda_launch_silu_backward(cudaStream_t stream, const float *input,
                                   const float *output_gradient, float *input_gradient,
                                   size_t count) {
    silu_backward_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                           LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(input, output_gradient,
                                                                    input_gradient, count);
}

void llm_cuda_launch_reduce_sum_last(cudaStream_t stream, const float *input, float *output,
                                     size_t outer_count, size_t reduction_size) {
    reduce_sum_last_kernel<<<static_cast<unsigned int>(outer_count), row_block_size(reduction_size),
                             0, stream>>>(input, output, outer_count, reduction_size);
}

void llm_cuda_launch_reduce_max_last(cudaStream_t stream, const float *input, float *output,
                                     size_t outer_count, size_t reduction_size) {
    reduce_max_last_kernel<<<static_cast<unsigned int>(outer_count), row_block_size(reduction_size),
                             0, stream>>>(input, output, outer_count, reduction_size);
}

void llm_cuda_launch_reduce_mean_square_last(cudaStream_t stream, const float *input, float *output,
                                             size_t outer_count, size_t reduction_size) {
    reduce_mean_square_last_kernel<<<static_cast<unsigned int>(outer_count),
                                     row_block_size(reduction_size), 0, stream>>>(
        input, output, outer_count, reduction_size);
}

void llm_cuda_launch_softmax_last(cudaStream_t stream, const float *input, float *output,
                                  size_t outer_count, size_t row_width) {
    softmax_last_kernel<<<static_cast<unsigned int>(outer_count), row_block_size(row_width), 0,
                          stream>>>(input, output, outer_count, row_width);
}

void llm_cuda_launch_gather_rows(cudaStream_t stream, const float *table, const uint32_t *indices,
                                 float *output, size_t row_count, size_t row_width,
                                 size_t index_count) {
    const size_t count = index_count * row_width;
    gather_rows_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                         LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(table, indices, output, row_count,
                                                                  row_width, index_count);
}

void llm_cuda_launch_scatter_add_rows(cudaStream_t stream, const float *source,
                                      const uint32_t *indices, float *table, size_t row_count,
                                      size_t row_width, size_t index_count) {
    const size_t count = index_count * row_width;
    scatter_add_rows_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                              LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(
        source, indices, table, row_count, row_width, index_count);
}

void llm_cuda_launch_rms_norm(cudaStream_t stream, const float *input, const float *weight,
                              float epsilon, float *output, size_t outer_count, size_t row_width) {
    rms_norm_kernel<<<static_cast<unsigned int>(outer_count), row_block_size(row_width), 0,
                      stream>>>(input, weight, epsilon, output, outer_count, row_width);
}

void llm_cuda_launch_rms_norm_backward(cudaStream_t stream, const float *input, const float *weight,
                                       const float *output_gradient, float epsilon,
                                       float *input_gradient, float *weight_gradient,
                                       size_t outer_count, size_t row_width) {
    rms_norm_backward_kernel<<<static_cast<unsigned int>(outer_count), row_block_size(row_width), 0,
                               stream>>>(input, weight, output_gradient, epsilon, input_gradient,
                                         weight_gradient, outer_count, row_width);
}

void llm_cuda_launch_rope(cudaStream_t stream, const float *input, const float *cos_table,
                          const float *sin_table, float *output, size_t pair_count,
                          size_t sequence_length, size_t pairs_per_head, size_t head_count) {
    rope_kernel<<<elementwise_blocks(pair_count, LLM_CUDA_ELEMENTWISE_BLOCK),
                  LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(input, cos_table, sin_table, output,
                                                           pair_count, sequence_length,
                                                           pairs_per_head, head_count);
}

void llm_cuda_launch_rope_backward(cudaStream_t stream, const float *output_gradient,
                                   const float *cos_table, const float *sin_table,
                                   float *input_gradient, size_t pair_count, size_t sequence_length,
                                   size_t pairs_per_head, size_t head_count) {
    rope_backward_kernel<<<elementwise_blocks(pair_count, LLM_CUDA_ELEMENTWISE_BLOCK),
                           LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(
        output_gradient, cos_table, sin_table, input_gradient, pair_count, sequence_length,
        pairs_per_head, head_count);
}

void llm_cuda_launch_attention_forward(cudaStream_t stream, const float *query, const float *key,
                                       const float *value, float *output, float scale,
                                       size_t query_rows, size_t sequence_length,
                                       size_t query_head_count, size_t key_value_head_count,
                                       size_t head_dimension) {
    const size_t shared_bytes = (sequence_length + 32U) * sizeof(float);
    attention_forward_kernel<<<static_cast<unsigned int>(query_rows), LLM_CUDA_ATTENTION_BLOCK,
                               shared_bytes, stream>>>(query, key, value, output, scale, query_rows,
                                                       sequence_length, query_head_count,
                                                       key_value_head_count, head_dimension);
}

void llm_cuda_launch_attention_backward(cudaStream_t stream, const float *query, const float *key,
                                        const float *value, const float *output_gradient,
                                        float *query_gradient, float *key_gradient,
                                        float *value_gradient, float scale, size_t query_rows,
                                        size_t sequence_length, size_t query_head_count,
                                        size_t key_value_head_count, size_t head_dimension) {
    const size_t shared_bytes = (2U * sequence_length + 32U) * sizeof(float);
    attention_backward_kernel<<<static_cast<unsigned int>(query_rows), LLM_CUDA_ATTENTION_BLOCK,
                                shared_bytes, stream>>>(
        query, key, value, output_gradient, query_gradient, key_gradient, value_gradient, scale,
        query_rows, sequence_length, query_head_count, key_value_head_count, head_dimension);
}

void llm_cuda_launch_cross_entropy_forward(cudaStream_t stream, const float *logits,
                                           const uint32_t *targets, float *loss, size_t row_count,
                                           size_t vocabulary_size) {
    cross_entropy_forward_kernel<<<static_cast<unsigned int>(row_count),
                                   row_block_size(vocabulary_size), 0, stream>>>(
        logits, targets, loss, row_count, vocabulary_size);
}

void llm_cuda_launch_cross_entropy_backward(cudaStream_t stream, const float *logits,
                                            const uint32_t *targets, float *gradient,
                                            size_t row_count, size_t vocabulary_size) {
    cross_entropy_backward_kernel<<<static_cast<unsigned int>(row_count),
                                    row_block_size(vocabulary_size), 0, stream>>>(
        logits, targets, gradient, row_count, vocabulary_size);
}

void llm_cuda_launch_adamw(cudaStream_t stream, float *parameter, const float *gradient,
                           float *first_moment, float *second_moment, size_t count,
                           float learning_rate, float beta1, float beta2, float epsilon,
                           float weight_decay, float gradient_scale, float inverse_first_bias,
                           float inverse_second_bias) {
    adamw_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                   LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(
        parameter, gradient, first_moment, second_moment, count, learning_rate, beta1, beta2,
        epsilon, weight_decay, gradient_scale, inverse_first_bias, inverse_second_bias);
}

void llm_cuda_launch_check_finite(cudaStream_t stream, const float *values, size_t count,
                                  int *flags) {
    check_finite_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                          LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(values, count, flags);
}

void llm_cuda_launch_check_indices(cudaStream_t stream, const uint32_t *indices, size_t count,
                                   uint32_t bound, int *flags) {
    check_indices_kernel<<<elementwise_blocks(count, LLM_CUDA_ELEMENTWISE_BLOCK),
                           LLM_CUDA_ELEMENTWISE_BLOCK, 0, stream>>>(indices, count, bound, flags);
}
