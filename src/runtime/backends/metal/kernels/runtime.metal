#include <metal_stdlib>

using namespace metal;

struct ElementwiseParameters {
    uint count;
    float scalar;
};

struct ReductionParameters {
    uint outer_count;
    uint reduction_size;
};

struct MatmulParameters {
    uint rows;
    uint inner_size;
    uint columns;
};

struct GatherParameters {
    uint row_count;
    uint row_width;
    uint index_count;
};

struct CrossEntropyParameters {
    uint row_count;
    uint vocabulary_size;
    uint normalization_row_count;
    uint use_loss_mask;
};

struct RmsNormParameters {
    uint outer_count;
    uint row_width;
    float epsilon;
};

struct RopeParameters {
    uint sequence_length;
    uint head_count;
    uint pairs_per_head;
};

struct RopePositionParameters {
    uint head_count;
    uint pairs_per_head;
    uint position;
};

struct AttentionParameters {
    uint batch_count;
    uint sequence_length;
    uint query_head_count;
    uint key_value_head_count;
    uint head_dimension;
    float scale;
};

struct AttentionDecodeParameters {
    uint batch_count;
    uint cache_capacity;
    uint head_count;
    uint head_dimension;
    uint position;
    float scale;
};

struct AdamwParameters {
    uint count;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float gradient_scale;
    float inverse_first_bias;
    float inverse_second_bias;
    uint zero_gradient;
};

constant uint llm_simd_width = 32;

inline float llm_threadgroup_sum(float value, threadgroup float *partial, uint thread_index,
                                 uint lane, uint simdgroup, uint threads_per_group) {
    /* A single SIMD group reduces in one hardware instruction, with no barrier. */
    if (threads_per_group <= llm_simd_width) {
        return simd_sum(value);
    }
    const float simd_value = simd_sum(value);
    if (lane == 0) {
        partial[simdgroup] = simd_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint simdgroup_count = (threads_per_group + llm_simd_width - 1) / llm_simd_width;
    float group_value = thread_index < simdgroup_count ? partial[thread_index] : 0.0f;
    if (simdgroup == 0) {
        group_value = simd_sum(group_value);
        if (lane == 0) {
            partial[0] = group_value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = partial[0];
    /* Every SIMD group must consume the result before partial is reused. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline float llm_threadgroup_max(float value, threadgroup float *partial, uint thread_index,
                                 uint lane, uint simdgroup, uint threads_per_group) {
    /* A single SIMD group reduces in one hardware instruction, with no barrier. */
    if (threads_per_group <= llm_simd_width) {
        return simd_max(value);
    }
    const float simd_value = simd_max(value);
    if (lane == 0) {
        partial[simdgroup] = simd_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint simdgroup_count = (threads_per_group + llm_simd_width - 1) / llm_simd_width;
    float group_value = thread_index < simdgroup_count ? partial[thread_index] : -INFINITY;
    if (simdgroup == 0) {
        group_value = simd_max(group_value);
        if (lane == 0) {
            partial[0] = group_value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = partial[0];
    /* Every SIMD group must consume the result before partial is reused. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void llm_fill_f32(device float *output [[buffer(0)]],
                         constant ElementwiseParameters &parameters [[buffer(1)]],
                         uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    if (base + 3 < parameters.count) {
        reinterpret_cast<device float4 *>(output)[vector_index] = float4(parameters.scalar);
    } else {
        for (uint index = base; index < parameters.count; ++index) {
            output[index] = parameters.scalar;
        }
    }
}

kernel void llm_add_f32(device const float *left [[buffer(0)]],
                        device const float *right [[buffer(1)]], device float *output [[buffer(2)]],
                        constant ElementwiseParameters &parameters [[buffer(3)]],
                        uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    if (base + 3 < parameters.count) {
        reinterpret_cast<device float4 *>(output)[vector_index] =
            reinterpret_cast<device const float4 *>(left)[vector_index] +
            reinterpret_cast<device const float4 *>(right)[vector_index];
    } else {
        for (uint index = base; index < parameters.count; ++index) {
            output[index] = left[index] + right[index];
        }
    }
}

kernel void llm_multiply_f32(device const float *left [[buffer(0)]],
                             device const float *right [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant ElementwiseParameters &parameters [[buffer(3)]],
                             uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    if (base + 3 < parameters.count) {
        reinterpret_cast<device float4 *>(output)[vector_index] =
            reinterpret_cast<device const float4 *>(left)[vector_index] *
            reinterpret_cast<device const float4 *>(right)[vector_index];
    } else {
        for (uint index = base; index < parameters.count; ++index) {
            output[index] = left[index] * right[index];
        }
    }
}

kernel void llm_scale_f32(device const float *input [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          constant ElementwiseParameters &parameters [[buffer(2)]],
                          uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    if (base + 3 < parameters.count) {
        reinterpret_cast<device float4 *>(output)[vector_index] =
            reinterpret_cast<device const float4 *>(input)[vector_index] * parameters.scalar;
    } else {
        for (uint index = base; index < parameters.count; ++index) {
            output[index] = input[index] * parameters.scalar;
        }
    }
}

kernel void llm_accumulate_f32(device const float *source [[buffer(0)]],
                               device float *destination [[buffer(1)]],
                               constant ElementwiseParameters &parameters [[buffer(2)]],
                               uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    if (base + 3 < parameters.count) {
        reinterpret_cast<device float4 *>(destination)[vector_index] +=
            reinterpret_cast<device const float4 *>(source)[vector_index];
    } else {
        for (uint index = base; index < parameters.count; ++index) {
            destination[index] += source[index];
        }
    }
}

inline float llm_sigmoid(float value) {
    return value >= 0.0f ? 1.0f / (1.0f + exp(-value)) : exp(value) / (1.0f + exp(value));
}

inline void llm_atomic_add_float(device atomic_uint *destination, float value);
inline void llm_atomic_add_float(device atomic_float *destination, float value);

kernel void llm_silu_f32(device const float *input [[buffer(0)]],
                         device float *output [[buffer(1)]],
                         constant ElementwiseParameters &parameters [[buffer(2)]],
                         uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    for (uint index = base; index < min(base + 4, parameters.count); ++index) {
        output[index] = input[index] * llm_sigmoid(input[index]);
    }
}

kernel void llm_silu_backward_f32(device const float *input [[buffer(0)]],
                                  device const float *output_gradient [[buffer(1)]],
                                  device float *input_gradient [[buffer(2)]],
                                  constant ElementwiseParameters &parameters [[buffer(3)]],
                                  uint vector_index [[thread_position_in_grid]]) {
    const uint base = vector_index * 4;
    for (uint index = base; index < min(base + 4, parameters.count); ++index) {
        const float sigmoid = llm_sigmoid(input[index]);
        input_gradient[index] =
            output_gradient[index] * sigmoid * (1.0f + input[index] * (1.0f - sigmoid));
    }
}

kernel void llm_rms_norm_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant RmsNormParameters &parameters [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.outer_count) {
        return;
    }
    threadgroup float partial[32];
    const uint offset = row * parameters.row_width;
    float square_sum = 0.0f;
    for (uint column = thread_index; column < parameters.row_width; column += threads_per_group) {
        const float value = input[offset + column];
        square_sum += value * value;
    }
    square_sum =
        llm_threadgroup_sum(square_sum, partial, thread_index, lane, simdgroup, threads_per_group);
    const float inverse_rms = rsqrt(square_sum / float(parameters.row_width) + parameters.epsilon);
    for (uint column = thread_index; column < parameters.row_width; column += threads_per_group) {
        output[offset + column] = input[offset + column] * inverse_rms * weight[column];
    }
}

kernel void llm_rms_norm_backward_f32(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *output_gradient [[buffer(2)]], device float *input_gradient [[buffer(3)]],
    device atomic_uint *weight_gradient [[buffer(4)]],
    constant RmsNormParameters &parameters [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.outer_count) {
        return;
    }
    threadgroup float partial[32];
    const uint offset = row * parameters.row_width;
    float square_sum = 0.0f;
    float projected_gradient = 0.0f;
    for (uint column = thread_index; column < parameters.row_width; column += threads_per_group) {
        const float x = input[offset + column];
        square_sum += x * x;
        projected_gradient += output_gradient[offset + column] * weight[column] * x;
    }
    square_sum =
        llm_threadgroup_sum(square_sum, partial, thread_index, lane, simdgroup, threads_per_group);
    projected_gradient = llm_threadgroup_sum(projected_gradient, partial, thread_index, lane,
                                             simdgroup, threads_per_group);
    const float inverse_rms = rsqrt(square_sum / float(parameters.row_width) + parameters.epsilon);
    const float correction =
        projected_gradient * inverse_rms * inverse_rms * inverse_rms / float(parameters.row_width);
    for (uint column = thread_index; column < parameters.row_width; column += threads_per_group) {
        const uint index = offset + column;
        input_gradient[index] =
            output_gradient[index] * weight[column] * inverse_rms - input[index] * correction;
        llm_atomic_add_float(weight_gradient + column,
                             output_gradient[index] * input[index] * inverse_rms);
    }
}

kernel void llm_rope_f32(device const float *input [[buffer(0)]],
                         device const float *cos_table [[buffer(1)]],
                         device const float *sin_table [[buffer(2)]],
                         device float *output [[buffer(3)]],
                         constant RopeParameters &parameters [[buffer(4)]],
                         uint pair_index [[thread_position_in_grid]]) {
    const uint pairs_per_position = parameters.head_count * parameters.pairs_per_head;
    const uint position_index = pair_index / pairs_per_position;
    const uint pair = (pair_index % pairs_per_position) % parameters.pairs_per_head;
    const uint sequence = position_index % parameters.sequence_length;
    const uint value_index = pair_index * 2;
    const uint table_index = sequence * parameters.pairs_per_head + pair;
    const float first = input[value_index];
    const float second = input[value_index + 1];
    const float cosine = cos_table[table_index];
    const float sine = sin_table[table_index];
    output[value_index] = first * cosine - second * sine;
    output[value_index + 1] = first * sine + second * cosine;
}

kernel void llm_rope_backward_f32(device const float *output_gradient [[buffer(0)]],
                                  device const float *cos_table [[buffer(1)]],
                                  device const float *sin_table [[buffer(2)]],
                                  device float *input_gradient [[buffer(3)]],
                                  constant RopeParameters &parameters [[buffer(4)]],
                                  uint pair_index [[thread_position_in_grid]]) {
    const uint pairs_per_position = parameters.head_count * parameters.pairs_per_head;
    const uint position_index = pair_index / pairs_per_position;
    const uint pair = (pair_index % pairs_per_position) % parameters.pairs_per_head;
    const uint sequence = position_index % parameters.sequence_length;
    const uint value_index = pair_index * 2;
    const uint table_index = sequence * parameters.pairs_per_head + pair;
    const float first = output_gradient[value_index];
    const float second = output_gradient[value_index + 1];
    const float cosine = cos_table[table_index];
    const float sine = sin_table[table_index];
    input_gradient[value_index] = first * cosine + second * sine;
    input_gradient[value_index + 1] = -first * sine + second * cosine;
}

kernel void llm_rope_position_f32(device const float *input [[buffer(0)]],
                                  device const float *cos_table [[buffer(1)]],
                                  device const float *sin_table [[buffer(2)]],
                                  device float *output [[buffer(3)]],
                                  constant RopePositionParameters &parameters [[buffer(4)]],
                                  uint pair_index [[thread_position_in_grid]]) {
    const uint pair = pair_index % parameters.pairs_per_head;
    const uint value_index = pair_index * 2;
    const uint table_index = parameters.position * parameters.pairs_per_head + pair;
    const float first = input[value_index];
    const float second = input[value_index + 1];
    const float cosine = cos_table[table_index];
    const float sine = sin_table[table_index];
    output[value_index] = first * cosine - second * sine;
    output[value_index + 1] = first * sine + second * cosine;
}

inline uint llm_attention_offset(uint batch, uint sequence, uint head, uint sequence_length,
                                 uint head_count, uint head_dimension) {
    return ((batch * sequence_length + sequence) * head_count + head) * head_dimension;
}

kernel void llm_attention_forward_f32(
    device const float *query [[buffer(0)]], device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]], device float *output [[buffer(3)]],
    constant AttentionParameters &parameters [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]], uint query_row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    const uint rows_per_batch = parameters.sequence_length * parameters.query_head_count;
    const uint batch = query_row / rows_per_batch;
    const uint within_batch = query_row % rows_per_batch;
    const uint query_position = within_batch / parameters.query_head_count;
    const uint query_head = within_batch % parameters.query_head_count;
    const uint heads_per_group = parameters.query_head_count / parameters.key_value_head_count;
    const uint key_value_head = query_head / heads_per_group;
    const uint query_index =
        llm_attention_offset(batch, query_position, query_head, parameters.sequence_length,
                             parameters.query_head_count, parameters.head_dimension);
    const device float *query_row_values = query + query_index;
    threadgroup float *partial = scratch;
    threadgroup float *accumulator = scratch + 32;
    for (uint dimension = thread_index; dimension < parameters.head_dimension;
         dimension += threads_per_group) {
        accumulator[dimension] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float running_maximum = -INFINITY;
    float running_sum = 0.0f;
    for (uint key_position = 0; key_position <= query_position; ++key_position) {
        const uint key_index =
            llm_attention_offset(batch, key_position, key_value_head, parameters.sequence_length,
                                 parameters.key_value_head_count, parameters.head_dimension);
        float dot = 0.0f;
        for (uint dimension = thread_index; dimension < parameters.head_dimension;
             dimension += threads_per_group) {
            dot += query_row_values[dimension] * key[key_index + dimension];
        }
        const float score =
            llm_threadgroup_sum(dot, partial, thread_index, lane, simdgroup, threads_per_group) *
            parameters.scale;
        const float new_maximum = max(running_maximum, score);
        const float previous_scale =
            running_sum == 0.0f ? 0.0f : exp(running_maximum - new_maximum);
        const float score_scale = exp(score - new_maximum);
        running_sum = running_sum * previous_scale + score_scale;
        for (uint dimension = thread_index; dimension < parameters.head_dimension;
             dimension += threads_per_group) {
            accumulator[dimension] = accumulator[dimension] * previous_scale +
                                     score_scale * value[key_index + dimension];
        }
        running_maximum = new_maximum;
    }
    for (uint dimension = thread_index; dimension < parameters.head_dimension;
         dimension += threads_per_group) {
        output[query_index + dimension] = accumulator[dimension] / running_sum;
    }
}

kernel void llm_attention_decode_f32(
    device const float *query [[buffer(0)]], device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]], device float *key_cache [[buffer(3)]],
    device float *value_cache [[buffer(4)]], device float *output [[buffer(5)]],
    constant AttentionDecodeParameters &parameters [[buffer(6)]],
    threadgroup float *scratch [[threadgroup(0)]], uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    const uint batch = row / parameters.head_count;
    const uint head = row % parameters.head_count;
    const uint current_index = row * parameters.head_dimension;
    const uint cache_current =
        llm_attention_offset(batch, parameters.position, head, parameters.cache_capacity,
                             parameters.head_count, parameters.head_dimension);
    for (uint dimension = thread_index; dimension < parameters.head_dimension;
         dimension += threads_per_group) {
        key_cache[cache_current + dimension] = key[current_index + dimension];
        value_cache[cache_current + dimension] = value[current_index + dimension];
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    threadgroup float *partial = scratch;
    threadgroup float *accumulator = scratch + 32;
    for (uint dimension = thread_index; dimension < parameters.head_dimension;
         dimension += threads_per_group) {
        accumulator[dimension] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float running_maximum = -INFINITY;
    float running_sum = 0.0f;
    for (uint key_position = 0; key_position <= parameters.position; ++key_position) {
        const uint cache_index =
            llm_attention_offset(batch, key_position, head, parameters.cache_capacity,
                                 parameters.head_count, parameters.head_dimension);
        float dot = 0.0f;
        for (uint dimension = thread_index; dimension < parameters.head_dimension;
             dimension += threads_per_group) {
            dot += query[current_index + dimension] * key_cache[cache_index + dimension];
        }
        const float score =
            llm_threadgroup_sum(dot, partial, thread_index, lane, simdgroup, threads_per_group) *
            parameters.scale;
        const float maximum = max(running_maximum, score);
        const float previous_scale = running_sum == 0.0f ? 0.0f : exp(running_maximum - maximum);
        const float score_scale = exp(score - maximum);
        running_sum = running_sum * previous_scale + score_scale;
        for (uint dimension = thread_index; dimension < parameters.head_dimension;
             dimension += threads_per_group) {
            accumulator[dimension] = accumulator[dimension] * previous_scale +
                                     score_scale * value_cache[cache_index + dimension];
        }
        running_maximum = maximum;
    }
    for (uint dimension = thread_index; dimension < parameters.head_dimension;
         dimension += threads_per_group) {
        output[current_index + dimension] = accumulator[dimension] / running_sum;
    }
}

kernel void llm_attention_backward_f32(
    device const float *query [[buffer(0)]], device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]], device const float *output_gradient [[buffer(3)]],
    device float *query_gradient [[buffer(4)]], device atomic_float *key_gradient [[buffer(5)]],
    device atomic_float *value_gradient [[buffer(6)]],
    constant AttentionParameters &parameters [[buffer(7)]],
    threadgroup float *scratch [[threadgroup(0)]], uint query_row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    const uint rows_per_batch = parameters.sequence_length * parameters.query_head_count;
    const uint batch = query_row / rows_per_batch;
    const uint within_batch = query_row % rows_per_batch;
    const uint query_position = within_batch / parameters.query_head_count;
    const uint query_head = within_batch % parameters.query_head_count;
    const uint heads_per_group = parameters.query_head_count / parameters.key_value_head_count;
    const uint key_value_head = query_head / heads_per_group;
    const uint query_index =
        llm_attention_offset(batch, query_position, query_head, parameters.sequence_length,
                             parameters.query_head_count, parameters.head_dimension);
    const device float *query_row_values = query + query_index;
    const device float *output_gradient_row = output_gradient + query_index;
    threadgroup float partial[32];
    threadgroup float *probabilities = scratch;
    threadgroup float *probability_gradients = scratch + parameters.sequence_length;
    for (uint key_position = 0; key_position <= query_position; ++key_position) {
        const uint key_index =
            llm_attention_offset(batch, key_position, key_value_head, parameters.sequence_length,
                                 parameters.key_value_head_count, parameters.head_dimension);
        float dot = 0.0f;
        float probability_gradient = 0.0f;
        for (uint dimension = thread_index; dimension < parameters.head_dimension;
             dimension += threads_per_group) {
            dot += query_row_values[dimension] * key[key_index + dimension];
            probability_gradient += output_gradient_row[dimension] * value[key_index + dimension];
        }
        dot = llm_threadgroup_sum(dot, partial, thread_index, lane, simdgroup, threads_per_group);
        probability_gradient = llm_threadgroup_sum(probability_gradient, partial, thread_index,
                                                   lane, simdgroup, threads_per_group);
        if (thread_index == 0) {
            probabilities[key_position] = dot * parameters.scale;
            probability_gradients[key_position] = probability_gradient;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /*
     * The softmax runs across the whole threadgroup instead of on thread zero.
     * Each thread owns the key positions congruent to its index, so the three
     * passes never share an element, and both reductions broadcast their result.
     */
    float maximum = -INFINITY;
    for (uint key_position = thread_index; key_position <= query_position;
         key_position += threads_per_group) {
        maximum = max(maximum, probabilities[key_position]);
    }
    maximum =
        llm_threadgroup_max(maximum, partial, thread_index, lane, simdgroup, threads_per_group);
    float denominator = 0.0f;
    for (uint key_position = thread_index; key_position <= query_position;
         key_position += threads_per_group) {
        const float probability = exp(probabilities[key_position] - maximum);
        probabilities[key_position] = probability;
        denominator += probability;
    }
    denominator =
        llm_threadgroup_sum(denominator, partial, thread_index, lane, simdgroup, threads_per_group);
    const float inverse_denominator = 1.0f / denominator;
    for (uint key_position = thread_index; key_position <= query_position;
         key_position += threads_per_group) {
        probabilities[key_position] *= inverse_denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float weighted_probability_gradient = 0.0f;
    for (uint key_position = thread_index; key_position <= query_position;
         key_position += threads_per_group) {
        weighted_probability_gradient +=
            probabilities[key_position] * probability_gradients[key_position];
    }
    weighted_probability_gradient = llm_threadgroup_sum(
        weighted_probability_gradient, partial, thread_index, lane, simdgroup, threads_per_group);
    for (uint dimension = thread_index; dimension < parameters.head_dimension;
         dimension += threads_per_group) {
        float query_value_gradient = 0.0f;
        for (uint key_position = 0; key_position <= query_position; ++key_position) {
            const uint key_index = llm_attention_offset(
                batch, key_position, key_value_head, parameters.sequence_length,
                parameters.key_value_head_count, parameters.head_dimension);
            const float score_gradient =
                probabilities[key_position] *
                (probability_gradients[key_position] - weighted_probability_gradient);
            query_value_gradient += parameters.scale * score_gradient * key[key_index + dimension];
            llm_atomic_add_float(key_gradient + key_index + dimension,
                                 parameters.scale * score_gradient * query_row_values[dimension]);
            llm_atomic_add_float(value_gradient + key_index + dimension,
                                 probabilities[key_position] * output_gradient_row[dimension]);
        }
        query_gradient[query_index + dimension] = query_value_gradient;
    }
}

kernel void llm_adamw_update_f32(device float *parameter [[buffer(0)]],
                                 device float *gradient [[buffer(1)]],
                                 device float *first_moment [[buffer(2)]],
                                 device float *second_moment [[buffer(3)]],
                                 constant AdamwParameters &options [[buffer(4)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= options.count) {
        return;
    }
    const float scaled_gradient = gradient[index] * options.gradient_scale;
    const float first =
        options.beta1 * first_moment[index] + (1.0f - options.beta1) * scaled_gradient;
    const float second = options.beta2 * second_moment[index] +
                         (1.0f - options.beta2) * scaled_gradient * scaled_gradient;
    const float corrected_first = first * options.inverse_first_bias;
    const float corrected_second = second * options.inverse_second_bias;
    parameter[index] -=
        options.learning_rate * (corrected_first / (sqrt(corrected_second) + options.epsilon) +
                                 options.weight_decay * parameter[index]);
    first_moment[index] = first;
    second_moment[index] = second;
    if (options.zero_gradient != 0) {
        gradient[index] = 0.0f;
    }
}

inline void llm_atomic_add_float(device atomic_uint *destination, float value) {
    uint expected = atomic_load_explicit(destination, memory_order_relaxed);
    uint desired = 0;
    do {
        desired = as_type<uint>(as_type<float>(expected) + value);
    } while (!atomic_compare_exchange_weak_explicit(destination, &expected, desired,
                                                    memory_order_relaxed, memory_order_relaxed));
}

/* Apple GPUs support native floating-point atomics. Attention updates K/V from
 * many causal query rows, so using the hardware operation avoids the heavily
 * contended compare-and-swap retry loop used by the generic fallback above. */
inline void llm_atomic_add_float(device atomic_float *destination, float value) {
    atomic_fetch_add_explicit(destination, value, memory_order_relaxed);
}

kernel void llm_reduce_sum_last_f32(device const float *input [[buffer(0)]],
                                    device float *output [[buffer(1)]],
                                    constant ReductionParameters &parameters [[buffer(2)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint thread_index [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simdgroup [[simdgroup_index_in_threadgroup]],
                                    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.outer_count) {
        return;
    }
    threadgroup float partial[32];
    float sum = 0.0f;
    const uint offset = row * parameters.reduction_size;
    for (uint column = thread_index; column < parameters.reduction_size;
         column += threads_per_group) {
        sum += input[offset + column];
    }
    sum = llm_threadgroup_sum(sum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (thread_index == 0) {
        output[row] = sum;
    }
}

kernel void llm_reduce_max_last_f32(device const float *input [[buffer(0)]],
                                    device float *output [[buffer(1)]],
                                    constant ReductionParameters &parameters [[buffer(2)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint thread_index [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simdgroup [[simdgroup_index_in_threadgroup]],
                                    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.outer_count) {
        return;
    }
    threadgroup float partial[32];
    const uint offset = row * parameters.reduction_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (uint column = thread_index; column < parameters.reduction_size;
         column += threads_per_group) {
        const float value = input[offset + column];
        if (!isfinite(value)) {
            invalid = 1.0f;
        } else {
            maximum = max(maximum, value);
        }
    }
    invalid =
        llm_threadgroup_sum(invalid, partial, thread_index, lane, simdgroup, threads_per_group);
    maximum =
        llm_threadgroup_max(maximum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (thread_index == 0) {
        output[row] = invalid > 0.0f ? as_type<float>(0x7fc00000u) : maximum;
    }
}

kernel void llm_reduce_mean_square_last_f32(device const float *input [[buffer(0)]],
                                            device float *output [[buffer(1)]],
                                            constant ReductionParameters &parameters [[buffer(2)]],
                                            uint row [[threadgroup_position_in_grid]],
                                            uint thread_index [[thread_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.outer_count) {
        return;
    }
    threadgroup float partial[32];
    float sum = 0.0f;
    const uint offset = row * parameters.reduction_size;
    for (uint column = thread_index; column < parameters.reduction_size;
         column += threads_per_group) {
        const float value = input[offset + column];
        sum += value * value;
    }
    sum = llm_threadgroup_sum(sum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (thread_index == 0) {
        output[row] = sum / float(parameters.reduction_size);
    }
}

kernel void llm_accumulate_sum_squares_f32(device const float *input [[buffer(0)]],
                                           device atomic_float *accumulator [[buffer(1)]],
                                           constant ElementwiseParameters &parameters [[buffer(2)]],
                                           uint group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint simdgroup [[simdgroup_index_in_threadgroup]],
                                           uint threads_per_group [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    const uint begin = group * 4096u;
    const uint end = min(begin + 4096u, parameters.count);
    float sum = 0.0f;
    for (uint index = begin + thread_index; index < end; index += threads_per_group) {
        const float value = input[index];
        sum += value * value;
    }
    sum = llm_threadgroup_sum(sum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (thread_index == 0) {
        llm_atomic_add_float(accumulator, sum);
    }
}

constant uint llm_matmul_tile = 16;

kernel void llm_matmul_f32(device const float *left [[buffer(0)]],
                           device const float *right [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant MatmulParameters &parameters [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint2 local [[thread_position_in_threadgroup]]) {
    threadgroup float left_tile[16][16];
    threadgroup float right_tile[16][16];
    const uint row = group.y * llm_matmul_tile + local.y;
    const uint column = group.x * llm_matmul_tile + local.x;
    float sum = 0.0f;

    for (uint tile = 0; tile < parameters.inner_size; tile += llm_matmul_tile) {
        const uint left_column = tile + local.x;
        const uint right_row = tile + local.y;
        left_tile[local.y][local.x] = row < parameters.rows && left_column < parameters.inner_size
                                          ? left[row * parameters.inner_size + left_column]
                                          : 0.0f;
        right_tile[local.y][local.x] =
            right_row < parameters.inner_size && column < parameters.columns
                ? right[right_row * parameters.columns + column]
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < llm_matmul_tile; ++inner) {
            sum += left_tile[local.y][inner] * right_tile[inner][local.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < parameters.rows && column < parameters.columns) {
        output[row * parameters.columns + column] = sum;
    }
}

kernel void llm_matmul_f32_tiled32(device const float *left [[buffer(0)]],
                                   device const float *right [[buffer(1)]],
                                   device float *output [[buffer(2)]],
                                   constant MatmulParameters &parameters [[buffer(3)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint2 local [[thread_position_in_threadgroup]]) {
    threadgroup float left_tile[32][32];
    threadgroup float right_tile[32][32];
    const uint row0 = group.y * 32 + local.y;
    const uint row1 = row0 + 16;
    const uint column0 = group.x * 32 + local.x;
    const uint column1 = column0 + 16;
    float4 sums = 0.0f;

    for (uint tile = 0; tile < parameters.inner_size; tile += 32) {
        const uint inner0 = tile + local.x;
        const uint inner1 = inner0 + 16;
        const uint right_row0 = tile + local.y;
        const uint right_row1 = right_row0 + 16;
        left_tile[local.y][local.x] = row0 < parameters.rows && inner0 < parameters.inner_size
                                          ? left[row0 * parameters.inner_size + inner0]
                                          : 0.0f;
        left_tile[local.y][local.x + 16] = row0 < parameters.rows && inner1 < parameters.inner_size
                                               ? left[row0 * parameters.inner_size + inner1]
                                               : 0.0f;
        left_tile[local.y + 16][local.x] = row1 < parameters.rows && inner0 < parameters.inner_size
                                               ? left[row1 * parameters.inner_size + inner0]
                                               : 0.0f;
        left_tile[local.y + 16][local.x + 16] =
            row1 < parameters.rows && inner1 < parameters.inner_size
                ? left[row1 * parameters.inner_size + inner1]
                : 0.0f;
        right_tile[local.y][local.x] =
            right_row0 < parameters.inner_size && column0 < parameters.columns
                ? right[right_row0 * parameters.columns + column0]
                : 0.0f;
        right_tile[local.y][local.x + 16] =
            right_row0 < parameters.inner_size && column1 < parameters.columns
                ? right[right_row0 * parameters.columns + column1]
                : 0.0f;
        right_tile[local.y + 16][local.x] =
            right_row1 < parameters.inner_size && column0 < parameters.columns
                ? right[right_row1 * parameters.columns + column0]
                : 0.0f;
        right_tile[local.y + 16][local.x + 16] =
            right_row1 < parameters.inner_size && column1 < parameters.columns
                ? right[right_row1 * parameters.columns + column1]
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < 32; ++inner) {
            const float2 left_values =
                float2(left_tile[local.y][inner], left_tile[local.y + 16][inner]);
            const float2 right_values =
                float2(right_tile[inner][local.x], right_tile[inner][local.x + 16]);
            sums += float4(left_values.x * right_values.x, left_values.x * right_values.y,
                           left_values.y * right_values.x, left_values.y * right_values.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < parameters.rows && column0 < parameters.columns) {
        output[row0 * parameters.columns + column0] = sums.x;
    }
    if (row0 < parameters.rows && column1 < parameters.columns) {
        output[row0 * parameters.columns + column1] = sums.y;
    }
    if (row1 < parameters.rows && column0 < parameters.columns) {
        output[row1 * parameters.columns + column0] = sums.z;
    }
    if (row1 < parameters.rows && column1 < parameters.columns) {
        output[row1 * parameters.columns + column1] = sums.w;
    }
}

kernel void llm_gather_rows_f32(device const float *table [[buffer(0)]],
                                device const uint *indices [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant GatherParameters &parameters [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    const uint count = parameters.index_count * parameters.row_width;
    if (index < count) {
        const uint source_row = indices[index / parameters.row_width];
        const uint column = index % parameters.row_width;
        output[index] = table[source_row * parameters.row_width + column];
    }
}

kernel void llm_scatter_add_rows_f32(device const float *source [[buffer(0)]],
                                     device const uint *indices [[buffer(1)]],
                                     device atomic_uint *table [[buffer(2)]],
                                     constant GatherParameters &parameters [[buffer(3)]],
                                     uint index [[thread_position_in_grid]]) {
    const uint count = parameters.index_count * parameters.row_width;
    if (index >= count) {
        return;
    }
    const uint source_row = index / parameters.row_width;
    const uint column = index % parameters.row_width;
    const uint destination = indices[source_row] * parameters.row_width + column;
    uint expected = atomic_load_explicit(&table[destination], memory_order_relaxed);
    uint desired = 0;
    do {
        desired = as_type<uint>(as_type<float>(expected) + source[index]);
    } while (!atomic_compare_exchange_weak_explicit(&table[destination], &expected, desired,
                                                    memory_order_relaxed, memory_order_relaxed));
}

kernel void llm_softmax_last_f32(device const float *input [[buffer(0)]],
                                 device float *output [[buffer(1)]],
                                 constant ReductionParameters &parameters [[buffer(2)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint thread_index [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup [[simdgroup_index_in_threadgroup]],
                                 uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.outer_count) {
        return;
    }
    threadgroup float partial[32];
    const uint offset = row * parameters.reduction_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (uint column = thread_index; column < parameters.reduction_size;
         column += threads_per_group) {
        const float value = input[offset + column];
        if (!isfinite(value)) {
            invalid = 1.0f;
        } else {
            maximum = max(maximum, value);
        }
    }
    invalid =
        llm_threadgroup_sum(invalid, partial, thread_index, lane, simdgroup, threads_per_group);
    maximum =
        llm_threadgroup_max(maximum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (invalid > 0.0f) {
        if (thread_index == 0) {
            output[offset] = as_type<float>(0x7fc00000u);
        }
        return;
    }
    float sum = 0.0f;
    for (uint column = thread_index; column < parameters.reduction_size;
         column += threads_per_group) {
        const float probability = exp(input[offset + column] - maximum);
        output[offset + column] = probability;
        sum += probability;
    }
    sum = llm_threadgroup_sum(sum, partial, thread_index, lane, simdgroup, threads_per_group);
    const float inverse_sum = 1.0f / sum;
    for (uint column = thread_index; column < parameters.reduction_size;
         column += threads_per_group) {
        output[offset + column] *= inverse_sum;
    }
}

kernel void llm_cross_entropy_forward_f32(
    device const float *logits [[buffer(0)]], device const uint *targets [[buffer(1)]],
    device const uint *loss_mask [[buffer(2)]], device atomic_uint *loss [[buffer(3)]],
    constant CrossEntropyParameters &parameters [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.row_count) {
        return;
    }
    if (parameters.use_loss_mask != 0 && loss_mask[row] == 0) {
        return;
    }
    threadgroup float partial[32];
    const uint offset = row * parameters.vocabulary_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (uint column = thread_index; column < parameters.vocabulary_size;
         column += threads_per_group) {
        const float value = logits[offset + column];
        if (!isfinite(value)) {
            invalid = 1.0f;
        } else {
            maximum = max(maximum, value);
        }
    }
    invalid =
        llm_threadgroup_sum(invalid, partial, thread_index, lane, simdgroup, threads_per_group);
    maximum =
        llm_threadgroup_max(maximum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (invalid > 0.0f) {
        if (thread_index == 0) {
            atomic_store_explicit(loss, 0x7fc00000u, memory_order_relaxed);
        }
        return;
    }
    float sum = 0.0f;
    for (uint column = thread_index; column < parameters.vocabulary_size;
         column += threads_per_group) {
        sum += exp(logits[offset + column] - maximum);
    }
    sum = llm_threadgroup_sum(sum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (thread_index == 0) {
        const float row_loss = maximum + log(sum) - logits[offset + targets[row]];
        llm_atomic_add_float(loss, row_loss / float(parameters.normalization_row_count));
    }
}

kernel void llm_cross_entropy_backward_f32(
    device const float *logits [[buffer(0)]], device const uint *targets [[buffer(1)]],
    device const uint *loss_mask [[buffer(2)]], device float *gradient [[buffer(3)]],
    constant CrossEntropyParameters &parameters [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.row_count) {
        return;
    }
    if (parameters.use_loss_mask != 0 && loss_mask[row] == 0) {
        const uint ignored_offset = row * parameters.vocabulary_size;
        for (uint column = thread_index; column < parameters.vocabulary_size;
             column += threads_per_group) {
            gradient[ignored_offset + column] = 0.0f;
        }
        return;
    }
    threadgroup float partial[32];
    const uint offset = row * parameters.vocabulary_size;
    float maximum = -INFINITY;
    float invalid = 0.0f;
    for (uint column = thread_index; column < parameters.vocabulary_size;
         column += threads_per_group) {
        const float value = logits[offset + column];
        if (!isfinite(value)) {
            invalid = 1.0f;
        } else {
            maximum = max(maximum, value);
        }
    }
    invalid =
        llm_threadgroup_sum(invalid, partial, thread_index, lane, simdgroup, threads_per_group);
    maximum =
        llm_threadgroup_max(maximum, partial, thread_index, lane, simdgroup, threads_per_group);
    if (invalid > 0.0f) {
        if (thread_index == 0) {
            gradient[offset] = as_type<float>(0x7fc00000u);
        }
        return;
    }
    float sum = 0.0f;
    for (uint column = thread_index; column < parameters.vocabulary_size;
         column += threads_per_group) {
        sum += exp(logits[offset + column] - maximum);
    }
    sum = llm_threadgroup_sum(sum, partial, thread_index, lane, simdgroup, threads_per_group);
    const float scale = 1.0f / (sum * float(parameters.normalization_row_count));
    for (uint column = thread_index; column < parameters.vocabulary_size;
         column += threads_per_group) {
        float value = exp(logits[offset + column] - maximum) * scale;
        if (column == targets[row]) {
            value -= 1.0f / float(parameters.normalization_row_count);
        }
        gradient[offset + column] = value;
    }
}
