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
};

constant uint llm_simd_width = 32;

inline float llm_threadgroup_sum(float value, threadgroup float *partial, uint thread_index,
                                 uint lane, uint simdgroup, uint threads_per_group) {
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
    return partial[0];
}

inline float llm_threadgroup_max(float value, threadgroup float *partial, uint thread_index,
                                 uint lane, uint simdgroup, uint threads_per_group) {
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
    return partial[0];
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

inline void llm_atomic_add_float(device atomic_uint *destination, float value) {
    uint expected = atomic_load_explicit(destination, memory_order_relaxed);
    uint desired = 0;
    do {
        desired = as_type<uint>(as_type<float>(expected) + value);
    } while (!atomic_compare_exchange_weak_explicit(destination, &expected, desired,
                                                    memory_order_relaxed, memory_order_relaxed));
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

kernel void llm_cross_entropy_forward_f32(device const float *logits [[buffer(0)]],
                                          device const uint *targets [[buffer(1)]],
                                          device atomic_uint *loss [[buffer(2)]],
                                          constant CrossEntropyParameters &parameters [[buffer(3)]],
                                          uint row [[threadgroup_position_in_grid]],
                                          uint thread_index [[thread_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint simdgroup [[simdgroup_index_in_threadgroup]],
                                          uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.row_count) {
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
        llm_atomic_add_float(loss, row_loss / float(parameters.row_count));
    }
}

kernel void llm_cross_entropy_backward_f32(
    device const float *logits [[buffer(0)]], device const uint *targets [[buffer(1)]],
    device float *gradient [[buffer(2)]], constant CrossEntropyParameters &parameters [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    if (row >= parameters.row_count) {
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
    const float scale = 1.0f / (sum * float(parameters.row_count));
    for (uint column = thread_index; column < parameters.vocabulary_size;
         column += threads_per_group) {
        float value = exp(logits[offset + column] - maximum) * scale;
        if (column == targets[row]) {
            value -= 1.0f / float(parameters.row_count);
        }
        gradient[offset + column] = value;
    }
}
