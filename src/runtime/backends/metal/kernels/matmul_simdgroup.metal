#include <metal_stdlib>

using namespace metal;

struct MatmulParameters {
    uint rows;
    uint inner_size;
    uint columns;
};

kernel void llm_matmul_f32_simdgroup(device const float *left [[buffer(0)]],
                                     device const float *right [[buffer(1)]],
                                     device float *output [[buffer(2)]],
                                     constant MatmulParameters &parameters [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]],
                                     uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left_tile[16][16];
    threadgroup float right_tile[16][16];
    simdgroup_float8x8 left_matrix;
    simdgroup_float8x8 right_matrix;
    simdgroup_float8x8 result_matrix(0.0f);
    const uint row_base = group.y * 16;
    const uint column_base = group.x * 16;
    const uint result_row = (simdgroup / 2) * 8;
    const uint result_column = (simdgroup % 2) * 8;

    for (uint tile = 0; tile < parameters.inner_size; tile += 16) {
        for (uint index = thread_index; index < 256; index += 128) {
            const uint row = index / 16;
            const uint column = index % 16;
            left_tile[row][column] = left[(row_base + row) * parameters.inner_size + tile + column];
            right_tile[row][column] =
                right[(tile + row) * parameters.columns + column_base + column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(left_matrix, &left_tile[result_row][0], 16);
        simdgroup_load(right_matrix, &right_tile[0][result_column], 16);
        simdgroup_multiply_accumulate(result_matrix, left_matrix, right_matrix, result_matrix);
        simdgroup_load(left_matrix, &left_tile[result_row][8], 16);
        simdgroup_load(right_matrix, &right_tile[8][result_column], 16);
        simdgroup_multiply_accumulate(result_matrix, left_matrix, right_matrix, result_matrix);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(result_matrix,
                    output + (row_base + result_row) * parameters.columns + column_base +
                        result_column,
                    parameters.columns);
}
