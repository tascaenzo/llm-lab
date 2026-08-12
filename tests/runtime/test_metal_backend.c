#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/backend.h"
#include "runtime/operations.h"
#include "test_support.h"

static int close_enough(float left, float right) { return fabsf(left - right) < 2.0e-5F; }

static int test_memory_and_elementwise(llm_backend *backend) {
    const size_t shape[] = {2U, 3U};
    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_device(&left) == LLM_DEVICE_METAL);

    const float left_values[] = {1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F};
    const float right_values[] = {6.0F, 5.0F, -4.0F, 3.0F, 2.0F, -1.0F};
    float actual[6] = {0};
    TEST_ASSERT(llm_tensor_write(backend, &left, left_values, sizeof(left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &right, right_values, sizeof(right_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_copy(backend, &left, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(memcmp(actual, left_values, sizeof(actual)) == 0);

    TEST_ASSERT(llm_add(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    const float expected_add[] = {7.0F, 3.0F, -1.0F, 7.0F, -3.0F, 5.0F};
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(actual[index] == expected_add[index]);
    }

    TEST_ASSERT(llm_multiply(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(actual[0] == 6.0F && actual[1] == -10.0F && actual[5] == -6.0F);

    TEST_ASSERT(llm_scale(backend, &left, -0.5F, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(actual[0] == -0.5F && actual[1] == 1.0F && actual[5] == -3.0F);

    TEST_ASSERT(llm_tensor_fill_f32(backend, &output, 2.5F) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(actual[index] == 2.5F);
    }
    TEST_ASSERT(llm_tensor_zero(backend, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(actual[index] == 0.0F);
    }

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    return EXIT_SUCCESS;
}

static int test_reshape_shared_storage(llm_backend *backend) {
    llm_metal_backend_metrics baseline = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &baseline) == LLM_OK);

    const size_t source_shape[] = {2U, 3U};
    const size_t view_shape[] = {6U};
    llm_tensor source = {0};
    llm_tensor view = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, source_shape, &source) == LLM_OK);

    llm_metal_backend_metrics after_create = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &after_create) == LLM_OK);
    TEST_ASSERT(after_create.active_buffer_count == baseline.active_buffer_count + 1U);

    TEST_ASSERT(llm_tensor_reshape(&source, 1U, view_shape, &view) == LLM_OK);
    llm_metal_backend_metrics after_reshape = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &after_reshape) == LLM_OK);
    TEST_ASSERT(after_reshape.active_buffer_count == after_create.active_buffer_count);

    const float expected[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    TEST_ASSERT(llm_tensor_write(backend, &source, expected, sizeof(expected)) == LLM_OK);
    llm_tensor_destroy(&source);
    llm_metal_backend_metrics after_source_destroy = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &after_source_destroy) == LLM_OK);
    TEST_ASSERT(after_source_destroy.active_buffer_count == after_create.active_buffer_count);

    float actual[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &view, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(memcmp(expected, actual, sizeof(expected)) == 0);

    llm_tensor_destroy(&view);
    llm_metal_backend_metrics after_view_destroy = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &after_view_destroy) == LLM_OK);
    TEST_ASSERT(after_view_destroy.active_buffer_count == baseline.active_buffer_count);
    return EXIT_SUCCESS;
}

static int test_batch_metrics_and_buffer_pool(llm_backend *backend) {
    TEST_ASSERT(llm_backend_metal_reset_metrics(backend) == LLM_OK);
    const size_t shape[] = {1024U};
    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor temporary = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &temporary) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &left, 2.0F) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &right, 4.0F) == LLM_OK);

    TEST_ASSERT(llm_backend_metal_begin_batch(backend) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_begin_batch(backend) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_add(backend, &left, &right, &temporary) == LLM_OK);
    TEST_ASSERT(llm_scale(backend, &temporary, 0.5F, &output) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_end_batch(backend) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_end_batch(backend) == LLM_INVALID_ARGUMENT);
    float values[1024] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, values, sizeof(values)) == LLM_OK);
    for (size_t index = 0U; index < 1024U; ++index) {
        TEST_ASSERT(values[index] == 3.0F);
    }

    llm_metal_backend_metrics metrics = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &metrics) == LLM_OK);
    TEST_ASSERT(metrics.submitted_command_buffers == 3U);
    TEST_ASSERT(metrics.kernel_dispatches == 4U);
    TEST_ASSERT(metrics.total_gpu_seconds > 0.0);
    TEST_ASSERT(metrics.active_buffer_count >= 4U);

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&temporary);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &metrics) == LLM_OK);
    TEST_ASSERT(metrics.cached_buffer_count >= 4U);

    llm_tensor reused = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &reused) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &metrics) == LLM_OK);
    TEST_ASSERT(metrics.reused_buffer_allocations >= 1U);
    llm_tensor_destroy(&reused);
    return EXIT_SUCCESS;
}

static int test_reductions_and_matmul(llm_backend *backend) {
    const size_t input_shape[] = {2U, 3U};
    const size_t reduction_shape[] = {2U};
    llm_tensor input = {0};
    llm_tensor reduction = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, input_shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, reduction_shape, &reduction) ==
                LLM_OK);
    const float values[] = {-1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F};
    float reduced[2] = {0};
    TEST_ASSERT(llm_tensor_write(backend, &input, values, sizeof(values)) == LLM_OK);
    TEST_ASSERT(llm_reduce_sum_last(backend, &input, &reduction) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &reduction, reduced, sizeof(reduced)) == LLM_OK);
    TEST_ASSERT(reduced[0] == -6.0F && reduced[1] == -15.0F);
    TEST_ASSERT(llm_reduce_max_last(backend, &input, &reduction) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &reduction, reduced, sizeof(reduced)) == LLM_OK);
    TEST_ASSERT(reduced[0] == -1.0F && reduced[1] == -4.0F);
    TEST_ASSERT(llm_reduce_mean_square_last(backend, &input, &reduction) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &reduction, reduced, sizeof(reduced)) == LLM_OK);
    TEST_ASSERT(close_enough(reduced[0], 14.0F / 3.0F));
    TEST_ASSERT(close_enough(reduced[1], 77.0F / 3.0F));

    const size_t left_shape[] = {2U, 3U};
    const size_t right_shape[] = {3U, 2U};
    const size_t output_shape[] = {2U, 2U};
    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, left_shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, right_shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape, &output) == LLM_OK);
    const float left_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float right_values[] = {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    TEST_ASSERT(llm_tensor_write(backend, &left, left_values, sizeof(left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &right, right_values, sizeof(right_values)) == LLM_OK);
    TEST_ASSERT(llm_matmul(backend, &left, &right, &output) == LLM_OK);
    float product[4] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, product, sizeof(product)) == LLM_OK);
    const float expected[] = {58.0F, 64.0F, 139.0F, 154.0F};
    for (size_t index = 0U; index < 4U; ++index) {
        TEST_ASSERT(close_enough(product[index], expected[index]));
    }

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    llm_tensor_destroy(&reduction);
    llm_tensor_destroy(&input);
    return EXIT_SUCCESS;
}

static int test_matmul_tile_boundaries(llm_backend *metal_backend) {
    enum {
        ROWS = 67,
        INNER_SIZE = 71,
        COLUMNS = 65,
    };
    float left_values[ROWS * INNER_SIZE] = {0};
    float right_values[INNER_SIZE * COLUMNS] = {0};
    for (size_t index = 0U; index < ROWS * INNER_SIZE; ++index) {
        left_values[index] = (float)((int)(index % 11U) - 5) * 0.125F;
    }
    for (size_t index = 0U; index < INNER_SIZE * COLUMNS; ++index) {
        right_values[index] = (float)((int)(index % 7U) - 3) * 0.25F;
    }

    llm_backend *cpu_backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&cpu_backend) == LLM_OK);
    const size_t left_shape[] = {ROWS, INNER_SIZE};
    const size_t right_shape[] = {INNER_SIZE, COLUMNS};
    const size_t output_shape[] = {ROWS, COLUMNS};
    llm_tensor metal_left = {0};
    llm_tensor metal_right = {0};
    llm_tensor metal_output = {0};
    llm_tensor cpu_left = {0};
    llm_tensor cpu_right = {0};
    llm_tensor cpu_output = {0};
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, left_shape, &metal_left) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, right_shape, &metal_right) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, output_shape, &metal_output) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, left_shape, &cpu_left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, right_shape, &cpu_right) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, output_shape, &cpu_output) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_left, left_values, sizeof(left_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_right, right_values, sizeof(right_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_left, left_values, sizeof(left_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_right, right_values, sizeof(right_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_matmul(metal_backend, &metal_left, &metal_right, &metal_output) == LLM_OK);
    TEST_ASSERT(llm_matmul(cpu_backend, &cpu_left, &cpu_right, &cpu_output) == LLM_OK);

    float metal_values[ROWS * COLUMNS] = {0};
    float cpu_values[ROWS * COLUMNS] = {0};
    TEST_ASSERT(llm_tensor_read(metal_backend, &metal_output, metal_values, sizeof(metal_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_read(cpu_backend, &cpu_output, cpu_values, sizeof(cpu_values)) ==
                LLM_OK);
    for (size_t index = 0U; index < ROWS * COLUMNS; ++index) {
        const float tolerance = 1.0e-4F * fmaxf(1.0F, fabsf(cpu_values[index]));
        TEST_ASSERT(fabsf(metal_values[index] - cpu_values[index]) <= tolerance);
    }

    llm_tensor_destroy(&cpu_output);
    llm_tensor_destroy(&cpu_right);
    llm_tensor_destroy(&cpu_left);
    llm_tensor_destroy(&metal_output);
    llm_tensor_destroy(&metal_right);
    llm_tensor_destroy(&metal_left);
    llm_backend_destroy(cpu_backend);
    return EXIT_SUCCESS;
}

static int test_mps_matmul_ex_in_batch(llm_backend *backend) {
    const size_t left_shape[] = {3U, 2U};
    const size_t right_shape[] = {4U, 3U};
    const size_t output_shape[] = {2U, 4U};
    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor output = {0};
    llm_tensor source = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, left_shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, right_shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape, &source) == LLM_OK);
    const float left_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float right_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                  7.0F, 8.0F, 9.0F, 1.0F, 0.0F, 1.0F};
    TEST_ASSERT(llm_tensor_write(backend, &left, left_values, sizeof(left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &right, right_values, sizeof(right_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &source, 0.5F) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_reset_metrics(backend) == LLM_OK);

    const llm_matmul_options options = {.transpose_left = 1, .transpose_right = 1};
    TEST_ASSERT(llm_backend_metal_begin_batch(backend) == LLM_OK);
    TEST_ASSERT(llm_matmul_ex(backend, &left, &right, &options, &output) == LLM_OK);
    TEST_ASSERT(llm_accumulate(backend, &source, &output) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_end_batch(backend) == LLM_OK);

    float actual[8] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 4U; ++column) {
            float expected = 0.5F;
            for (size_t inner = 0U; inner < 3U; ++inner) {
                expected += left_values[inner * 2U + row] * right_values[column * 3U + inner];
            }
            TEST_ASSERT(close_enough(actual[row * 4U + column], expected));
        }
    }
    llm_metal_backend_metrics metrics = {0};
    TEST_ASSERT(llm_backend_metal_get_metrics(backend, &metrics) == LLM_OK);
    TEST_ASSERT(metrics.submitted_command_buffers == 1U);
    TEST_ASSERT(metrics.kernel_dispatches == 2U);

    llm_tensor_destroy(&source);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    return EXIT_SUCCESS;
}

static int test_language_operations(llm_backend *backend) {
    const size_t table_shape[] = {4U, 3U};
    const size_t indices_shape[] = {3U};
    const size_t selected_shape[] = {3U, 3U};
    llm_tensor table = {0};
    llm_tensor indices = {0};
    llm_tensor selected = {0};
    llm_tensor accumulated = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &table) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, indices_shape, &indices) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, selected_shape, &selected) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &accumulated) == LLM_OK);
    const float table_values[] = {0.0F,  1.0F,  2.0F,  10.0F, 11.0F, 12.0F,
                                  20.0F, 21.0F, 22.0F, 30.0F, 31.0F, 32.0F};
    const uint32_t index_values[] = {2U, 0U, 2U};
    TEST_ASSERT(llm_tensor_write(backend, &table, table_values, sizeof(table_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &indices, index_values, sizeof(index_values)) == LLM_OK);
    TEST_ASSERT(llm_gather_rows(backend, &table, &indices, &selected) == LLM_OK);
    float gathered[9] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &selected, gathered, sizeof(gathered)) == LLM_OK);
    TEST_ASSERT(gathered[0] == 20.0F && gathered[3] == 0.0F && gathered[8] == 22.0F);
    TEST_ASSERT(llm_tensor_zero(backend, &accumulated) == LLM_OK);
    TEST_ASSERT(llm_scatter_add_rows(backend, &selected, &indices, &accumulated) == LLM_OK);
    float scattered[12] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &accumulated, scattered, sizeof(scattered)) == LLM_OK);
    TEST_ASSERT(scattered[1] == 1.0F && scattered[6] == 40.0F && scattered[8] == 44.0F);

    const uint32_t invalid_indices[] = {2U, 4U, 0U};
    TEST_ASSERT(llm_tensor_write(backend, &indices, invalid_indices, sizeof(invalid_indices)) ==
                LLM_OK);
    TEST_ASSERT(llm_gather_rows(backend, &table, &indices, &selected) == LLM_INVALID_INDEX);

    const size_t logits_shape[] = {2U, 3U};
    const size_t targets_shape[] = {2U};
    llm_tensor logits = {0};
    llm_tensor probabilities = {0};
    llm_tensor targets = {0};
    llm_tensor loss = {0};
    llm_tensor gradient = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &probabilities) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, targets_shape, &targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &gradient) == LLM_OK);
    const float logits_values[] = {2.0F, 1.0F, 0.0F, 0.0F, 1.0F, 2.0F};
    const uint32_t target_values[] = {0U, 2U};
    TEST_ASSERT(llm_tensor_write(backend, &logits, logits_values, sizeof(logits_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &targets, target_values, sizeof(target_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_softmax_last(backend, &logits, &probabilities) == LLM_OK);
    float probability_values[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &probabilities, probability_values,
                                sizeof(probability_values)) == LLM_OK);
    TEST_ASSERT(
        close_enough(probability_values[0] + probability_values[1] + probability_values[2], 1.0F));
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
    float loss_value = 0.0F;
    TEST_ASSERT(llm_tensor_read(backend, &loss, &loss_value, sizeof(loss_value)) == LLM_OK);
    TEST_ASSERT(close_enough(loss_value, 0.40760595F));
    TEST_ASSERT(llm_cross_entropy_backward(backend, &logits, &targets, &gradient) == LLM_OK);
    float gradient_values[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &gradient, gradient_values, sizeof(gradient_values)) ==
                LLM_OK);
    TEST_ASSERT(close_enough(gradient_values[0], -0.16737952F));
    TEST_ASSERT(close_enough(gradient_values[5], -0.16737952F));

    const float invalid_logits[] = {NAN, 0.0F, 1.0F, 1.0F, 2.0F, 3.0F};
    TEST_ASSERT(llm_tensor_write(backend, &logits, invalid_logits, sizeof(invalid_logits)) ==
                LLM_OK);
    TEST_ASSERT(llm_softmax_last(backend, &logits, &probabilities) == LLM_NUMERICAL_ERROR);

    llm_tensor_destroy(&gradient);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&probabilities);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&accumulated);
    llm_tensor_destroy(&selected);
    llm_tensor_destroy(&indices);
    llm_tensor_destroy(&table);
    return EXIT_SUCCESS;
}

int main(void) {
    TEST_ASSERT(llm_backend_metal_create(NULL) == LLM_INVALID_ARGUMENT);

    llm_backend *backend = NULL;
    if (llm_backend_metal_is_available() == 0) {
        TEST_ASSERT(llm_backend_metal_create(&backend) == LLM_UNSUPPORTED_DEVICE);
        TEST_ASSERT(backend == NULL);
        printf("Metal device unavailable: backend execution skipped\n");
        return EXIT_SUCCESS;
    }

    TEST_ASSERT(llm_backend_metal_create(&backend) == LLM_OK);
    TEST_ASSERT(backend != NULL);
    TEST_ASSERT(llm_backend_device(backend) == LLM_DEVICE_METAL);
    TEST_ASSERT(llm_backend_metal_device_name(backend) != NULL);
    TEST_ASSERT(llm_backend_synchronize(backend) == LLM_OK);

    llm_backend *cpu = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&cpu) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_device_name(cpu) == NULL);
    llm_backend_destroy(cpu);

    const int result = test_memory_and_elementwise(backend) == EXIT_SUCCESS &&
                               test_reshape_shared_storage(backend) == EXIT_SUCCESS &&
                               test_batch_metrics_and_buffer_pool(backend) == EXIT_SUCCESS &&
                               test_reductions_and_matmul(backend) == EXIT_SUCCESS &&
                               test_matmul_tile_boundaries(backend) == EXIT_SUCCESS &&
                               test_mps_matmul_ex_in_batch(backend) == EXIT_SUCCESS &&
                               test_language_operations(backend) == EXIT_SUCCESS
                           ? EXIT_SUCCESS
                           : EXIT_FAILURE;
    llm_backend_destroy(backend);
    return result;
}
