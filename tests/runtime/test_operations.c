#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "runtime/backend.h"
#include "runtime/operations.h"
#include "test_support.h"

static int close_enough(float left, float right) { return fabsf(left - right) < 1.0e-5F; }

static int test_elementwise_operations(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const size_t shape[] = {2U, 3U};
    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &output) == LLM_OK);

    const float left_values[] = {1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F};
    const float right_values[] = {6.0F, 5.0F, -4.0F, 3.0F, 2.0F, -1.0F};
    float actual[6] = {0};
    TEST_ASSERT(llm_tensor_write(backend, &left, left_values, sizeof(left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &right, right_values, sizeof(right_values)) == LLM_OK);

    TEST_ASSERT(llm_add(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    const float expected_add[] = {7.0F, 3.0F, -1.0F, 7.0F, -3.0F, 5.0F};
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(actual[index] == expected_add[index]);
    }

    TEST_ASSERT(llm_multiply(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    const float expected_multiply[] = {6.0F, -10.0F, -12.0F, 12.0F, -10.0F, -6.0F};
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(actual[index] == expected_multiply[index]);
    }

    TEST_ASSERT(llm_multiply(backend, &left, &left, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(actual[0] == 1.0F && actual[1] == 4.0F && actual[5] == 36.0F);

    TEST_ASSERT(llm_scale(backend, &left, -0.5F, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(actual[0] == -0.5F && actual[1] == 1.0F && actual[5] == -3.0F);

    TEST_ASSERT(llm_add(backend, &left, &right, &left) == LLM_INVALID_ARGUMENT);

    const size_t different_shape[] = {3U, 2U};
    llm_tensor different = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, different_shape, &different) ==
                LLM_OK);
    TEST_ASSERT(llm_add(backend, &left, &right, &different) == LLM_INVALID_SHAPE);

    llm_tensor ids = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 2U, shape, &ids) == LLM_OK);
    TEST_ASSERT(llm_add(backend, &ids, &ids, &output) == LLM_UNSUPPORTED_DTYPE);

    llm_tensor_destroy(&ids);
    llm_tensor_destroy(&different);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_reductions(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const size_t input_shape[] = {2U, 3U};
    const size_t output_shape[] = {2U};
    llm_tensor input = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, input_shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, output_shape, &output) == LLM_OK);

    const float values[] = {-1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F};
    float actual[2] = {0};
    TEST_ASSERT(llm_tensor_write(backend, &input, values, sizeof(values)) == LLM_OK);

    TEST_ASSERT(llm_reduce_sum_last(backend, &input, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(actual[0] == -6.0F && actual[1] == -15.0F);

    TEST_ASSERT(llm_reduce_max_last(backend, &input, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(actual[0] == -1.0F && actual[1] == -4.0F);

    TEST_ASSERT(llm_reduce_mean_square_last(backend, &input, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(close_enough(actual[0], 14.0F / 3.0F));
    TEST_ASSERT(close_enough(actual[1], 77.0F / 3.0F));

    const size_t vector_shape[] = {3U};
    llm_tensor vector = {0};
    llm_tensor scalar = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, vector_shape, &vector) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &scalar) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &vector, values, 3U * sizeof(float)) == LLM_OK);
    TEST_ASSERT(llm_reduce_sum_last(backend, &vector, &scalar) == LLM_OK);
    float scalar_value = 0.0F;
    TEST_ASSERT(llm_tensor_read(backend, &scalar, &scalar_value, sizeof(scalar_value)) == LLM_OK);
    TEST_ASSERT(scalar_value == -6.0F);
    TEST_ASSERT(llm_reduce_sum_last(backend, &scalar, &scalar) == LLM_INVALID_SHAPE);

    llm_tensor_destroy(&scalar);
    llm_tensor_destroy(&vector);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&input);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_matrix_multiplication(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
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

    float actual[4] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    const float expected[] = {58.0F, 64.0F, 139.0F, 154.0F};
    for (size_t index = 0U; index < 4U; ++index) {
        TEST_ASSERT(actual[index] == expected[index]);
    }
    TEST_ASSERT(llm_matmul(backend, &left, &right, &left) == LLM_INVALID_SHAPE);

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

int main(void) {
    if (test_elementwise_operations() != EXIT_SUCCESS || test_reductions() != EXIT_SUCCESS ||
        test_matrix_multiplication() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
