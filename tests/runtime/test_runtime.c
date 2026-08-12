#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/backend.h"
#include "runtime/operations.h"
#include "runtime/tensor.h"
#include "test_support.h"

static int test_backend_lifecycle(void) {
    TEST_ASSERT(llm_backend_cpu_create(NULL) == LLM_INVALID_ARGUMENT);

    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    TEST_ASSERT(backend != NULL);
    TEST_ASSERT(llm_backend_device(backend) == LLM_DEVICE_CPU);
    TEST_ASSERT(llm_backend_device(NULL) == LLM_DEVICE_NONE);
    TEST_ASSERT(llm_backend_synchronize(backend) == LLM_OK);
    TEST_ASSERT(llm_backend_synchronize(NULL) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(strcmp(llm_status_string(LLM_INVALID_SHAPE), "invalid tensor shape") == 0);

    llm_backend_destroy(backend);
    llm_backend_destroy(NULL);
    return EXIT_SUCCESS;
}

static int test_tensor_layout_and_lifecycle(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);

    llm_tensor scalar = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &scalar) == LLM_OK);
    TEST_ASSERT(scalar.rank == 0U);
    TEST_ASSERT(scalar.element_count == 1U);
    TEST_ASSERT(llm_tensor_is_contiguous(&scalar) != 0);
    TEST_ASSERT(llm_tensor_device(&scalar) == LLM_DEVICE_CPU);

    const size_t shape[] = {2U, 3U, 4U};
    llm_tensor tensor = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 3U, shape, &tensor) == LLM_OK);
    TEST_ASSERT(tensor.rank == 3U);
    TEST_ASSERT(tensor.element_count == 24U);
    TEST_ASSERT(tensor.shape[0] == 2U && tensor.shape[1] == 3U && tensor.shape[2] == 4U);
    TEST_ASSERT(tensor.strides[0] == 12U && tensor.strides[1] == 4U && tensor.strides[2] == 1U);
    TEST_ASSERT(llm_tensor_is_contiguous(&tensor) != 0);

    llm_tensor occupied = tensor;
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 3U, shape, &occupied) ==
                LLM_INVALID_ARGUMENT);

    llm_tensor moved = {0};
    TEST_ASSERT(llm_tensor_move(&tensor, &moved) == LLM_OK);
    TEST_ASSERT(tensor.storage == NULL);
    TEST_ASSERT(moved.element_count == 24U);
    TEST_ASSERT(llm_tensor_move(&moved, &moved) == LLM_INVALID_ARGUMENT);

    const size_t invalid_shape[] = {2U, 0U};
    llm_tensor invalid = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, invalid_shape, &invalid) ==
                LLM_INVALID_SHAPE);
    TEST_ASSERT(invalid.storage == NULL);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, LLM_TENSOR_MAX_RANK + 1U, shape,
                                  &invalid) == LLM_INVALID_SHAPE);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F8_E4M3, 1U, shape, &invalid) ==
                LLM_UNSUPPORTED_DTYPE);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F16, 1U, shape, &invalid) ==
                LLM_UNSUPPORTED_DTYPE);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_BF16, 1U, shape, &invalid) ==
                LLM_UNSUPPORTED_DTYPE);

    const size_t overflow_shape[] = {SIZE_MAX, 2U};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, overflow_shape, &invalid) ==
                LLM_OVERFLOW);

    llm_tensor_destroy(&scalar);
    llm_tensor_destroy(&moved);
    llm_tensor_destroy(&invalid);
    llm_tensor_destroy(NULL);
    TEST_ASSERT(llm_tensor_device(&moved) == LLM_DEVICE_NONE);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_tensor_reshape_and_shared_storage(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);

    const size_t source_shape[] = {2U, 3U, 4U};
    const size_t matrix_shape[] = {6U, 4U};
    const size_t vector_shape[] = {24U};
    llm_tensor source = {0};
    llm_tensor matrix_view = {0};
    llm_tensor vector_view = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 3U, source_shape, &source) == LLM_OK);
    TEST_ASSERT(llm_tensor_reshape(&source, 2U, matrix_shape, &matrix_view) == LLM_OK);
    TEST_ASSERT(matrix_view.storage == source.storage);
    TEST_ASSERT(matrix_view.rank == 2U && matrix_view.shape[0] == 6U && matrix_view.shape[1] == 4U);
    TEST_ASSERT(matrix_view.strides[0] == 4U && matrix_view.strides[1] == 1U);
    TEST_ASSERT(matrix_view.element_count == source.element_count);
    TEST_ASSERT(matrix_view.dtype == source.dtype);
    TEST_ASSERT(llm_tensor_device(&matrix_view) == LLM_DEVICE_CPU);
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, 1U, vector_shape, &vector_view) == LLM_OK);
    TEST_ASSERT(vector_view.storage == source.storage);

    float input[24] = {0};
    for (size_t index = 0U; index < 24U; ++index) {
        input[index] = (float)index * 0.25F;
    }
    TEST_ASSERT(llm_tensor_write(backend, &source, input, sizeof(input)) == LLM_OK);
    float output[24] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &vector_view, output, sizeof(output)) == LLM_OK);
    TEST_ASSERT(memcmp(input, output, sizeof(input)) == 0);

    TEST_ASSERT(llm_tensor_fill_f32(backend, &matrix_view, 3.5F) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &source, output, sizeof(output)) == LLM_OK);
    for (size_t index = 0U; index < 24U; ++index) {
        TEST_ASSERT(output[index] == 3.5F);
    }

    llm_tensor_destroy(&source);
    TEST_ASSERT(llm_tensor_device(&source) == LLM_DEVICE_NONE);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &matrix_view, -2.0F) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &vector_view, output, sizeof(output)) == LLM_OK);
    for (size_t index = 0U; index < 24U; ++index) {
        TEST_ASSERT(output[index] == -2.0F);
    }

    const size_t wrong_shape[] = {5U, 5U};
    const size_t zero_shape[] = {24U, 0U};
    llm_tensor invalid = {0};
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, 2U, wrong_shape, &invalid) == LLM_INVALID_SHAPE);
    TEST_ASSERT(invalid.storage == NULL);
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, 2U, zero_shape, &invalid) == LLM_INVALID_SHAPE);
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, LLM_TENSOR_MAX_RANK + 1U, source_shape,
                                   &invalid) == LLM_INVALID_SHAPE);
    TEST_ASSERT(llm_tensor_reshape(NULL, 1U, vector_shape, &invalid) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, 1U, vector_shape, NULL) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, 1U, vector_shape, &matrix_view) ==
                LLM_INVALID_ARGUMENT);

    llm_tensor malformed = matrix_view;
    malformed.strides[0] = 5U;
    TEST_ASSERT(llm_tensor_reshape(&malformed, 1U, vector_shape, &invalid) ==
                LLM_UNSUPPORTED_LAYOUT);

    llm_tensor occupied = {0};
    const size_t occupied_shape[] = {1U};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, occupied_shape, &occupied) == LLM_OK);
    llm_storage *occupied_storage = occupied.storage;
    TEST_ASSERT(llm_tensor_reshape(&matrix_view, 1U, vector_shape, &occupied) ==
                LLM_INVALID_ARGUMENT);
    TEST_ASSERT(occupied.storage == occupied_storage);

    llm_tensor scalar_source = {0};
    llm_tensor scalar_view = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, occupied_shape, &scalar_source) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_reshape(&scalar_source, 0U, NULL, &scalar_view) == LLM_OK);
    TEST_ASSERT(scalar_view.rank == 0U && scalar_view.element_count == 1U);

    llm_tensor_destroy(&scalar_source);
    llm_tensor_destroy(&scalar_view);
    llm_tensor_destroy(&occupied);
    llm_tensor_destroy(&invalid);
    llm_tensor_destroy(&matrix_view);
    llm_tensor_destroy(&vector_view);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_tensor_memory_operations(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);

    const size_t shape[] = {2U, 3U};
    llm_tensor source = {0};
    llm_tensor destination = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &source) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &destination) == LLM_OK);

    const float input[] = {1.0F, -2.0F, 3.5F, 4.0F, 5.25F, -6.0F};
    float output[6] = {0};
    TEST_ASSERT(llm_tensor_write(backend, &source, input, sizeof(input)) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &source, output, sizeof(output)) == LLM_OK);
    TEST_ASSERT(memcmp(input, output, sizeof(input)) == 0);

    TEST_ASSERT(llm_tensor_copy(backend, &source, &destination) == LLM_OK);
    (void)memset(output, 0, sizeof(output));
    TEST_ASSERT(llm_tensor_read(backend, &destination, output, sizeof(output)) == LLM_OK);
    TEST_ASSERT(memcmp(input, output, sizeof(input)) == 0);

    TEST_ASSERT(llm_tensor_fill_f32(backend, &destination, 2.5F) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &destination, output, sizeof(output)) == LLM_OK);
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(output[index] == 2.5F);
    }

    TEST_ASSERT(llm_tensor_zero(backend, &destination) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &destination, output, sizeof(output)) == LLM_OK);
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(output[index] == 0.0F);
    }

    TEST_ASSERT(llm_tensor_write(backend, &source, input, sizeof(input) - 1U) ==
                LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_tensor_read(backend, &source, output, sizeof(output) - 1U) ==
                LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_tensor_copy(backend, &source, &source) == LLM_INVALID_ARGUMENT);

    const size_t different_shape[] = {3U, 2U};
    llm_tensor different = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, different_shape, &different) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_copy(backend, &source, &different) == LLM_INVALID_SHAPE);

    llm_backend *other_backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&other_backend) == LLM_OK);
    llm_tensor foreign = {0};
    TEST_ASSERT(llm_tensor_create(other_backend, LLM_DTYPE_F32, 2U, shape, &foreign) == LLM_OK);
    TEST_ASSERT(llm_tensor_copy(backend, &source, &foreign) == LLM_DEVICE_MISMATCH);

    llm_tensor_destroy(&foreign);
    llm_backend_destroy(other_backend);
    llm_tensor_destroy(&different);
    llm_tensor_destroy(&destination);
    llm_tensor_destroy(&source);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_u32_tensor_memory(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);

    const size_t shape[] = {4U};
    llm_tensor tensor = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, shape, &tensor) == LLM_OK);

    const uint32_t input[] = {0U, 7U, UINT32_MAX, 42U};
    uint32_t output[4] = {0};
    TEST_ASSERT(llm_tensor_write(backend, &tensor, input, sizeof(input)) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &tensor, output, sizeof(output)) == LLM_OK);
    TEST_ASSERT(memcmp(input, output, sizeof(input)) == 0);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &tensor, 1.0F) == LLM_UNSUPPORTED_DTYPE);
    TEST_ASSERT(llm_tensor_zero(backend, &tensor) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &tensor, output, sizeof(output)) == LLM_OK);
    for (size_t index = 0U; index < 4U; ++index) {
        TEST_ASSERT(output[index] == 0U);
    }

    llm_tensor_destroy(&tensor);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

int main(void) {
    if (test_backend_lifecycle() != EXIT_SUCCESS ||
        test_tensor_layout_and_lifecycle() != EXIT_SUCCESS ||
        test_tensor_reshape_and_shared_storage() != EXIT_SUCCESS ||
        test_tensor_memory_operations() != EXIT_SUCCESS ||
        test_u32_tensor_memory() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
