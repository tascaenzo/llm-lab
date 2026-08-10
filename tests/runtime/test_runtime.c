#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/runtime.h"
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
        test_tensor_memory_operations() != EXIT_SUCCESS ||
        test_u32_tensor_memory() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
