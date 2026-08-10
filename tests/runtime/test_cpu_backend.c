#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "cpu_executor.h"
#include "runtime/runtime.h"
#include "test_support.h"

typedef struct visit_job {
    uint32_t *visits;
    size_t failure_index;
} visit_job;

static llm_status visit_range(void *context, size_t begin, size_t end) {
    visit_job *job = context;
    if (job->failure_index >= begin && job->failure_index < end) {
        return LLM_NUMERICAL_ERROR;
    }
    for (size_t index = begin; index < end; ++index) {
        ++job->visits[index];
    }
    return LLM_OK;
}

static int close_enough(float left, float right) { return fabsf(left - right) < 1.0e-5F; }

static int test_executor(void) {
    llm_cpu_executor *executor = NULL;
    TEST_ASSERT(llm_cpu_executor_create(0U, &executor) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_cpu_executor_create(4U, &executor) == LLM_OK);
    TEST_ASSERT(llm_cpu_executor_thread_count(executor) == 4U);

    const size_t item_count = 100000U;
    uint32_t *visits = calloc(item_count, sizeof(*visits));
    TEST_ASSERT(visits != NULL);
    visit_job job = {.visits = visits, .failure_index = SIZE_MAX};
    TEST_ASSERT(llm_cpu_parallel_for(executor, item_count, 64U, visit_range, &job) == LLM_OK);
    for (size_t index = 0U; index < item_count; ++index) {
        TEST_ASSERT(visits[index] == 1U);
    }

    job.failure_index = 5000U;
    TEST_ASSERT(llm_cpu_parallel_for(executor, item_count, 64U, visit_range, &job) ==
                LLM_NUMERICAL_ERROR);
    job.failure_index = SIZE_MAX;
    TEST_ASSERT(llm_cpu_parallel_for(executor, 17U, 64U, visit_range, &job) == LLM_OK);

    free(visits);
    llm_cpu_executor_destroy(executor);
    return EXIT_SUCCESS;
}

static int test_configuration(void) {
    llm_backend *backend = NULL;
    const llm_cpu_backend_config invalid_determinism = {
        .thread_count = 1U,
        .deterministic = 2,
    };
    const llm_cpu_backend_config too_many_threads = {
        .thread_count = SIZE_MAX,
        .deterministic = 1,
    };
    TEST_ASSERT(llm_backend_cpu_create_with_config(NULL, &backend) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_backend_cpu_create_with_config(&invalid_determinism, &backend) ==
                LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_backend_cpu_create_with_config(&too_many_threads, &backend) ==
                LLM_INVALID_ARGUMENT);

    const llm_cpu_backend_config single_thread = {
        .thread_count = 1U,
        .deterministic = 1,
    };
    TEST_ASSERT(llm_backend_cpu_create_with_config(&single_thread, &backend) == LLM_OK);
    TEST_ASSERT(llm_backend_cpu_thread_count(backend) == 1U);
    llm_backend_destroy(backend);

    const llm_cpu_backend_config four_threads = {
        .thread_count = 4U,
        .deterministic = 1,
    };
    backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create_with_config(&four_threads, &backend) == LLM_OK);
    TEST_ASSERT(llm_backend_cpu_thread_count(backend) == 4U);
    llm_backend_destroy(backend);
    TEST_ASSERT(llm_backend_cpu_thread_count(NULL) == 0U);
    return EXIT_SUCCESS;
}

static int test_parallel_elementwise_and_memory(void) {
    const llm_cpu_backend_config config = {.thread_count = 4U, .deterministic = 1};
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create_with_config(&config, &backend) == LLM_OK);

    const size_t value_count = 524288U;
    const size_t shape[] = {value_count};
    const size_t byte_count = value_count * sizeof(float);
    float *left_values = malloc(byte_count);
    float *right_values = malloc(byte_count);
    float *actual = malloc(byte_count);
    TEST_ASSERT(left_values != NULL && right_values != NULL && actual != NULL);
    for (size_t index = 0U; index < value_count; ++index) {
        left_values[index] = (float)(index % 17U);
        right_values[index] = (float)(index % 13U);
    }

    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor output = {0};
    llm_tensor copy = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &copy) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &left, left_values, byte_count) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &right, right_values, byte_count) == LLM_OK);

    TEST_ASSERT(llm_add(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, byte_count) == LLM_OK);
    for (size_t index = 0U; index < value_count; ++index) {
        TEST_ASSERT(actual[index] == left_values[index] + right_values[index]);
    }

    TEST_ASSERT(llm_multiply(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_scale(backend, &output, 0.5F, &copy) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &copy, actual, byte_count) == LLM_OK);
    for (size_t index = 0U; index < value_count; ++index) {
        TEST_ASSERT(actual[index] == left_values[index] * right_values[index] * 0.5F);
    }

    TEST_ASSERT(llm_tensor_fill_f32(backend, &output, 3.25F) == LLM_OK);
    TEST_ASSERT(llm_tensor_copy(backend, &output, &copy) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &copy, actual, byte_count) == LLM_OK);
    for (size_t index = 0U; index < value_count; ++index) {
        TEST_ASSERT(actual[index] == 3.25F);
    }
    TEST_ASSERT(llm_tensor_zero(backend, &copy) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &copy, actual, byte_count) == LLM_OK);
    for (size_t index = 0U; index < value_count; ++index) {
        TEST_ASSERT(actual[index] == 0.0F);
    }

    llm_tensor_destroy(&copy);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    free(actual);
    free(right_values);
    free(left_values);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_parallel_reductions_and_matmul(void) {
    const llm_cpu_backend_config config = {.thread_count = 4U, .deterministic = 1};
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create_with_config(&config, &backend) == LLM_OK);

    const size_t reduction_shape[] = {512U, 128U};
    const size_t reduced_shape[] = {512U};
    llm_tensor input = {0};
    llm_tensor reduced = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, reduction_shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, reduced_shape, &reduced) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &input, 0.25F) == LLM_OK);
    float reduced_values[512] = {0};

    TEST_ASSERT(llm_reduce_sum_last(backend, &input, &reduced) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &reduced, reduced_values, sizeof(reduced_values)) ==
                LLM_OK);
    for (size_t row = 0U; row < 512U; ++row) {
        TEST_ASSERT(reduced_values[row] == 32.0F);
    }
    TEST_ASSERT(llm_reduce_max_last(backend, &input, &reduced) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &reduced, reduced_values, sizeof(reduced_values)) ==
                LLM_OK);
    for (size_t row = 0U; row < 512U; ++row) {
        TEST_ASSERT(reduced_values[row] == 0.25F);
    }
    TEST_ASSERT(llm_reduce_mean_square_last(backend, &input, &reduced) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &reduced, reduced_values, sizeof(reduced_values)) ==
                LLM_OK);
    for (size_t row = 0U; row < 512U; ++row) {
        TEST_ASSERT(reduced_values[row] == 0.0625F);
    }

    const size_t left_shape[] = {128U, 64U};
    const size_t right_shape[] = {64U, 64U};
    const size_t output_shape[] = {128U, 64U};
    llm_tensor left = {0};
    llm_tensor right = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, left_shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, right_shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &left, 1.0F) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &right, 0.5F) == LLM_OK);
    TEST_ASSERT(llm_matmul(backend, &left, &right, &output) == LLM_OK);
    float matmul_values[128U * 64U] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, matmul_values, sizeof(matmul_values)) == LLM_OK);
    for (size_t index = 0U; index < 128U * 64U; ++index) {
        TEST_ASSERT(matmul_values[index] == 32.0F);
    }

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    llm_tensor_destroy(&reduced);
    llm_tensor_destroy(&input);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_parallel_language_operations(void) {
    const llm_cpu_backend_config config = {.thread_count = 4U, .deterministic = 1};
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create_with_config(&config, &backend) == LLM_OK);

    const size_t shape[] = {256U, 128U};
    const size_t target_shape[] = {256U};
    llm_tensor logits = {0};
    llm_tensor probabilities = {0};
    llm_tensor gradient = {0};
    llm_tensor targets = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &probabilities) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_zero(backend, &logits) == LLM_OK);
    uint32_t target_values[256] = {0};
    for (size_t row = 0U; row < 256U; ++row) {
        target_values[row] = (uint32_t)(row % 128U);
    }
    TEST_ASSERT(llm_tensor_write(backend, &targets, target_values, sizeof(target_values)) ==
                LLM_OK);

    TEST_ASSERT(llm_softmax_last(backend, &logits, &probabilities) == LLM_OK);
    float values[256U * 128U] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &probabilities, values, sizeof(values)) == LLM_OK);
    for (size_t index = 0U; index < 256U * 128U; ++index) {
        TEST_ASSERT(values[index] == 1.0F / 128.0F);
    }

    TEST_ASSERT(llm_cross_entropy_backward(backend, &logits, &targets, &gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &gradient, values, sizeof(values)) == LLM_OK);
    for (size_t row = 0U; row < 256U; ++row) {
        float row_sum = 0.0F;
        for (size_t column = 0U; column < 128U; ++column) {
            row_sum += values[row * 128U + column];
        }
        TEST_ASSERT(close_enough(row_sum, 0.0F));
        TEST_ASSERT(values[row * 128U + target_values[row]] < 0.0F);
    }

    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&gradient);
    llm_tensor_destroy(&probabilities);
    llm_tensor_destroy(&logits);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

int main(void) {
    if (test_executor() != EXIT_SUCCESS || test_configuration() != EXIT_SUCCESS ||
        test_parallel_elementwise_and_memory() != EXIT_SUCCESS ||
        test_parallel_reductions_and_matmul() != EXIT_SUCCESS ||
        test_parallel_language_operations() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
