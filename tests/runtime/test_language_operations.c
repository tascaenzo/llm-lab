#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "runtime/backend.h"
#include "runtime/operations.h"
#include "test_support.h"

static int close_enough(float left, float right) { return fabsf(left - right) < 1.0e-5F; }

static int test_gather_and_scatter(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const size_t table_shape[] = {4U, 3U};
    const size_t indices_shape[] = {3U};
    const size_t output_shape[] = {3U, 3U};
    llm_tensor table = {0};
    llm_tensor indices = {0};
    llm_tensor output = {0};
    llm_tensor accumulated = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &table) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, indices_shape, &indices) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &accumulated) == LLM_OK);

    const float table_values[] = {0.0F,  1.0F,  2.0F,  10.0F, 11.0F, 12.0F,
                                  20.0F, 21.0F, 22.0F, 30.0F, 31.0F, 32.0F};
    const uint32_t index_values[] = {2U, 0U, 2U};
    TEST_ASSERT(llm_tensor_write(backend, &table, table_values, sizeof(table_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &indices, index_values, sizeof(index_values)) == LLM_OK);
    TEST_ASSERT(llm_gather_rows(backend, &table, &indices, &output) == LLM_OK);

    float gathered[9] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, gathered, sizeof(gathered)) == LLM_OK);
    const float expected[] = {20.0F, 21.0F, 22.0F, 0.0F, 1.0F, 2.0F, 20.0F, 21.0F, 22.0F};
    for (size_t index = 0U; index < 9U; ++index) {
        TEST_ASSERT(gathered[index] == expected[index]);
    }

    TEST_ASSERT(llm_tensor_zero(backend, &accumulated) == LLM_OK);
    TEST_ASSERT(llm_scatter_add_rows(backend, &output, &indices, &accumulated) == LLM_OK);
    float scattered[12] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &accumulated, scattered, sizeof(scattered)) == LLM_OK);
    TEST_ASSERT(scattered[0] == 0.0F && scattered[1] == 1.0F && scattered[2] == 2.0F);
    TEST_ASSERT(scattered[6] == 40.0F && scattered[7] == 42.0F && scattered[8] == 44.0F);
    TEST_ASSERT(scattered[3] == 0.0F && scattered[9] == 0.0F);

    const uint32_t invalid_indices[] = {2U, 4U, 0U};
    TEST_ASSERT(llm_tensor_write(backend, &indices, invalid_indices, sizeof(invalid_indices)) ==
                LLM_OK);
    TEST_ASSERT(llm_gather_rows(backend, &table, &indices, &output) == LLM_INVALID_INDEX);

    llm_tensor_destroy(&accumulated);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&indices);
    llm_tensor_destroy(&table);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_softmax(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const size_t shape[] = {2U, 3U};
    llm_tensor input = {0};
    llm_tensor output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &output) == LLM_OK);

    const float values[] = {1000.0F, 1001.0F, 1002.0F, -3.0F, -3.0F, -3.0F};
    TEST_ASSERT(llm_tensor_write(backend, &input, values, sizeof(values)) == LLM_OK);
    TEST_ASSERT(llm_softmax_last(backend, &input, &output) == LLM_OK);

    float probabilities[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, probabilities, sizeof(probabilities)) == LLM_OK);
    TEST_ASSERT(close_enough(probabilities[0] + probabilities[1] + probabilities[2], 1.0F));
    TEST_ASSERT(probabilities[2] > probabilities[1] && probabilities[1] > probabilities[0]);
    TEST_ASSERT(close_enough(probabilities[3], 1.0F / 3.0F));
    TEST_ASSERT(close_enough(probabilities[4], 1.0F / 3.0F));
    TEST_ASSERT(close_enough(probabilities[5], 1.0F / 3.0F));
    TEST_ASSERT(llm_softmax_last(backend, &input, &input) == LLM_INVALID_ARGUMENT);

    const float invalid_values[] = {NAN, 0.0F, 1.0F, 1.0F, 2.0F, 3.0F};
    TEST_ASSERT(llm_tensor_write(backend, &input, invalid_values, sizeof(invalid_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_softmax_last(backend, &input, &output) == LLM_NUMERICAL_ERROR);

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&input);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_cross_entropy(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const size_t logits_shape[] = {2U, 3U};
    const size_t targets_shape[] = {2U};
    llm_tensor logits = {0};
    llm_tensor targets = {0};
    llm_tensor loss_mask = {0};
    llm_tensor loss = {0};
    llm_tensor gradient = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, targets_shape, &targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, targets_shape, &loss_mask) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &gradient) == LLM_OK);

    const float logits_values[] = {2.0F, 1.0F, 0.0F, 0.0F, 1.0F, 2.0F};
    const uint32_t target_values[] = {0U, 2U};
    TEST_ASSERT(llm_tensor_write(backend, &logits, logits_values, sizeof(logits_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &targets, target_values, sizeof(target_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);

    float loss_value = 0.0F;
    TEST_ASSERT(llm_tensor_read(backend, &loss, &loss_value, sizeof(loss_value)) == LLM_OK);
    TEST_ASSERT(close_enough(loss_value, 0.40760595F));

    TEST_ASSERT(llm_cross_entropy_backward(backend, &logits, &targets, &gradient) == LLM_OK);
    float gradient_values[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &gradient, gradient_values, sizeof(gradient_values)) ==
                LLM_OK);
    const float expected[] = {-0.16737952F, 0.12236424F, 0.04501529F,
                              0.04501529F,  0.12236424F, -0.16737952F};
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(close_enough(gradient_values[index], expected[index]));
    }
    TEST_ASSERT(close_enough(gradient_values[0] + gradient_values[1] + gradient_values[2], 0.0F));

    const uint32_t mask_values[] = {1U, 0U};
    TEST_ASSERT(llm_tensor_write(backend, &loss_mask, mask_values, sizeof(mask_values)) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_masked_forward(backend, &logits, &targets, &loss_mask, 1U,
                                                 &loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &loss, &loss_value, sizeof(loss_value)) == LLM_OK);
    TEST_ASSERT(close_enough(loss_value, 0.40760595F));
    TEST_ASSERT(llm_cross_entropy_masked_backward(backend, &logits, &targets, &loss_mask, 1U,
                                                  &gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &gradient, gradient_values, sizeof(gradient_values)) ==
                LLM_OK);
    TEST_ASSERT(close_enough(gradient_values[0], -0.33475904F));
    TEST_ASSERT(close_enough(gradient_values[1], 0.24472848F));
    TEST_ASSERT(close_enough(gradient_values[2], 0.09003058F));
    TEST_ASSERT(gradient_values[3] == 0.0F && gradient_values[4] == 0.0F &&
                gradient_values[5] == 0.0F);

    const uint32_t invalid_mask[] = {1U, 2U};
    TEST_ASSERT(llm_tensor_write(backend, &loss_mask, invalid_mask, sizeof(invalid_mask)) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_masked_forward(backend, &logits, &targets, &loss_mask, 1U,
                                                 &loss) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_tensor_write(backend, &loss_mask, mask_values, sizeof(mask_values)) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_masked_forward(backend, &logits, &targets, &loss_mask, 2U,
                                                 &loss) == LLM_OK);

    for (size_t index = 0U; index < 6U; ++index) {
        gradient_values[index] = expected[index];
    }
    const float epsilon = 1.0e-3F;
    for (size_t changed = 0U; changed < 6U; ++changed) {
        float perturbed[6] = {0};
        for (size_t index = 0U; index < 6U; ++index) {
            perturbed[index] = logits_values[index];
        }
        perturbed[changed] += epsilon;
        TEST_ASSERT(llm_tensor_write(backend, &logits, perturbed, sizeof(perturbed)) == LLM_OK);
        TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
        float positive_loss = 0.0F;
        TEST_ASSERT(llm_tensor_read(backend, &loss, &positive_loss, sizeof(positive_loss)) ==
                    LLM_OK);

        perturbed[changed] -= 2.0F * epsilon;
        TEST_ASSERT(llm_tensor_write(backend, &logits, perturbed, sizeof(perturbed)) == LLM_OK);
        TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
        float negative_loss = 0.0F;
        TEST_ASSERT(llm_tensor_read(backend, &loss, &negative_loss, sizeof(negative_loss)) ==
                    LLM_OK);

        const float numerical_gradient = (positive_loss - negative_loss) / (2.0F * epsilon);
        TEST_ASSERT(fabsf(numerical_gradient - gradient_values[changed]) < 1.0e-3F);
    }

    const uint32_t invalid_targets[] = {0U, 3U};
    TEST_ASSERT(llm_tensor_write(backend, &targets, invalid_targets, sizeof(invalid_targets)) ==
                LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_INVALID_INDEX);

    llm_tensor_destroy(&gradient);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&loss_mask);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&logits);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

int main(void) {
    if (test_gather_and_scatter() != EXIT_SUCCESS || test_softmax() != EXIT_SUCCESS ||
        test_cross_entropy() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
