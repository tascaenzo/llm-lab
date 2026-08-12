#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "model/model.h"
#include "test_support.h"

static int close_enough(float left, float right, float tolerance) {
    return fabsf(left - right) < tolerance;
}

static int create_fixture(llm_backend **out_backend, lm_model **out_model) {
    *out_backend = NULL;
    *out_model = NULL;
    TEST_ASSERT(llm_backend_cpu_create(out_backend) == LLM_OK);
    const lm_model_config config = {.vocabulary_size = 3U,
                                    .context_length = 2U,
                                    .hidden_size = 2U,
                                    .layer_count = 0U,
                                    .head_count = 0U,
                                    .feed_forward_size = 0U,
                                    .seed = UINT64_C(7)};
    TEST_ASSERT(lm_model_create(*out_backend, &config, out_model) == LLM_OK);
    const float embedding[] = {1.0F, 0.0F, 0.5F, 0.5F, 0.0F, 1.0F};
    const float output_weight[] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
    TEST_ASSERT(llm_tensor_write(*out_backend, lm_model_parameter_value(*out_model, 0U), embedding,
                                 sizeof(embedding)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(*out_backend, lm_model_parameter_value(*out_model, 1U),
                                 output_weight, sizeof(output_weight)) == LLM_OK);
    return EXIT_SUCCESS;
}

static int test_forward_and_gradients(void) {
    llm_backend *backend = NULL;
    lm_model *model = NULL;
    if (create_fixture(&backend, &model) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    const size_t input_shape[] = {1U, 2U};
    const size_t logits_shape[] = {2U, 3U};
    const size_t targets_shape[] = {2U};
    llm_tensor inputs = {0};
    llm_tensor logits = {0};
    llm_tensor targets = {0};
    llm_tensor loss = {0};
    llm_tensor logits_gradient = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &inputs) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, targets_shape, &targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits_gradient) ==
                LLM_OK);
    const uint32_t input_values[] = {0U, 2U};
    const uint32_t target_values[] = {1U, 0U};
    TEST_ASSERT(llm_tensor_write(backend, &inputs, input_values, sizeof(input_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &targets, target_values, sizeof(target_values)) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &inputs, &logits) == LLM_OK);
    float actual_logits[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &logits, actual_logits, sizeof(actual_logits)) == LLM_OK);
    const float expected_logits[] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(close_enough(actual_logits[index], expected_logits[index], 1.0e-6F));
    }
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_backward(backend, &logits, &targets, &logits_gradient) == LLM_OK);
    TEST_ASSERT(lm_model_zero_grad(model) == LLM_OK);
    TEST_ASSERT(lm_model_backward(model, &inputs, &logits_gradient) == LLM_OK);

    float output_gradient[6] = {0};
    float embedding_gradient[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, lm_model_parameter_gradient(model, 1U), output_gradient,
                                sizeof(output_gradient)) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, lm_model_parameter_gradient(model, 0U), embedding_gradient,
                                sizeof(embedding_gradient)) == LLM_OK);

    float output_weight[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, lm_model_parameter_value(model, 1U), output_weight,
                                sizeof(output_weight)) == LLM_OK);
    const float epsilon = 1.0e-3F;
    output_weight[0] += epsilon;
    TEST_ASSERT(llm_tensor_write(backend, lm_model_parameter_value(model, 1U), output_weight,
                                 sizeof(output_weight)) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &inputs, &logits) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
    float positive_loss = 0.0F;
    TEST_ASSERT(llm_tensor_read(backend, &loss, &positive_loss, sizeof(positive_loss)) == LLM_OK);
    output_weight[0] -= 2.0F * epsilon;
    TEST_ASSERT(llm_tensor_write(backend, lm_model_parameter_value(model, 1U), output_weight,
                                 sizeof(output_weight)) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &inputs, &logits) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
    float negative_loss = 0.0F;
    TEST_ASSERT(llm_tensor_read(backend, &loss, &negative_loss, sizeof(negative_loss)) == LLM_OK);
    TEST_ASSERT(close_enough(output_gradient[0], (positive_loss - negative_loss) / (2.0F * epsilon),
                             1.0e-3F));
    output_weight[0] += epsilon;
    TEST_ASSERT(llm_tensor_write(backend, lm_model_parameter_value(model, 1U), output_weight,
                                 sizeof(output_weight)) == LLM_OK);
    float embedding[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, lm_model_parameter_value(model, 0U), embedding,
                                sizeof(embedding)) == LLM_OK);
    embedding[0] += epsilon;
    TEST_ASSERT(llm_tensor_write(backend, lm_model_parameter_value(model, 0U), embedding,
                                 sizeof(embedding)) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &inputs, &logits) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &loss, &positive_loss, sizeof(positive_loss)) == LLM_OK);
    embedding[0] -= 2.0F * epsilon;
    TEST_ASSERT(llm_tensor_write(backend, lm_model_parameter_value(model, 0U), embedding,
                                 sizeof(embedding)) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &inputs, &logits) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(backend, &logits, &targets, &loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &loss, &negative_loss, sizeof(negative_loss)) == LLM_OK);
    TEST_ASSERT(close_enough(embedding_gradient[0],
                             (positive_loss - negative_loss) / (2.0F * epsilon), 1.0e-3F));

    llm_tensor_destroy(&logits_gradient);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&inputs);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_seed_is_reproducible(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const lm_model_config config = {.vocabulary_size = 5U,
                                    .context_length = 1U,
                                    .hidden_size = 3U,
                                    .layer_count = 0U,
                                    .head_count = 0U,
                                    .feed_forward_size = 0U,
                                    .seed = UINT64_C(99)};
    lm_model *first = NULL;
    lm_model *second = NULL;
    TEST_ASSERT(lm_model_create(backend, &config, &first) == LLM_OK);
    TEST_ASSERT(lm_model_create(backend, &config, &second) == LLM_OK);
    float first_values[15] = {0};
    float second_values[15] = {0};
    TEST_ASSERT(llm_tensor_read(backend, lm_model_parameter_value(first, 0U), first_values,
                                sizeof(first_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, lm_model_parameter_value(second, 0U), second_values,
                                sizeof(second_values)) == LLM_OK);
    for (size_t index = 0U; index < 15U; ++index) {
        TEST_ASSERT(first_values[index] == second_values[index]);
    }
    TEST_ASSERT(lm_model_parameter_count(first) == 2U);
    TEST_ASSERT(lm_model_parameter_name(first, 0U) != NULL);
    lm_model_destroy(second);
    lm_model_destroy(first);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

static int test_transformer_is_causal(void) {
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const lm_model_config config = {.vocabulary_size = 5U,
                                    .context_length = 3U,
                                    .hidden_size = 4U,
                                    .layer_count = 1U,
                                    .head_count = 1U,
                                    .feed_forward_size = 0U,
                                    .seed = UINT64_C(44)};
    lm_model *model = NULL;
    TEST_ASSERT(lm_model_create(backend, &config, &model) == LLM_OK);
    TEST_ASSERT(lm_model_parameter_count(model) == 7U);
    const size_t input_shape[] = {1U, 3U};
    const size_t logits_shape[] = {3U, 5U};
    llm_tensor first_input = {0};
    llm_tensor second_input = {0};
    llm_tensor first_logits = {0};
    llm_tensor second_logits = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &first_input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &second_input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &first_logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &second_logits) == LLM_OK);
    const uint32_t original[] = {1U, 2U, 3U};
    const uint32_t changed_future[] = {1U, 2U, 4U};
    TEST_ASSERT(llm_tensor_write(backend, &first_input, original, sizeof(original)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &second_input, changed_future, sizeof(changed_future)) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &first_input, &first_logits) == LLM_OK);
    TEST_ASSERT(lm_model_forward(model, &second_input, &second_logits) == LLM_OK);
    float first_values[15] = {0};
    float second_values[15] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &first_logits, first_values, sizeof(first_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &second_logits, second_values, sizeof(second_values)) == LLM_OK);
    for (size_t index = 0U; index < 10U; ++index) {
        TEST_ASSERT(close_enough(first_values[index], second_values[index], 1.0e-6F));
    }
    llm_tensor_destroy(&second_logits);
    llm_tensor_destroy(&first_logits);
    llm_tensor_destroy(&second_input);
    llm_tensor_destroy(&first_input);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    return EXIT_SUCCESS;
}

int main(void) {
    if (test_forward_and_gradients() != EXIT_SUCCESS || test_seed_is_reproducible() != EXIT_SUCCESS ||
        test_transformer_is_causal() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
