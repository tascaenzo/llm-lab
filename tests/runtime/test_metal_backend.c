#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/backend.h"
#include "runtime/operations.h"
#include "backend_contract_suite.h"
#include "model/model.h"
#include "test_support.h"

static int close_enough(float left, float right) { return fabsf(left - right) < 2.0e-5F; }

static int compare_tensor_values(llm_backend *cpu_backend, const llm_tensor *cpu_tensor,
                                 llm_backend *metal_backend, const llm_tensor *metal_tensor,
                                 const char *stage, float absolute_tolerance) {
    if (cpu_tensor == NULL || metal_tensor == NULL || stage == NULL ||
        cpu_tensor->element_count != metal_tensor->element_count) {
        return EXIT_FAILURE;
    }
    float *cpu_values = calloc(cpu_tensor->element_count, sizeof(*cpu_values));
    float *metal_values = calloc(metal_tensor->element_count, sizeof(*metal_values));
    if (cpu_values == NULL || metal_values == NULL ||
        llm_tensor_read(cpu_backend, cpu_tensor, cpu_values,
                        cpu_tensor->element_count * sizeof(*cpu_values)) != LLM_OK ||
        llm_tensor_read(metal_backend, metal_tensor, metal_values,
                        metal_tensor->element_count * sizeof(*metal_values)) != LLM_OK) {
        free(metal_values);
        free(cpu_values);
        return EXIT_FAILURE;
    }
    float maximum_error = 0.0F;
    size_t maximum_index = 0U;
    for (size_t index = 0U; index < cpu_tensor->element_count; ++index) {
        const float error = fabsf(cpu_values[index] - metal_values[index]);
        if (!isfinite(metal_values[index]) || error > maximum_error) {
            maximum_error = error;
            maximum_index = index;
        }
    }
    const int matches = isfinite(metal_values[maximum_index]) != 0 &&
                        maximum_error <= absolute_tolerance;
    if (matches == 0) {
        fprintf(stderr,
                "%s parity mismatch: max_abs=%g at %zu (cpu=%g metal=%g), tolerance=%g\n",
                stage, (double)maximum_error, maximum_index,
                (double)cpu_values[maximum_index], (double)metal_values[maximum_index],
                (double)absolute_tolerance);
    }
    free(metal_values);
    free(cpu_values);
    return matches != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int test_training_shape_matmul_and_loss_parity(llm_backend *metal_backend) {
    const size_t rows = 64U, hidden = 64U, vocabulary = 32001U;
    const size_t left_shape[] = {rows, hidden};
    const size_t right_shape[] = {hidden, vocabulary};
    const size_t logits_shape[] = {rows, vocabulary};
    const size_t targets_shape[] = {rows};
    llm_backend *cpu_backend = NULL;
    llm_tensor cpu_left = {0}, metal_left = {0}, cpu_right = {0}, metal_right = {0};
    llm_tensor cpu_logits = {0}, metal_logits = {0};
    llm_tensor cpu_targets = {0}, metal_targets = {0};
    llm_tensor cpu_loss = {0}, metal_loss = {0};
    llm_tensor cpu_gradient = {0}, metal_gradient = {0};
    TEST_ASSERT(llm_backend_cpu_create(&cpu_backend) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, left_shape, &cpu_left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, left_shape, &metal_left) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, right_shape, &cpu_right) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, right_shape, &metal_right) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, logits_shape, &cpu_logits) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, logits_shape, &metal_logits) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_U32, 1U, targets_shape, &cpu_targets) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_U32, 1U, targets_shape,
                                  &metal_targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 0U, NULL, &cpu_loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 0U, NULL, &metal_loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, logits_shape, &cpu_gradient) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, logits_shape,
                                  &metal_gradient) == LLM_OK);

    float *left_values = malloc(rows * hidden * sizeof(*left_values));
    float *right_values = malloc(hidden * vocabulary * sizeof(*right_values));
    uint32_t targets[64] = {0};
    TEST_ASSERT(left_values != NULL && right_values != NULL);
    for (size_t index = 0U; index < rows * hidden; ++index) {
        left_values[index] = ((float)((index * 17U + 3U) % 101U) - 50.0F) * 0.0004F;
    }
    for (size_t index = 0U; index < hidden * vocabulary; ++index) {
        right_values[index] = ((float)((index * 29U + 7U) % 103U) - 51.0F) * 0.0004F;
    }
    for (size_t index = 0U; index < rows; ++index) {
        targets[index] = (uint32_t)((index * 997U + 13U) % vocabulary);
    }
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_left, left_values,
                                 rows * hidden * sizeof(*left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_left, left_values,
                                 rows * hidden * sizeof(*left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_right, right_values,
                                 hidden * vocabulary * sizeof(*right_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_right, right_values,
                                 hidden * vocabulary * sizeof(*right_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_targets, targets, sizeof(targets)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_targets, targets, sizeof(targets)) == LLM_OK);
    free(right_values);
    free(left_values);

    TEST_ASSERT(llm_matmul(cpu_backend, &cpu_left, &cpu_right, &cpu_logits) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_begin_batch(metal_backend) == LLM_OK);
    TEST_ASSERT(llm_matmul(metal_backend, &metal_left, &metal_right, &metal_logits) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_end_batch(metal_backend) == LLM_OK);
    TEST_ASSERT(compare_tensor_values(cpu_backend, &cpu_logits, metal_backend, &metal_logits,
                                      "training-shape matmul", 2.0e-5F) == EXIT_SUCCESS);

    TEST_ASSERT(llm_cross_entropy_forward(cpu_backend, &cpu_logits, &cpu_targets, &cpu_loss) ==
                LLM_OK);
    TEST_ASSERT(llm_cross_entropy_backward(cpu_backend, &cpu_logits, &cpu_targets,
                                           &cpu_gradient) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_begin_batch(metal_backend) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_forward(metal_backend, &metal_logits, &metal_targets,
                                          &metal_loss) == LLM_OK);
    TEST_ASSERT(llm_cross_entropy_backward(metal_backend, &metal_logits, &metal_targets,
                                           &metal_gradient) == LLM_OK);
    TEST_ASSERT(llm_backend_metal_end_batch(metal_backend) == LLM_OK);
    TEST_ASSERT(compare_tensor_values(cpu_backend, &cpu_loss, metal_backend, &metal_loss,
                                      "training-shape cross-entropy loss", 2.0e-4F) ==
                EXIT_SUCCESS);
    TEST_ASSERT(compare_tensor_values(cpu_backend, &cpu_gradient, metal_backend, &metal_gradient,
                                      "training-shape cross-entropy gradient", 2.0e-7F) ==
                EXIT_SUCCESS);

    llm_tensor_destroy(&metal_gradient);
    llm_tensor_destroy(&cpu_gradient);
    llm_tensor_destroy(&metal_loss);
    llm_tensor_destroy(&cpu_loss);
    llm_tensor_destroy(&metal_targets);
    llm_tensor_destroy(&cpu_targets);
    llm_tensor_destroy(&metal_logits);
    llm_tensor_destroy(&cpu_logits);
    llm_tensor_destroy(&metal_right);
    llm_tensor_destroy(&cpu_right);
    llm_tensor_destroy(&metal_left);
    llm_tensor_destroy(&cpu_left);
    llm_backend_destroy(cpu_backend);
    return EXIT_SUCCESS;
}

static int run_model_forward(llm_backend *backend, lm_model *model, llm_tensor *inputs,
                             llm_tensor *logits) {
    const int is_metal = llm_backend_device(backend) == LLM_DEVICE_METAL;
    llm_status status = is_metal != 0 ? llm_backend_metal_begin_batch(backend) : LLM_OK;
    if (status == LLM_OK) {
        status = lm_model_forward(model, inputs, logits);
    }
    if (is_metal != 0) {
        const llm_status batch_status = llm_backend_metal_end_batch(backend);
        if (status == LLM_OK) {
            status = batch_status;
        }
    }
    return status == LLM_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}

static float host_cross_entropy_loss(const float *logits, const uint32_t *targets,
                                     size_t row_count, size_t vocabulary_size) {
    double total = 0.0;
    for (size_t row = 0U; row < row_count; ++row) {
        const float *const values = logits + row * vocabulary_size;
        float maximum = values[0];
        for (size_t column = 1U; column < vocabulary_size; ++column) {
            maximum = values[column] > maximum ? values[column] : maximum;
        }
        double sum = 0.0;
        for (size_t column = 0U; column < vocabulary_size; ++column) {
            sum += exp((double)values[column] - (double)maximum);
        }
        total += (double)maximum + log(sum) - (double)values[targets[row]];
    }
    return (float)(total / (double)row_count);
}

static int test_model_forward_parity(llm_backend *cpu_backend, lm_model *cpu_model,
                                     llm_backend *metal_backend, lm_model *metal_model,
                                     llm_tensor *cpu_inputs, llm_tensor *metal_inputs,
                                     llm_tensor *cpu_logits, llm_tensor *metal_logits,
                                     const uint32_t *targets, size_t layer_count) {
    TEST_ASSERT(run_model_forward(cpu_backend, cpu_model, cpu_inputs, cpu_logits) == EXIT_SUCCESS);
    TEST_ASSERT(run_model_forward(metal_backend, metal_model, metal_inputs, metal_logits) ==
                EXIT_SUCCESS);
    float *cpu_values = calloc(cpu_logits->element_count, sizeof(*cpu_values));
    float *metal_values = calloc(metal_logits->element_count, sizeof(*metal_values));
    TEST_ASSERT(cpu_values != NULL && metal_values != NULL);
    TEST_ASSERT(llm_tensor_read(cpu_backend, cpu_logits, cpu_values,
                                cpu_logits->element_count * sizeof(*cpu_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(metal_backend, metal_logits, metal_values,
                                metal_logits->element_count * sizeof(*metal_values)) == LLM_OK);
    float maximum_error = 0.0F;
    size_t maximum_index = 0U;
    for (size_t index = 0U; index < cpu_logits->element_count; ++index) {
        const float error = fabsf(cpu_values[index] - metal_values[index]);
        if (error > maximum_error) {
            maximum_error = error;
            maximum_index = index;
        }
    }
    const float cpu_loss = host_cross_entropy_loss(cpu_values, targets, 64U, 32001U);
    const float metal_loss = host_cross_entropy_loss(metal_values, targets, 64U, 32001U);
    if (maximum_error > 2.0e-4F) {
        fprintf(stderr,
                "model-forward parity mismatch (layers=%zu): max_abs=%g at %zu "
                "(cpu=%g metal=%g); host loss cpu=%g metal=%g\\n",
                layer_count, (double)maximum_error, maximum_index,
                (double)cpu_values[maximum_index], (double)metal_values[maximum_index],
                (double)cpu_loss, (double)metal_loss);
    }
    free(metal_values);
    free(cpu_values);
    TEST_ASSERT(maximum_error <= 2.0e-4F);
    return EXIT_SUCCESS;
}

static int run_model_step(llm_backend *backend, lm_model *model, llm_tensor *inputs,
                          llm_tensor *targets, llm_tensor *logits, llm_tensor *loss,
                          llm_tensor *logits_gradient, unsigned long long step, float *out_loss) {
    const int is_metal = llm_backend_device(backend) == LLM_DEVICE_METAL;
    llm_status status = is_metal != 0 ? llm_backend_metal_begin_batch(backend) : LLM_OK;
    if (status == LLM_OK) {
        status = lm_model_zero_grad(model);
    }
    if (status == LLM_OK) {
        status = lm_model_forward(model, inputs, logits);
    }
    if (status == LLM_OK) {
        status = llm_cross_entropy_forward(backend, logits, targets, loss);
    }
    if (status == LLM_OK) {
        status = llm_cross_entropy_backward(backend, logits, targets, logits_gradient);
    }
    if (status == LLM_OK) {
        status = lm_model_backward(model, inputs, logits_gradient);
    }
    const llm_adamw_options options = {.learning_rate = 1.0e-3F,
                                       .beta1 = 0.9F,
                                       .beta2 = 0.999F,
                                       .epsilon = 1.0e-8F,
                                       .weight_decay = 0.01F,
                                       .gradient_scale = 1.0F,
                                       .step = step};
    if (status == LLM_OK) {
        status = lm_model_apply_adamw(model, &options);
    }
    if (is_metal != 0) {
        const llm_status batch_status = llm_backend_metal_end_batch(backend);
        if (status == LLM_OK) {
            status = batch_status;
        }
    }
    if (status == LLM_OK) {
        status = llm_tensor_read(backend, loss, out_loss, sizeof(*out_loss));
    }
    return status == LLM_OK && isfinite(*out_loss) != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int test_model_training_step_parity_for_layers(llm_backend *metal_backend,
                                                      size_t layer_count) {
    const lm_model_config config = {.vocabulary_size = 32001U,
                                    .context_length = 32U,
                                    .hidden_size = 64U,
                                    .layer_count = layer_count,
                                    .head_count = layer_count == 0U ? 0U : 1U,
                                    .feed_forward_size = 0U,
                                    .seed = UINT64_C(31)};
    const lm_model_config model_config = config;
    llm_backend *cpu_backend = NULL;
    lm_model *cpu_model = NULL;
    lm_model *metal_model = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&cpu_backend) == LLM_OK);
    TEST_ASSERT(lm_model_create(cpu_backend, &model_config, &cpu_model) == LLM_OK);
    TEST_ASSERT(lm_model_create(metal_backend, &model_config, &metal_model) == LLM_OK);
    TEST_ASSERT(lm_model_parameter_count(cpu_model) == lm_model_parameter_count(metal_model));
    for (size_t parameter = 0U; parameter < lm_model_parameter_count(cpu_model); ++parameter) {
        char stage[128] = {0};
        (void)snprintf(stage, sizeof(stage), "initial parameter %s",
                       lm_model_parameter_name(cpu_model, parameter));
        TEST_ASSERT(compare_tensor_values(cpu_backend, lm_model_parameter_value(cpu_model, parameter),
                                          metal_backend,
                                          lm_model_parameter_value(metal_model, parameter), stage,
                                          0.0F) == EXIT_SUCCESS);
    }
    const size_t input_shape[] = {2U, 32U};
    const size_t target_shape[] = {64U};
    const size_t logits_shape[] = {64U, 32001U};
    llm_tensor cpu_inputs = {0}, metal_inputs = {0}, cpu_targets = {0}, metal_targets = {0};
    llm_tensor cpu_logits = {0}, metal_logits = {0}, cpu_loss = {0}, metal_loss = {0};
    llm_tensor cpu_gradient = {0}, metal_gradient = {0};
    uint32_t inputs[64] = {0};
    uint32_t targets[64] = {0};
    for (size_t index = 0U; index < 64U; ++index) {
        inputs[index] = (uint32_t)((index * 17U + 11U) % 32000U);
        targets[index] = (uint32_t)((index * 19U + 7U) % 32000U);
    }
    llm_tensor *const tensors[] = {&cpu_inputs, &metal_inputs, &cpu_targets, &metal_targets,
                                   &cpu_logits, &metal_logits, &cpu_loss, &metal_loss,
                                   &cpu_gradient, &metal_gradient};
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_U32, 2U, input_shape, &cpu_inputs) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_U32, 2U, input_shape, &metal_inputs) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_U32, 1U, target_shape, &cpu_targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_U32, 1U, target_shape, &metal_targets) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, logits_shape, &cpu_logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, logits_shape, &metal_logits) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 0U, NULL, &cpu_loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 0U, NULL, &metal_loss) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, logits_shape, &cpu_gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(metal_backend, LLM_DTYPE_F32, 2U, logits_shape, &metal_gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_inputs, inputs, sizeof(inputs)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_inputs, inputs, sizeof(inputs)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_targets, targets, sizeof(targets)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(metal_backend, &metal_targets, targets, sizeof(targets)) == LLM_OK);
    TEST_ASSERT(test_model_forward_parity(cpu_backend, cpu_model, metal_backend, metal_model,
                                          &cpu_inputs, &metal_inputs, &cpu_logits, &metal_logits,
                                          targets, layer_count) == EXIT_SUCCESS);
    for (unsigned long long step = 1U; step <= 4U; ++step) {
        float cpu_loss_value = 0.0F, metal_loss_value = 0.0F;
        TEST_ASSERT(run_model_step(cpu_backend, cpu_model, &cpu_inputs, &cpu_targets, &cpu_logits,
                                   &cpu_loss, &cpu_gradient, step, &cpu_loss_value) == EXIT_SUCCESS);
        TEST_ASSERT(run_model_step(metal_backend, metal_model, &metal_inputs, &metal_targets,
                                   &metal_logits, &metal_loss, &metal_gradient, step,
                                   &metal_loss_value) == EXIT_SUCCESS);
        if (fabsf(cpu_loss_value - metal_loss_value) > 2.0e-3F) {
            fprintf(stderr, "model-step parity mismatch (layers=%zu) at step %llu: cpu=%g metal=%g\n",
                    layer_count, step, (double)cpu_loss_value, (double)metal_loss_value);
        }
        TEST_ASSERT(fabsf(cpu_loss_value - metal_loss_value) <= 2.0e-3F);
    }
    for (size_t parameter = 0U; parameter < lm_model_parameter_count(cpu_model); ++parameter) {
        const llm_tensor *cpu_value = lm_model_parameter_value(cpu_model, parameter);
        const llm_tensor *metal_value = lm_model_parameter_value(metal_model, parameter);
        float *cpu_values = calloc(cpu_value->element_count, sizeof(*cpu_values));
        float *metal_values = calloc(metal_value->element_count, sizeof(*metal_values));
        TEST_ASSERT(cpu_values != NULL && metal_values != NULL);
        TEST_ASSERT(llm_tensor_read(cpu_backend, cpu_value, cpu_values,
                                    cpu_value->element_count * sizeof(*cpu_values)) == LLM_OK);
        TEST_ASSERT(llm_tensor_read(metal_backend, metal_value, metal_values,
                                    metal_value->element_count * sizeof(*metal_values)) == LLM_OK);
        for (size_t value = 0U; value < cpu_value->element_count; ++value) {
            TEST_ASSERT(isfinite(metal_values[value]) != 0);
            TEST_ASSERT(fabsf(cpu_values[value] - metal_values[value]) <= 3.0e-3F);
        }
        free(metal_values);
        free(cpu_values);
    }
    for (size_t index = 0U; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        llm_tensor_destroy(tensors[index]);
    }
    lm_model_destroy(metal_model);
    lm_model_destroy(cpu_model);
    llm_backend_destroy(cpu_backend);
    return EXIT_SUCCESS;
}

static int test_model_training_step_parity(llm_backend *metal_backend) {
    return test_model_training_step_parity_for_layers(metal_backend, 0U) == EXIT_SUCCESS &&
                   test_model_training_step_parity_for_layers(metal_backend, 1U) == EXIT_SUCCESS
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

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

typedef int (*metal_backend_test)(llm_backend *backend);

static int run_named_metal_test(const char *name, metal_backend_test test, llm_backend *backend) {
    const int result = test(backend);
    printf("Metal verification: %-36s %s\n", name,
           result == EXIT_SUCCESS ? "PASS" : "FAIL");
    return result;
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

    int result = EXIT_SUCCESS;
    if (run_named_metal_test("memory and elementwise", test_memory_and_elementwise, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("reshape and shared storage", test_reshape_shared_storage, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("batching and buffer pool", test_batch_metrics_and_buffer_pool,
                             backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("reductions and matmul", test_reductions_and_matmul, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("matmul tile boundaries", test_matmul_tile_boundaries, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("MPS matmul in batch", test_mps_matmul_ex_in_batch, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("language operations", test_language_operations, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("analytic forward/backward contract", runtime_backend_contract_suite,
                             backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("training-shape matmul and loss",
                             test_training_shape_matmul_and_loss_parity, backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_metal_test("complete model training step", test_model_training_step_parity,
                             backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    llm_backend_destroy(backend);
    return result;
}
