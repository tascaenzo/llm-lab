#include <math.h>
#include <stdlib.h>

#include "backend_contract_suite.h"
#include "runtime/backend.h"
#include "runtime/operations.h"
#include "test_support.h"

static int close_with_tolerance(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

static float dot_product(const float *left, const float *right, size_t value_count) {
    float result = 0.0F;
    for (size_t index = 0U; index < value_count; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

static int test_matmul_ex_and_accumulate(llm_backend *backend) {
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
    const llm_matmul_options options = {.transpose_left = 1, .transpose_right = 1};
    TEST_ASSERT(llm_matmul_ex(backend, &left, &right, &options, &output) == LLM_OK);
    float actual[8] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 4U; ++column) {
            float expected = 0.0F;
            for (size_t inner = 0U; inner < 3U; ++inner) {
                expected += left_values[inner * 2U + row] * right_values[column * 3U + inner];
            }
            TEST_ASSERT(actual[row * 4U + column] == expected);
        }
    }

    TEST_ASSERT(llm_tensor_fill_f32(backend, &source, 0.5F) == LLM_OK);
    TEST_ASSERT(llm_accumulate(backend, &source, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, actual, sizeof(actual)) == LLM_OK);
    TEST_ASSERT(close_with_tolerance(actual[0], 22.5F, 1.0e-6F));
    TEST_ASSERT(llm_accumulate(backend, &output, &output) == LLM_INVALID_ARGUMENT);

    llm_tensor_destroy(&source);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    return EXIT_SUCCESS;
}

static float silu_objective(llm_backend *backend, llm_tensor *input, llm_tensor *output,
                            const float *input_values, const float *output_gradient,
                            size_t value_count) {
    float output_values[8] = {0};
    if (llm_tensor_write(backend, input, input_values, value_count * sizeof(float)) != LLM_OK ||
        llm_silu(backend, input, output) != LLM_OK ||
        llm_tensor_read(backend, output, output_values, value_count * sizeof(float)) != LLM_OK) {
        return NAN;
    }
    return dot_product(output_values, output_gradient, value_count);
}

static int test_silu_gradient(llm_backend *backend) {
    const size_t shape[] = {5U};
    llm_tensor input = {0};
    llm_tensor output = {0};
    llm_tensor output_gradient = {0};
    llm_tensor input_gradient = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &output_gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &input_gradient) == LLM_OK);
    float input_values[] = {-3.0F, -0.5F, 0.0F, 1.0F, 4.0F};
    const float output_gradient_values[] = {0.5F, -1.0F, 2.0F, 0.25F, -0.75F};
    TEST_ASSERT(llm_tensor_write(backend, &input, input_values, sizeof(input_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &output_gradient, output_gradient_values,
                                 sizeof(output_gradient_values)) == LLM_OK);
    TEST_ASSERT(llm_silu_backward(backend, &input, &output_gradient, &input_gradient) == LLM_OK);
    float analytic[5] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &input_gradient, analytic, sizeof(analytic)) == LLM_OK);
    const float epsilon = 1.0e-3F;
    for (size_t index = 0U; index < 5U; ++index) {
        const float original = input_values[index];
        input_values[index] = original + epsilon;
        const float positive =
            silu_objective(backend, &input, &output, input_values, output_gradient_values, 5U);
        input_values[index] = original - epsilon;
        const float negative =
            silu_objective(backend, &input, &output, input_values, output_gradient_values, 5U);
        input_values[index] = original;
        TEST_ASSERT(close_with_tolerance(analytic[index], (positive - negative) / (2.0F * epsilon),
                                         5.0e-4F));
    }

    llm_tensor_destroy(&input_gradient);
    llm_tensor_destroy(&output_gradient);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&input);
    return EXIT_SUCCESS;
}

static float rms_objective(llm_backend *backend, llm_tensor *input, llm_tensor *weight,
                           llm_tensor *output, const float *input_values,
                           const float *weight_values, const float *output_gradient) {
    float output_values[6] = {0};
    if (llm_tensor_write(backend, input, input_values, sizeof(output_values)) != LLM_OK ||
        llm_tensor_write(backend, weight, weight_values, 3U * sizeof(float)) != LLM_OK ||
        llm_rms_norm(backend, input, weight, 1.0e-5F, output) != LLM_OK ||
        llm_tensor_read(backend, output, output_values, sizeof(output_values)) != LLM_OK) {
        return NAN;
    }
    return dot_product(output_values, output_gradient, 6U);
}

static int test_rms_norm_gradients(llm_backend *backend) {
    const size_t input_shape[] = {2U, 3U};
    const size_t weight_shape[] = {3U};
    llm_tensor input = {0};
    llm_tensor weight = {0};
    llm_tensor output = {0};
    llm_tensor output_gradient = {0};
    llm_tensor input_gradient = {0};
    llm_tensor weight_gradient = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, input_shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, weight_shape, &weight) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, input_shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, input_shape, &output_gradient) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, input_shape, &input_gradient) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, weight_shape, &weight_gradient) ==
                LLM_OK);
    float input_values[] = {0.5F, -1.0F, 2.0F, -0.25F, 0.75F, 1.5F};
    float weight_values[] = {1.0F, 0.75F, -0.5F};
    const float output_gradient_values[] = {0.2F, -0.3F, 0.5F, -0.7F, 0.1F, 0.4F};
    TEST_ASSERT(llm_tensor_write(backend, &input, input_values, sizeof(input_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &weight, weight_values, sizeof(weight_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &output_gradient, output_gradient_values,
                                 sizeof(output_gradient_values)) == LLM_OK);
    TEST_ASSERT(llm_rms_norm_backward(backend, &input, &weight, &output_gradient, 1.0e-5F,
                                      &input_gradient, &weight_gradient) == LLM_OK);
    float analytic_input[6] = {0};
    float analytic_weight[3] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &input_gradient, analytic_input, sizeof(analytic_input)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &weight_gradient, analytic_weight,
                                sizeof(analytic_weight)) == LLM_OK);
    const float epsilon = 1.0e-3F;
    for (size_t index = 0U; index < 6U; ++index) {
        const float original = input_values[index];
        input_values[index] = original + epsilon;
        const float positive = rms_objective(backend, &input, &weight, &output, input_values,
                                             weight_values, output_gradient_values);
        input_values[index] = original - epsilon;
        const float negative = rms_objective(backend, &input, &weight, &output, input_values,
                                             weight_values, output_gradient_values);
        input_values[index] = original;
        TEST_ASSERT(close_with_tolerance(analytic_input[index],
                                         (positive - negative) / (2.0F * epsilon), 1.0e-3F));
    }
    for (size_t index = 0U; index < 3U; ++index) {
        const float original = weight_values[index];
        weight_values[index] = original + epsilon;
        const float positive = rms_objective(backend, &input, &weight, &output, input_values,
                                             weight_values, output_gradient_values);
        weight_values[index] = original - epsilon;
        const float negative = rms_objective(backend, &input, &weight, &output, input_values,
                                             weight_values, output_gradient_values);
        weight_values[index] = original;
        TEST_ASSERT(close_with_tolerance(analytic_weight[index],
                                         (positive - negative) / (2.0F * epsilon), 1.0e-3F));
    }

    llm_tensor_destroy(&weight_gradient);
    llm_tensor_destroy(&input_gradient);
    llm_tensor_destroy(&output_gradient);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&weight);
    llm_tensor_destroy(&input);
    return EXIT_SUCCESS;
}

static int test_rope_forward_backward(llm_backend *backend) {
    const size_t input_shape[] = {1U, 2U, 1U, 4U};
    const size_t table_shape[] = {2U, 2U};
    llm_tensor input = {0};
    llm_tensor rotated = {0};
    llm_tensor recovered = {0};
    llm_tensor cosine = {0};
    llm_tensor sine = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, input_shape, &input) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, input_shape, &rotated) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, input_shape, &recovered) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &cosine) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &sine) == LLM_OK);
    const float input_values[] = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, 0.5F, 2.0F, -3.0F};
    float cosine_values[4] = {0};
    float sine_values[4] = {0};
    for (size_t position = 0U; position < 2U; ++position) {
        for (size_t pair = 0U; pair < 2U; ++pair) {
            const float angle = (float)(position + pair) * 0.2F;
            cosine_values[position * 2U + pair] = cosf(angle);
            sine_values[position * 2U + pair] = sinf(angle);
        }
    }
    TEST_ASSERT(llm_tensor_write(backend, &input, input_values, sizeof(input_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &cosine, cosine_values, sizeof(cosine_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &sine, sine_values, sizeof(sine_values)) == LLM_OK);
    TEST_ASSERT(llm_rope(backend, &input, &cosine, &sine, &rotated) == LLM_OK);
    TEST_ASSERT(llm_rope_backward(backend, &rotated, &cosine, &sine, &recovered) == LLM_OK);
    float recovered_values[8] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &recovered, recovered_values, sizeof(recovered_values)) ==
                LLM_OK);
    for (size_t index = 0U; index < 8U; ++index) {
        TEST_ASSERT(close_with_tolerance(recovered_values[index], input_values[index], 1.0e-5F));
    }
    const size_t invalid_table_shape[] = {3U, 2U};
    llm_tensor invalid_cosine = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, invalid_table_shape,
                                  &invalid_cosine) == LLM_OK);
    TEST_ASSERT(llm_rope(backend, &input, &invalid_cosine, &sine, &rotated) == LLM_INVALID_SHAPE);
    llm_tensor_destroy(&invalid_cosine);
    llm_tensor_destroy(&sine);
    llm_tensor_destroy(&cosine);
    llm_tensor_destroy(&recovered);
    llm_tensor_destroy(&rotated);
    llm_tensor_destroy(&input);
    return EXIT_SUCCESS;
}

static float attention_objective(llm_backend *backend, llm_tensor *query, llm_tensor *key,
                                 llm_tensor *value, llm_tensor *output,
                                 const llm_attention_options *options, const float *query_values,
                                 const float *key_values, const float *value_values,
                                 const float *output_gradient_values) {
    float output_values[8] = {0};
    if (llm_tensor_write(backend, query, query_values, sizeof(output_values)) != LLM_OK ||
        llm_tensor_write(backend, key, key_values, 4U * sizeof(float)) != LLM_OK ||
        llm_tensor_write(backend, value, value_values, 4U * sizeof(float)) != LLM_OK ||
        llm_attention_forward(backend, query, key, value, options, output) != LLM_OK ||
        llm_tensor_read(backend, output, output_values, sizeof(output_values)) != LLM_OK) {
        return NAN;
    }
    return dot_product(output_values, output_gradient_values, 8U);
}

static int test_attention_gradients(llm_backend *backend) {
    const size_t query_shape[] = {1U, 2U, 2U, 2U};
    const size_t key_value_shape[] = {1U, 2U, 1U, 2U};
    llm_tensor query = {0};
    llm_tensor key = {0};
    llm_tensor value = {0};
    llm_tensor output = {0};
    llm_tensor output_gradient = {0};
    llm_tensor query_gradient = {0};
    llm_tensor key_gradient = {0};
    llm_tensor value_gradient = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, query_shape, &query) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, key_value_shape, &key) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, key_value_shape, &value) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, query_shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, query_shape, &output_gradient) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, query_shape, &query_gradient) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, key_value_shape, &key_gradient) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, key_value_shape, &value_gradient) ==
                LLM_OK);
    float query_values[] = {0.2F, -0.1F, 0.4F, 0.3F, -0.2F, 0.5F, 0.1F, -0.4F};
    float key_values[] = {0.3F, -0.2F, -0.1F, 0.6F};
    float value_values[] = {1.0F, -0.5F, 0.25F, 0.75F};
    const float output_gradient_values[] = {0.5F, -0.25F, -0.3F, 0.1F, 0.2F, 0.4F, -0.1F, 0.6F};
    const llm_attention_options options = {.scale = 0.70710678F};
    TEST_ASSERT(llm_tensor_write(backend, &query, query_values, sizeof(query_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &key, key_values, sizeof(key_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &value, value_values, sizeof(value_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &output_gradient, output_gradient_values,
                                 sizeof(output_gradient_values)) == LLM_OK);
    TEST_ASSERT(llm_attention_forward(backend, &query, &key, &value, &options, &output) == LLM_OK);
    float output_values[8] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, output_values, sizeof(output_values)) == LLM_OK);
    TEST_ASSERT(output_values[0] == value_values[0] && output_values[1] == value_values[1]);
    TEST_ASSERT(output_values[2] == value_values[0] && output_values[3] == value_values[1]);
    const float first_position[] = {output_values[0], output_values[1], output_values[2],
                                    output_values[3]};
    value_values[2] = 1000.0F;
    value_values[3] = -1000.0F;
    TEST_ASSERT(llm_tensor_write(backend, &value, value_values, sizeof(value_values)) == LLM_OK);
    TEST_ASSERT(llm_attention_forward(backend, &query, &key, &value, &options, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, output_values, sizeof(output_values)) == LLM_OK);
    for (size_t index = 0U; index < 4U; ++index) {
        TEST_ASSERT(output_values[index] == first_position[index]);
    }
    value_values[2] = 0.25F;
    value_values[3] = 0.75F;
    TEST_ASSERT(llm_tensor_write(backend, &value, value_values, sizeof(value_values)) == LLM_OK);
    TEST_ASSERT(llm_attention_backward(backend, &query, &key, &value, &output_gradient, &options,
                                       &query_gradient, &key_gradient, &value_gradient) == LLM_OK);
    float analytic_query[8] = {0};
    float analytic_key[4] = {0};
    float analytic_value[4] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &query_gradient, analytic_query, sizeof(analytic_query)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &key_gradient, analytic_key, sizeof(analytic_key)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &value_gradient, analytic_value, sizeof(analytic_value)) ==
                LLM_OK);
    const float epsilon = 1.0e-3F;
    for (size_t index = 0U; index < 8U; ++index) {
        const float original = query_values[index];
        query_values[index] = original + epsilon;
        const float positive =
            attention_objective(backend, &query, &key, &value, &output, &options, query_values,
                                key_values, value_values, output_gradient_values);
        query_values[index] = original - epsilon;
        const float negative =
            attention_objective(backend, &query, &key, &value, &output, &options, query_values,
                                key_values, value_values, output_gradient_values);
        query_values[index] = original;
        TEST_ASSERT(close_with_tolerance(analytic_query[index],
                                         (positive - negative) / (2.0F * epsilon), 2.0e-3F));
    }
    for (size_t index = 0U; index < 4U; ++index) {
        const float original = key_values[index];
        key_values[index] = original + epsilon;
        const float positive =
            attention_objective(backend, &query, &key, &value, &output, &options, query_values,
                                key_values, value_values, output_gradient_values);
        key_values[index] = original - epsilon;
        const float negative =
            attention_objective(backend, &query, &key, &value, &output, &options, query_values,
                                key_values, value_values, output_gradient_values);
        key_values[index] = original;
        TEST_ASSERT(close_with_tolerance(analytic_key[index],
                                         (positive - negative) / (2.0F * epsilon), 2.0e-3F));
    }
    for (size_t index = 0U; index < 4U; ++index) {
        const float original = value_values[index];
        value_values[index] = original + epsilon;
        const float positive =
            attention_objective(backend, &query, &key, &value, &output, &options, query_values,
                                key_values, value_values, output_gradient_values);
        value_values[index] = original - epsilon;
        const float negative =
            attention_objective(backend, &query, &key, &value, &output, &options, query_values,
                                key_values, value_values, output_gradient_values);
        value_values[index] = original;
        TEST_ASSERT(close_with_tolerance(analytic_value[index],
                                         (positive - negative) / (2.0F * epsilon), 2.0e-3F));
    }

    const size_t mismatched_shape[] = {1U, 3U, 1U, 2U};
    llm_tensor mismatched_key = {0};
    llm_tensor mismatched_value = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, mismatched_shape, &mismatched_key) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 4U, mismatched_shape,
                                  &mismatched_value) == LLM_OK);
    TEST_ASSERT(llm_attention_forward(backend, &query, &mismatched_key, &mismatched_value, &options,
                                      &output) == LLM_INVALID_SHAPE);
    llm_tensor_destroy(&mismatched_value);
    llm_tensor_destroy(&mismatched_key);

    llm_tensor_destroy(&value_gradient);
    llm_tensor_destroy(&key_gradient);
    llm_tensor_destroy(&query_gradient);
    llm_tensor_destroy(&output_gradient);
    llm_tensor_destroy(&output);
    llm_tensor_destroy(&value);
    llm_tensor_destroy(&key);
    llm_tensor_destroy(&query);
    return EXIT_SUCCESS;
}

static int test_adamw(llm_backend *backend) {
    const size_t shape[] = {2U};
    llm_tensor parameter = {0};
    llm_tensor gradient = {0};
    llm_tensor first_moment = {0};
    llm_tensor second_moment = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &parameter) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &gradient) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &first_moment) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &second_moment) == LLM_OK);
    const float parameter_values[] = {1.0F, -2.0F};
    const float gradient_values[] = {2.0F, -4.0F};
    TEST_ASSERT(llm_tensor_write(backend, &parameter, parameter_values, sizeof(parameter_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &gradient, gradient_values, sizeof(gradient_values)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_zero(backend, &first_moment) == LLM_OK);
    TEST_ASSERT(llm_tensor_zero(backend, &second_moment) == LLM_OK);
    const llm_adamw_options options = {
        .learning_rate = 0.01F,
        .beta1 = 0.9F,
        .beta2 = 0.999F,
        .epsilon = 1.0e-8F,
        .weight_decay = 0.1F,
        .gradient_scale = 0.5F,
        .step = 1ULL,
    };
    TEST_ASSERT(llm_adamw_update(backend, &parameter, &gradient, &first_moment, &second_moment,
                                 &options) == LLM_OK);
    float actual_parameter[2] = {0};
    float actual_first[2] = {0};
    float actual_second[2] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &parameter, actual_parameter, sizeof(actual_parameter)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &first_moment, actual_first, sizeof(actual_first)) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &second_moment, actual_second, sizeof(actual_second)) ==
                LLM_OK);
    TEST_ASSERT(close_with_tolerance(actual_parameter[0], 0.989F, 1.0e-5F));
    TEST_ASSERT(close_with_tolerance(actual_parameter[1], -1.988F, 1.0e-5F));
    TEST_ASSERT(close_with_tolerance(actual_first[0], 0.1F, 1.0e-6F));
    TEST_ASSERT(close_with_tolerance(actual_first[1], -0.2F, 1.0e-6F));
    TEST_ASSERT(close_with_tolerance(actual_second[0], 0.001F, 1.0e-6F));
    TEST_ASSERT(close_with_tolerance(actual_second[1], 0.004F, 1.0e-6F));

    llm_tensor_destroy(&second_moment);
    llm_tensor_destroy(&first_moment);
    llm_tensor_destroy(&gradient);
    llm_tensor_destroy(&parameter);
    return EXIT_SUCCESS;
}

int runtime_backend_contract_suite(llm_backend *backend) {
    if (backend == NULL || test_matmul_ex_and_accumulate(backend) != EXIT_SUCCESS ||
        test_silu_gradient(backend) != EXIT_SUCCESS ||
        test_rms_norm_gradients(backend) != EXIT_SUCCESS ||
        test_rope_forward_backward(backend) != EXIT_SUCCESS ||
        test_attention_gradients(backend) != EXIT_SUCCESS || test_adamw(backend) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
