/*
 * CUDA backend verification.
 *
 * The test skips cleanly on a machine without a CUDA device, which is the normal
 * case on the Apple Silicon workstation: only the cloud GPU host runs it for
 * real. What it checks is the same contract the CPU and Metal backends satisfy,
 * plus parity with the CPU reference on a complete decoder forward pass.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend_contract_suite.h"
#include "model/model.h"
#include "runtime/backend.h"
#include "runtime/operations.h"
#include "test_support.h"

/* Reductions run in a different order on the GPU, so parity is a tolerance and
   never bit equality. The bound matches the one used for Metal. */
#define CUDA_PARITY_TOLERANCE 2.0e-3F

static int close_enough(float left, float right) { return fabsf(left - right) < 2.0e-5F; }

static int compare_backend_tensors(llm_backend *cpu_backend, const llm_tensor *cpu_tensor,
                                   llm_backend *cuda_backend, const llm_tensor *cuda_tensor,
                                   const char *stage, float tolerance) {
    if (cpu_tensor == NULL || cuda_tensor == NULL ||
        cpu_tensor->element_count != cuda_tensor->element_count) {
        return EXIT_FAILURE;
    }
    float *cpu_values = calloc(cpu_tensor->element_count, sizeof(*cpu_values));
    float *cuda_values = calloc(cuda_tensor->element_count, sizeof(*cuda_values));
    if (cpu_values == NULL || cuda_values == NULL ||
        llm_tensor_read(cpu_backend, cpu_tensor, cpu_values,
                        cpu_tensor->element_count * sizeof(*cpu_values)) != LLM_OK ||
        llm_tensor_read(cuda_backend, cuda_tensor, cuda_values,
                        cuda_tensor->element_count * sizeof(*cuda_values)) != LLM_OK) {
        free(cuda_values);
        free(cpu_values);
        return EXIT_FAILURE;
    }
    float maximum_error = 0.0F;
    size_t maximum_index = 0U;
    for (size_t index = 0U; index < cpu_tensor->element_count; ++index) {
        const float error = fabsf(cpu_values[index] - cuda_values[index]);
        if (isfinite(cuda_values[index]) == 0 || error > maximum_error) {
            maximum_error = error;
            maximum_index = index;
        }
    }
    const int matches = isfinite(cuda_values[maximum_index]) != 0 && maximum_error <= tolerance;
    if (matches == 0) {
        fprintf(stderr, "%s parity mismatch: max_abs=%g at %zu (cpu=%g cuda=%g), tolerance=%g\n",
                stage, (double)maximum_error, maximum_index, (double)cpu_values[maximum_index],
                (double)cuda_values[maximum_index], (double)tolerance);
    }
    free(cuda_values);
    free(cpu_values);
    return matches != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int test_memory_and_elementwise(llm_backend *backend) {
    const size_t shape[] = {2U, 3U};
    llm_tensor left = {0}, right = {0}, output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, shape, &output) == LLM_OK);

    const float left_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float right_values[] = {0.5F, 0.5F, 0.5F, 2.0F, 2.0F, 2.0F};
    TEST_ASSERT(llm_tensor_write(backend, &left, left_values, sizeof(left_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &right, right_values, sizeof(right_values)) == LLM_OK);

    float read_back[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &left, read_back, sizeof(read_back)) == LLM_OK);
    for (size_t index = 0U; index < 6U; ++index) {
        TEST_ASSERT(close_enough(read_back[index], left_values[index]));
    }

    TEST_ASSERT(llm_add(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, read_back, sizeof(read_back)) == LLM_OK);
    TEST_ASSERT(close_enough(read_back[0], 1.5F) && close_enough(read_back[5], 8.0F));

    TEST_ASSERT(llm_multiply(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, read_back, sizeof(read_back)) == LLM_OK);
    TEST_ASSERT(close_enough(read_back[0], 0.5F) && close_enough(read_back[5], 12.0F));

    TEST_ASSERT(llm_tensor_fill_f32(backend, &output, 3.0F) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, read_back, sizeof(read_back)) == LLM_OK);
    TEST_ASSERT(close_enough(read_back[3], 3.0F));

    TEST_ASSERT(llm_tensor_zero(backend, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &output, read_back, sizeof(read_back)) == LLM_OK);
    TEST_ASSERT(close_enough(read_back[3], 0.0F));

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    return EXIT_SUCCESS;
}

/* cuBLAS is column-major and the runtime is row-major, so the transposed forms
   are the ones most likely to be wired up wrong. All four are checked. */
static int test_matmul_transposes(llm_backend *backend) {
    const size_t a_shape[] = {2U, 3U};
    const size_t b_shape[] = {3U, 2U};
    const size_t c_shape[] = {2U, 2U};
    llm_tensor a = {0}, b = {0}, c = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, a_shape, &a) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, b_shape, &b) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, c_shape, &c) == LLM_OK);

    /* A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]] */
    const float a_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float b_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    TEST_ASSERT(llm_tensor_write(backend, &a, a_values, sizeof(a_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &b, b_values, sizeof(b_values)) == LLM_OK);

    float result[4] = {0};
    TEST_ASSERT(llm_matmul(backend, &a, &b, &c) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &c, result, sizeof(result)) == LLM_OK);
    /* A.B = [[22,28],[49,64]] */
    TEST_ASSERT(close_enough(result[0], 22.0F) && close_enough(result[1], 28.0F));
    TEST_ASSERT(close_enough(result[2], 49.0F) && close_enough(result[3], 64.0F));

    /* A.A^T = [[14,32],[32,77]] */
    const llm_matmul_options transpose_right = {.transpose_left = 0, .transpose_right = 1};
    TEST_ASSERT(llm_matmul_ex(backend, &a, &a, &transpose_right, &c) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &c, result, sizeof(result)) == LLM_OK);
    TEST_ASSERT(close_enough(result[0], 14.0F) && close_enough(result[1], 32.0F));
    TEST_ASSERT(close_enough(result[2], 32.0F) && close_enough(result[3], 77.0F));

    /* B^T.B = [[35,44],[44,56]] */
    const llm_matmul_options transpose_left = {.transpose_left = 1, .transpose_right = 0};
    TEST_ASSERT(llm_matmul_ex(backend, &b, &b, &transpose_left, &c) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &c, result, sizeof(result)) == LLM_OK);
    TEST_ASSERT(close_enough(result[0], 35.0F) && close_enough(result[1], 44.0F));
    TEST_ASSERT(close_enough(result[2], 44.0F) && close_enough(result[3], 56.0F));

    /* B^T.A^T = (A.B)^T = [[22,49],[28,64]] */
    const llm_matmul_options transpose_both = {.transpose_left = 1, .transpose_right = 1};
    TEST_ASSERT(llm_matmul_ex(backend, &b, &a, &transpose_both, &c) == LLM_OK);
    TEST_ASSERT(llm_tensor_read(backend, &c, result, sizeof(result)) == LLM_OK);
    TEST_ASSERT(close_enough(result[0], 22.0F) && close_enough(result[1], 49.0F));
    TEST_ASSERT(close_enough(result[2], 28.0F) && close_enough(result[3], 64.0F));

    llm_tensor_destroy(&c);
    llm_tensor_destroy(&b);
    llm_tensor_destroy(&a);
    return EXIT_SUCCESS;
}

static int test_language_operations(llm_backend *backend) {
    const size_t logits_shape[] = {2U, 3U};
    const size_t targets_shape[] = {2U};
    llm_tensor logits = {0}, probabilities = {0}, targets = {0}, loss = {0}, gradient = {0};
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

    /* A non-finite input must surface as a status, not as a NaN in the output. */
    const float invalid_logits[] = {NAN, 0.0F, 1.0F, 1.0F, 2.0F, 3.0F};
    TEST_ASSERT(llm_tensor_write(backend, &logits, invalid_logits, sizeof(invalid_logits)) ==
                LLM_OK);
    TEST_ASSERT(llm_softmax_last(backend, &logits, &probabilities) == LLM_NUMERICAL_ERROR);

    llm_tensor_destroy(&gradient);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&probabilities);
    llm_tensor_destroy(&logits);
    return EXIT_SUCCESS;
}

/* An out-of-range index must be rejected rather than corrupting the table. */
static int test_index_validation(llm_backend *backend) {
    const size_t table_shape[] = {4U, 2U};
    const size_t index_shape[] = {3U};
    const size_t output_shape[] = {3U, 2U};
    llm_tensor table = {0}, indices = {0}, output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, table_shape, &table) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_U32, 1U, index_shape, &indices) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape, &output) == LLM_OK);

    const float table_values[] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    const uint32_t valid[] = {0U, 3U, 1U};
    const uint32_t invalid[] = {0U, 4U, 1U};
    TEST_ASSERT(llm_tensor_write(backend, &table, table_values, sizeof(table_values)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(backend, &indices, valid, sizeof(valid)) == LLM_OK);
    TEST_ASSERT(llm_gather_rows(backend, &table, &indices, &output) == LLM_OK);
    float gathered[6] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, gathered, sizeof(gathered)) == LLM_OK);
    TEST_ASSERT(close_enough(gathered[2], 6.0F) && close_enough(gathered[3], 7.0F));

    TEST_ASSERT(llm_tensor_write(backend, &indices, invalid, sizeof(invalid)) == LLM_OK);
    TEST_ASSERT(llm_gather_rows(backend, &table, &indices, &output) == LLM_INVALID_INDEX);

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&indices);
    llm_tensor_destroy(&table);
    return EXIT_SUCCESS;
}

/* Inside a batch nothing synchronizes until the batch closes, so this checks that
   queued work is still ordered and observable afterwards. */
static int test_batch_and_metrics(llm_backend *backend) {
    llm_cuda_backend_metrics metrics = {0};
    TEST_ASSERT(llm_backend_cuda_reset_metrics(backend) == LLM_OK);
    TEST_ASSERT(llm_backend_cuda_end_batch(backend) == LLM_INVALID_ARGUMENT);

    const size_t shape[] = {64U};
    llm_tensor left = {0}, right = {0}, output = {0};
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &left) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &right) == LLM_OK);
    TEST_ASSERT(llm_tensor_create(backend, LLM_DTYPE_F32, 1U, shape, &output) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &left, 1.5F) == LLM_OK);
    TEST_ASSERT(llm_tensor_fill_f32(backend, &right, 2.5F) == LLM_OK);

    TEST_ASSERT(llm_backend_cuda_begin_batch(backend) == LLM_OK);
    TEST_ASSERT(llm_backend_cuda_begin_batch(backend) == LLM_INVALID_ARGUMENT);
    TEST_ASSERT(llm_add(backend, &left, &right, &output) == LLM_OK);
    TEST_ASSERT(llm_accumulate(backend, &left, &output) == LLM_OK);
    TEST_ASSERT(llm_backend_cuda_end_batch(backend) == LLM_OK);

    float values[64] = {0};
    TEST_ASSERT(llm_tensor_read(backend, &output, values, sizeof(values)) == LLM_OK);
    TEST_ASSERT(close_enough(values[0], 5.5F) && close_enough(values[63], 5.5F));

    TEST_ASSERT(llm_backend_cuda_get_metrics(backend, &metrics) == LLM_OK);
    TEST_ASSERT(metrics.kernel_launches > 0U);
    TEST_ASSERT(metrics.active_buffer_bytes >= 3U * 64U * sizeof(float));

    llm_tensor_destroy(&output);
    llm_tensor_destroy(&right);
    llm_tensor_destroy(&left);
    return EXIT_SUCCESS;
}

/* The full decoder forward pass, compared against the CPU reference. This is the
   check that catches a wrong attention, RoPE or RMSNorm port, because every one
   of those feeds the logits. */
static int test_model_forward_parity(llm_backend *cuda_backend) {
    const lm_model_config config = {.vocabulary_size = 1024U,
                                    .context_length = 16U,
                                    .hidden_size = 32U,
                                    .layer_count = 2U,
                                    .head_count = 4U,
                                    .feed_forward_size = 64U,
                                    .seed = UINT64_C(31)};
    llm_backend *cpu_backend = NULL;
    lm_model *cpu_model = NULL;
    lm_model *cuda_model = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&cpu_backend) == LLM_OK);
    TEST_ASSERT(lm_model_create(cpu_backend, &config, &cpu_model) == LLM_OK);
    TEST_ASSERT(lm_model_create(cuda_backend, &config, &cuda_model) == LLM_OK);
    TEST_ASSERT(lm_model_parameter_count(cpu_model) == lm_model_parameter_count(cuda_model));

    /* Both models seed their parameters identically, so any drift below is the
       backend's doing and not a different initialization. */
    for (size_t parameter = 0U; parameter < lm_model_parameter_count(cpu_model); ++parameter) {
        char stage[128] = {0};
        (void)snprintf(stage, sizeof(stage), "initial parameter %s",
                       lm_model_parameter_name(cpu_model, parameter));
        TEST_ASSERT(
            compare_backend_tensors(cpu_backend, lm_model_parameter_value(cpu_model, parameter),
                                    cuda_backend, lm_model_parameter_value(cuda_model, parameter),
                                    stage, 0.0F) == EXIT_SUCCESS);
    }

    const size_t input_shape[] = {2U, 16U};
    const size_t logits_shape[] = {32U, 1024U};
    llm_tensor cpu_inputs = {0}, cuda_inputs = {0}, cpu_logits = {0}, cuda_logits = {0};
    uint32_t inputs[32] = {0};
    for (size_t index = 0U; index < 32U; ++index) {
        inputs[index] = (uint32_t)((index * 17U + 11U) % 1024U);
    }
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_U32, 2U, input_shape, &cpu_inputs) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cuda_backend, LLM_DTYPE_U32, 2U, input_shape, &cuda_inputs) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cpu_backend, LLM_DTYPE_F32, 2U, logits_shape, &cpu_logits) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_create(cuda_backend, LLM_DTYPE_F32, 2U, logits_shape, &cuda_logits) ==
                LLM_OK);
    TEST_ASSERT(llm_tensor_write(cpu_backend, &cpu_inputs, inputs, sizeof(inputs)) == LLM_OK);
    TEST_ASSERT(llm_tensor_write(cuda_backend, &cuda_inputs, inputs, sizeof(inputs)) == LLM_OK);

    TEST_ASSERT(lm_model_forward(cpu_model, &cpu_inputs, &cpu_logits) == LLM_OK);
    TEST_ASSERT(lm_model_forward(cuda_model, &cuda_inputs, &cuda_logits) == LLM_OK);
    const int parity = compare_backend_tensors(cpu_backend, &cpu_logits, cuda_backend, &cuda_logits,
                                               "decoder logits", CUDA_PARITY_TOLERANCE);

    llm_tensor_destroy(&cuda_logits);
    llm_tensor_destroy(&cpu_logits);
    llm_tensor_destroy(&cuda_inputs);
    llm_tensor_destroy(&cpu_inputs);
    lm_model_destroy(cuda_model);
    lm_model_destroy(cpu_model);
    llm_backend_destroy(cpu_backend);
    return parity;
}

typedef int (*cuda_backend_test)(llm_backend *backend);

static int run_named_cuda_test(const char *name, cuda_backend_test test, llm_backend *backend) {
    const int result = test(backend);
    printf("CUDA verification: %-36s %s\n", name, result == EXIT_SUCCESS ? "PASS" : "FAIL");
    return result;
}

int main(void) {
    TEST_ASSERT(llm_backend_cuda_create(NULL) == LLM_INVALID_ARGUMENT);

    llm_backend *backend = NULL;
    if (llm_backend_cuda_is_available() == 0) {
        TEST_ASSERT(llm_backend_cuda_create(&backend) == LLM_UNSUPPORTED_DEVICE);
        TEST_ASSERT(backend == NULL);
        printf("CUDA device unavailable: backend execution skipped\n");
        return EXIT_SUCCESS;
    }

    TEST_ASSERT(llm_backend_cuda_create(&backend) == LLM_OK);
    TEST_ASSERT(backend != NULL);
    TEST_ASSERT(llm_backend_device(backend) == LLM_DEVICE_CUDA);
    TEST_ASSERT(llm_backend_cuda_device_name(backend) != NULL);
    TEST_ASSERT(llm_backend_synchronize(backend) == LLM_OK);
    printf("CUDA device: %s\n", llm_backend_cuda_device_name(backend));

    llm_backend *cpu = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&cpu) == LLM_OK);
    TEST_ASSERT(llm_backend_cuda_device_name(cpu) == NULL);
    TEST_ASSERT(llm_backend_begin_batch(cpu) == LLM_OK);
    TEST_ASSERT(llm_backend_end_batch(cpu) == LLM_OK);
    llm_backend_destroy(cpu);

    int result = EXIT_SUCCESS;
    if (run_named_cuda_test("memory and elementwise", test_memory_and_elementwise, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_cuda_test("matmul transposes", test_matmul_transposes, backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_cuda_test("language operations", test_language_operations, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_cuda_test("index validation", test_index_validation, backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_cuda_test("batching and metrics", test_batch_and_metrics, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_cuda_test("analytic forward/backward contract", runtime_backend_contract_suite,
                            backend) != EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    if (run_named_cuda_test("decoder forward parity", test_model_forward_parity, backend) !=
        EXIT_SUCCESS) {
        result = EXIT_FAILURE;
    }
    llm_backend_destroy(backend);
    return result;
}
