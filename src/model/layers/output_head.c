#include "model_internal.h"

llm_status lm_output_head_forward(lm_model *model, llm_tensor *logits) {
    if (model == NULL || logits == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t shape[] = {model->forward_batch_size * model->config.context_length,
                            model->config.hidden_size};
    const llm_tensor *input = model->config.layer_count == 0U ? &model->hidden : &model->transformed_hidden;
    llm_tensor hidden_rows = {0};
    llm_status status = llm_tensor_reshape(input, 2U, shape, &hidden_rows);
    if (status == LLM_OK) {
        status = llm_matmul(model->backend, &hidden_rows, &model->parameters[1].value, logits);
    }
    llm_tensor_destroy(&hidden_rows);
    return status;
}

llm_status lm_output_head_backward(lm_model *model, const llm_tensor *logits_gradient) {
    if (model == NULL || logits_gradient == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t row_count = model->forward_batch_size * model->config.context_length;
    const size_t shape[] = {row_count, model->config.hidden_size};
    const llm_tensor *input = model->config.layer_count == 0U ? &model->hidden : &model->transformed_hidden;
    llm_tensor *input_gradient = model->config.layer_count == 0U ? &model->hidden_gradient
                                                                   : &model->transformed_hidden_gradient;
    llm_tensor hidden_rows = {0};
    llm_tensor hidden_gradient_rows = {0};
    const llm_matmul_options input_gradient_options = {.transpose_left = 0, .transpose_right = 1};
    const llm_matmul_options weight_gradient_options = {.transpose_left = 1, .transpose_right = 0};
    llm_status status = llm_tensor_reshape(input, 2U, shape, &hidden_rows);
    if (status == LLM_OK) {
        status = llm_tensor_reshape(input_gradient, 2U, shape, &hidden_gradient_rows);
    }
    if (status == LLM_OK) {
        status = llm_matmul_ex(model->backend, logits_gradient, &model->parameters[1].value,
                               &input_gradient_options, &hidden_gradient_rows);
    }
    if (status == LLM_OK) {
        status = llm_matmul_ex(model->backend, &hidden_rows, logits_gradient,
                               &weight_gradient_options,
                               &model->output_weight_gradient_workspace);
    }
    if (status == LLM_OK) {
        status = llm_accumulate(model->backend, &model->output_weight_gradient_workspace,
                                &model->parameters[1].gradient);
    }
    llm_tensor_destroy(&hidden_gradient_rows);
    llm_tensor_destroy(&hidden_rows);
    return status;
}
