#include <math.h>

#include "model_internal.h"

enum {
    LM_PARAMETER_EMBEDDING = 0,
    LM_PARAMETER_OUTPUT = 1,
    LM_PARAMETER_ATTENTION_NORM = 2,
    LM_PARAMETER_QUERY = 3,
    LM_PARAMETER_KEY = 4,
    LM_PARAMETER_VALUE = 5,
    LM_PARAMETER_ATTENTION_OUTPUT = 6
};

static llm_status rows_view(llm_tensor *tensor, size_t rows, size_t columns, llm_tensor *out_view) {
    const size_t shape[] = {rows, columns};
    return llm_tensor_reshape(tensor, 2U, shape, out_view);
}

static llm_status linear_forward(lm_model *model, llm_tensor *input, llm_tensor *weight,
                                 llm_tensor *output) {
    const size_t rows = model->forward_batch_size * model->config.context_length;
    llm_tensor input_rows = {0};
    llm_tensor output_rows = {0};
    llm_status status = rows_view(input, rows, model->config.hidden_size, &input_rows);
    if (status == LLM_OK) {
        status = rows_view(output, rows, model->config.hidden_size, &output_rows);
    }
    if (status == LLM_OK) {
        status = llm_matmul(model->backend, &input_rows, weight, &output_rows);
    }
    llm_tensor_destroy(&output_rows);
    llm_tensor_destroy(&input_rows);
    return status;
}

static llm_status linear_backward_accumulate(lm_model *model, llm_tensor *input,
                                             llm_tensor *output_gradient, lm_model_parameter *weight,
                                             llm_tensor *input_gradient, llm_tensor *workspace) {
    const size_t rows = model->forward_batch_size * model->config.context_length;
    llm_tensor input_rows = {0};
    llm_tensor output_gradient_rows = {0};
    llm_tensor input_gradient_rows = {0};
    const llm_matmul_options input_options = {.transpose_left = 0, .transpose_right = 1};
    const llm_matmul_options weight_options = {.transpose_left = 1, .transpose_right = 0};
    llm_status status = rows_view(input, rows, model->config.hidden_size, &input_rows);
    if (status == LLM_OK) {
        status = rows_view(output_gradient, rows, model->config.hidden_size, &output_gradient_rows);
    }
    if (status == LLM_OK) {
        status = rows_view(input_gradient, rows, model->config.hidden_size, &input_gradient_rows);
    }
    if (status == LLM_OK) {
        status = llm_matmul_ex(model->backend, &output_gradient_rows, &weight->value, &input_options,
                               &input_gradient_rows);
    }
    if (status == LLM_OK) {
        status = llm_matmul_ex(model->backend, &input_rows, &output_gradient_rows, &weight_options,
                               workspace);
    }
    if (status == LLM_OK) {
        status = llm_accumulate(model->backend, workspace, &weight->gradient);
    }
    llm_tensor_destroy(&input_gradient_rows);
    llm_tensor_destroy(&output_gradient_rows);
    llm_tensor_destroy(&input_rows);
    return status;
}

llm_status lm_transformer_forward(lm_model *model) {
    if (model == NULL || model->config.layer_count != 1U || model->parameter_count != 7U) {
        return LLM_INVALID_ARGUMENT;
    }
    lm_model_parameter *parameters = model->parameters;
    llm_status status = llm_rms_norm(model->backend, &model->hidden,
                                     &parameters[LM_PARAMETER_ATTENTION_NORM].value, 1.0e-5F,
                                     &model->attention_norm);
    if (status == LLM_OK) {
        status = linear_forward(model, &model->attention_norm, &parameters[LM_PARAMETER_QUERY].value,
                                &model->query);
    }
    if (status == LLM_OK) {
        status = linear_forward(model, &model->attention_norm, &parameters[LM_PARAMETER_KEY].value,
                                &model->key);
    }
    if (status == LLM_OK) {
        status = linear_forward(model, &model->attention_norm, &parameters[LM_PARAMETER_VALUE].value,
                                &model->value);
    }
    if (status == LLM_OK) {
        status = llm_rope(model->backend, &model->query, &model->rope_cos_table,
                          &model->rope_sin_table, &model->rotated_query);
    }
    if (status == LLM_OK) {
        status = llm_rope(model->backend, &model->key, &model->rope_cos_table,
                          &model->rope_sin_table, &model->rotated_key);
    }
    const llm_attention_options options = {.scale = 1.0F / sqrtf((float)model->config.hidden_size)};
    if (status == LLM_OK) {
        status = llm_attention_forward(model->backend, &model->rotated_query, &model->rotated_key,
                                       &model->value, &options, &model->attention_output);
    }
    if (status == LLM_OK) {
        status = linear_forward(model, &model->attention_output,
                                &parameters[LM_PARAMETER_ATTENTION_OUTPUT].value,
                                &model->attention_projection);
    }
    if (status == LLM_OK) {
        status = llm_add(model->backend, &model->hidden, &model->attention_projection,
                         &model->transformed_hidden);
    }
    return status;
}

llm_status lm_transformer_backward(lm_model *model) {
    if (model == NULL || model->config.layer_count != 1U || model->parameter_count != 7U) {
        return LLM_INVALID_ARGUMENT;
    }
    lm_model_parameter *parameters = model->parameters;
    llm_status status = llm_tensor_copy(model->backend, &model->transformed_hidden_gradient,
                                        &model->hidden_gradient);
    if (status == LLM_OK) {
        status = linear_backward_accumulate(model, &model->attention_output,
                                            &model->transformed_hidden_gradient,
                                            &parameters[LM_PARAMETER_ATTENTION_OUTPUT],
                                            &model->attention_output_gradient,
                                            &model->attention_weight_gradient_workspace);
    }
    const llm_attention_options options = {.scale = 1.0F / sqrtf((float)model->config.hidden_size)};
    if (status == LLM_OK) {
        status = llm_attention_backward(model->backend, &model->rotated_query, &model->rotated_key,
                                        &model->value, &model->attention_output_gradient, &options,
                                        &model->rotated_query_gradient, &model->rotated_key_gradient,
                                        &model->value_gradient);
    }
    if (status == LLM_OK) {
        status = llm_rope_backward(model->backend, &model->rotated_query_gradient,
                                   &model->rope_cos_table, &model->rope_sin_table,
                                   &model->query_gradient);
    }
    if (status == LLM_OK) {
        status = llm_rope_backward(model->backend, &model->rotated_key_gradient,
                                   &model->rope_cos_table, &model->rope_sin_table,
                                   &model->key_gradient);
    }
    if (status == LLM_OK) {
        status = linear_backward_accumulate(model, &model->attention_norm, &model->query_gradient,
                                            &parameters[LM_PARAMETER_QUERY],
                                            &model->attention_norm_gradient,
                                            &model->attention_weight_gradient_workspace);
    }
    if (status == LLM_OK) {
        status = linear_backward_accumulate(model, &model->attention_norm, &model->key_gradient,
                                            &parameters[LM_PARAMETER_KEY],
                                            &model->transformer_input_gradient,
                                            &model->attention_weight_gradient_workspace);
    }
    if (status == LLM_OK) {
        status = llm_accumulate(model->backend, &model->transformer_input_gradient,
                                &model->attention_norm_gradient);
    }
    if (status == LLM_OK) {
        status = linear_backward_accumulate(model, &model->attention_norm, &model->value_gradient,
                                            &parameters[LM_PARAMETER_VALUE],
                                            &model->transformer_input_gradient,
                                            &model->attention_weight_gradient_workspace);
    }
    if (status == LLM_OK) {
        status = llm_accumulate(model->backend, &model->transformer_input_gradient,
                                &model->attention_norm_gradient);
    }
    if (status == LLM_OK) {
        status = llm_rms_norm_backward(model->backend, &model->hidden,
                                       &parameters[LM_PARAMETER_ATTENTION_NORM].value,
                                       &model->attention_norm_gradient, 1.0e-5F,
                                       &model->transformer_input_gradient,
                                       &parameters[LM_PARAMETER_ATTENTION_NORM].gradient);
    }
    if (status == LLM_OK) {
        status = llm_accumulate(model->backend, &model->transformer_input_gradient,
                                &model->hidden_gradient);
    }
    return status;
}
