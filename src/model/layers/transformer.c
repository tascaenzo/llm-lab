#include <math.h>

#include "model_internal.h"

enum {
    LM_BLOCK_PARAMETER_ATTENTION_NORM = 0,
    LM_BLOCK_PARAMETER_QUERY = 1,
    LM_BLOCK_PARAMETER_KEY = 2,
    LM_BLOCK_PARAMETER_VALUE = 3,
    LM_BLOCK_PARAMETER_ATTENTION_OUTPUT = 4,
    LM_BLOCK_PARAMETER_MLP_NORM = 5,
    LM_BLOCK_PARAMETER_GATE = 6,
    LM_BLOCK_PARAMETER_UP = 7,
    LM_BLOCK_PARAMETER_DOWN = 8
};

static int has_mlp(const lm_model *model) {
    return model != NULL && model->config.feed_forward_size != 0U;
}

static lm_model_parameter *parameter(lm_model *model, size_t layer, size_t offset) {
    return &model->parameters[model->blocks[layer].parameter_offset + offset];
}

static llm_status rows_view(const llm_tensor *tensor, size_t rows, size_t columns,
                            llm_tensor *out_view) {
    const size_t shape[] = {rows, columns};
    return llm_tensor_reshape(tensor, 2U, shape, out_view);
}

static llm_status linear_forward(lm_model *model, const llm_tensor *input, size_t input_columns,
                                 const lm_model_parameter *weight, llm_tensor *output,
                                 size_t output_columns) {
    const size_t rows = model->forward_batch_size * model->config.context_length;
    llm_tensor input_rows = {0};
    llm_tensor output_rows = {0};
    llm_status status = rows_view(input, rows, input_columns, &input_rows);
    if (status == LLM_OK)
        status = rows_view(output, rows, output_columns, &output_rows);
    if (status == LLM_OK)
        status = llm_matmul(model->backend, &input_rows, &weight->value, &output_rows);
    llm_tensor_destroy(&output_rows);
    llm_tensor_destroy(&input_rows);
    return status;
}

static llm_status linear_backward_accumulate(lm_model *model, const llm_tensor *input,
                                             size_t input_columns,
                                             const llm_tensor *output_gradient,
                                             size_t output_columns, lm_model_parameter *weight,
                                             llm_tensor *input_gradient, llm_tensor *workspace) {
    const size_t rows = model->forward_batch_size * model->config.context_length;
    llm_tensor input_rows = {0};
    llm_tensor output_gradient_rows = {0};
    llm_tensor input_gradient_rows = {0};
    const llm_matmul_options input_options = {.transpose_left = 0, .transpose_right = 1};
    const llm_matmul_options weight_options = {.transpose_left = 1, .transpose_right = 0};
    llm_status status = rows_view(input, rows, input_columns, &input_rows);
    if (status == LLM_OK)
        status = rows_view(output_gradient, rows, output_columns, &output_gradient_rows);
    if (status == LLM_OK)
        status = rows_view(input_gradient, rows, input_columns, &input_gradient_rows);
    if (status == LLM_OK) {
        status = llm_matmul_ex(model->backend, &output_gradient_rows, &weight->value,
                               &input_options, &input_gradient_rows);
    }
    if (status == LLM_OK) {
        status = llm_matmul_ex(model->backend, &input_rows, &output_gradient_rows, &weight_options,
                               workspace);
    }
    if (status == LLM_OK)
        status = llm_accumulate(model->backend, workspace, &weight->gradient);
    llm_tensor_destroy(&input_gradient_rows);
    llm_tensor_destroy(&output_gradient_rows);
    llm_tensor_destroy(&input_rows);
    return status;
}

static const llm_tensor *block_input(const lm_model *model, size_t layer) {
    return layer == 0U ? &model->hidden : &model->blocks[layer - 1U].output;
}

static llm_tensor *block_input_gradient(lm_model *model, size_t layer) {
    return layer == 0U ? &model->hidden_gradient : &model->blocks[layer - 1U].output_gradient;
}

llm_status lm_transformer_forward(lm_model *model) {
    if (model == NULL || model->config.layer_count == 0U)
        return LLM_INVALID_ARGUMENT;
    const size_t hidden = model->config.hidden_size;
    const size_t feed_forward = model->config.feed_forward_size;
    const size_t head_dimension = hidden / model->config.head_count;
    const llm_attention_options options = {.scale = 1.0F / sqrtf((float)head_dimension)};
    for (size_t layer = 0U; layer < model->config.layer_count; ++layer) {
        lm_transformer_block *block = &model->blocks[layer];
        const llm_tensor *input = block_input(model, layer);
        llm_status status =
            llm_rms_norm(model->backend, input,
                         &parameter(model, layer, LM_BLOCK_PARAMETER_ATTENTION_NORM)->value,
                         1.0e-5F, &block->attention_norm);
        if (status == LLM_OK)
            status = linear_forward(model, &block->attention_norm, hidden,
                                    parameter(model, layer, LM_BLOCK_PARAMETER_QUERY),
                                    &block->query, hidden);
        if (status == LLM_OK)
            status = linear_forward(model, &block->attention_norm, hidden,
                                    parameter(model, layer, LM_BLOCK_PARAMETER_KEY), &block->key,
                                    hidden);
        if (status == LLM_OK)
            status = linear_forward(model, &block->attention_norm, hidden,
                                    parameter(model, layer, LM_BLOCK_PARAMETER_VALUE),
                                    &block->value, hidden);
        if (status == LLM_OK)
            status = llm_rope(model->backend, &block->query, &model->rope_cos_table,
                              &model->rope_sin_table, &block->rotated_query);
        if (status == LLM_OK)
            status = llm_rope(model->backend, &block->key, &model->rope_cos_table,
                              &model->rope_sin_table, &block->rotated_key);
        if (status == LLM_OK)
            status =
                llm_attention_forward(model->backend, &block->rotated_query, &block->rotated_key,
                                      &block->value, &options, &block->attention_output);
        if (status == LLM_OK)
            status = linear_forward(model, &block->attention_output, hidden,
                                    parameter(model, layer, LM_BLOCK_PARAMETER_ATTENTION_OUTPUT),
                                    &block->attention_projection, hidden);
        if (has_mlp(model) != 0) {
            if (status == LLM_OK)
                status = llm_add(model->backend, input, &block->attention_projection,
                                 &block->attention_residual);
            if (status == LLM_OK)
                status = llm_rms_norm(model->backend, &block->attention_residual,
                                      &parameter(model, layer, LM_BLOCK_PARAMETER_MLP_NORM)->value,
                                      1.0e-5F, &block->mlp_norm);
            if (status == LLM_OK)
                status = linear_forward(model, &block->mlp_norm, hidden,
                                        parameter(model, layer, LM_BLOCK_PARAMETER_GATE),
                                        &block->gate, feed_forward);
            if (status == LLM_OK)
                status = linear_forward(model, &block->mlp_norm, hidden,
                                        parameter(model, layer, LM_BLOCK_PARAMETER_UP), &block->up,
                                        feed_forward);
            if (status == LLM_OK)
                status = llm_silu(model->backend, &block->gate, &block->silu_gate);
            if (status == LLM_OK)
                status =
                    llm_multiply(model->backend, &block->silu_gate, &block->up, &block->swiglu);
            if (status == LLM_OK)
                status = linear_forward(model, &block->swiglu, feed_forward,
                                        parameter(model, layer, LM_BLOCK_PARAMETER_DOWN),
                                        &block->mlp_projection, hidden);
            if (status == LLM_OK)
                status = llm_add(model->backend, &block->attention_residual, &block->mlp_projection,
                                 &block->output);
        } else if (status == LLM_OK) {
            status = llm_add(model->backend, input, &block->attention_projection, &block->output);
        }
        if (status != LLM_OK)
            return status;
    }
    return LLM_OK;
}

llm_status lm_transformer_backward(lm_model *model) {
    if (model == NULL || model->config.layer_count == 0U)
        return LLM_INVALID_ARGUMENT;
    const size_t hidden = model->config.hidden_size;
    const size_t feed_forward = model->config.feed_forward_size;
    const size_t head_dimension = hidden / model->config.head_count;
    const llm_attention_options options = {.scale = 1.0F / sqrtf((float)head_dimension)};
    for (size_t layer = model->config.layer_count; layer-- > 0U;) {
        lm_transformer_block *block = &model->blocks[layer];
        const llm_tensor *input = block_input(model, layer);
        llm_tensor *input_gradient = block_input_gradient(model, layer);
        llm_tensor *residual_gradient = &block->output_gradient;
        llm_status status = LLM_OK;
        if (has_mlp(model) != 0) {
            status = llm_tensor_copy(model->backend, &block->output_gradient,
                                     &block->attention_residual_gradient);
            if (status == LLM_OK)
                status = linear_backward_accumulate(
                    model, &block->swiglu, feed_forward, &block->output_gradient, hidden,
                    parameter(model, layer, LM_BLOCK_PARAMETER_DOWN), &block->swiglu_gradient,
                    &model->mlp_down_weight_gradient_workspace);
            if (status == LLM_OK)
                status = llm_multiply(model->backend, &block->swiglu_gradient, &block->up,
                                      &block->silu_gate_gradient);
            if (status == LLM_OK)
                status = llm_multiply(model->backend, &block->swiglu_gradient, &block->silu_gate,
                                      &block->up_gradient);
            if (status == LLM_OK)
                status = llm_silu_backward(model->backend, &block->gate, &block->silu_gate_gradient,
                                           &block->gate_gradient);
            if (status == LLM_OK)
                status = linear_backward_accumulate(
                    model, &block->mlp_norm, hidden, &block->up_gradient, feed_forward,
                    parameter(model, layer, LM_BLOCK_PARAMETER_UP), &block->mlp_norm_gradient,
                    &model->mlp_up_weight_gradient_workspace);
            if (status == LLM_OK)
                status = linear_backward_accumulate(
                    model, &block->mlp_norm, hidden, &block->gate_gradient, feed_forward,
                    parameter(model, layer, LM_BLOCK_PARAMETER_GATE), &block->linear_input_gradient,
                    &model->mlp_up_weight_gradient_workspace);
            if (status == LLM_OK)
                status = llm_accumulate(model->backend, &block->linear_input_gradient,
                                        &block->mlp_norm_gradient);
            if (status == LLM_OK)
                status = llm_rms_norm_backward(
                    model->backend, &block->attention_residual,
                    &parameter(model, layer, LM_BLOCK_PARAMETER_MLP_NORM)->value,
                    &block->mlp_norm_gradient, 1.0e-5F, &block->linear_input_gradient,
                    &parameter(model, layer, LM_BLOCK_PARAMETER_MLP_NORM)->gradient);
            if (status == LLM_OK)
                status = llm_accumulate(model->backend, &block->linear_input_gradient,
                                        &block->attention_residual_gradient);
            residual_gradient = &block->attention_residual_gradient;
        }
        if (status == LLM_OK)
            status = llm_tensor_copy(model->backend, residual_gradient, input_gradient);
        if (status == LLM_OK)
            status = linear_backward_accumulate(
                model, &block->attention_output, hidden, residual_gradient, hidden,
                parameter(model, layer, LM_BLOCK_PARAMETER_ATTENTION_OUTPUT),
                &block->attention_output_gradient, &model->attention_weight_gradient_workspace);
        if (status == LLM_OK)
            status = llm_attention_backward(
                model->backend, &block->rotated_query, &block->rotated_key, &block->value,
                &block->attention_output_gradient, &options, &block->rotated_query_gradient,
                &block->rotated_key_gradient, &block->value_gradient);
        if (status == LLM_OK)
            status = llm_rope_backward(model->backend, &block->rotated_query_gradient,
                                       &model->rope_cos_table, &model->rope_sin_table,
                                       &block->query_gradient);
        if (status == LLM_OK)
            status = llm_rope_backward(model->backend, &block->rotated_key_gradient,
                                       &model->rope_cos_table, &model->rope_sin_table,
                                       &block->key_gradient);
        if (status == LLM_OK)
            status = linear_backward_accumulate(
                model, &block->attention_norm, hidden, &block->query_gradient, hidden,
                parameter(model, layer, LM_BLOCK_PARAMETER_QUERY), &block->attention_norm_gradient,
                &model->attention_weight_gradient_workspace);
        if (status == LLM_OK)
            status = linear_backward_accumulate(
                model, &block->attention_norm, hidden, &block->key_gradient, hidden,
                parameter(model, layer, LM_BLOCK_PARAMETER_KEY), &block->linear_input_gradient,
                &model->attention_weight_gradient_workspace);
        if (status == LLM_OK)
            status = llm_accumulate(model->backend, &block->linear_input_gradient,
                                    &block->attention_norm_gradient);
        if (status == LLM_OK)
            status = linear_backward_accumulate(
                model, &block->attention_norm, hidden, &block->value_gradient, hidden,
                parameter(model, layer, LM_BLOCK_PARAMETER_VALUE), &block->linear_input_gradient,
                &model->attention_weight_gradient_workspace);
        if (status == LLM_OK)
            status = llm_accumulate(model->backend, &block->linear_input_gradient,
                                    &block->attention_norm_gradient);
        if (status == LLM_OK)
            status = llm_rms_norm_backward(
                model->backend, input,
                &parameter(model, layer, LM_BLOCK_PARAMETER_ATTENTION_NORM)->value,
                &block->attention_norm_gradient, 1.0e-5F, &block->linear_input_gradient,
                &parameter(model, layer, LM_BLOCK_PARAMETER_ATTENTION_NORM)->gradient);
        if (status == LLM_OK)
            status = llm_accumulate(model->backend, &block->linear_input_gradient, input_gradient);
        if (status != LLM_OK)
            return status;
    }
    return LLM_OK;
}
