#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"

enum {
    LM_PARAMETER_EMBEDDING = 0,
    LM_PARAMETER_OUTPUT = 1,
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

static int has_mlp(const lm_model_config *config) {
    return config != NULL && config->feed_forward_size != 0U;
}

static size_t block_parameter_count(const lm_model_config *config) {
    return has_mlp(config) != 0 ? 9U : 5U;
}

static int config_is_valid(const lm_model_config *config) {
    if (config == NULL || config->vocabulary_size == 0U || config->context_length == 0U ||
        config->hidden_size == 0U) {
        return 0;
    }
    if (config->layer_count == 0U) {
        return config->head_count == 0U && config->feed_forward_size == 0U;
    }
    if (config->head_count == 0U || config->hidden_size % config->head_count != 0U ||
        (config->hidden_size / config->head_count) % 2U != 0U) {
        return 0;
    }
    return config->feed_forward_size == 0U ? config->layer_count == 1U && config->head_count == 1U
                                           : 1;
}

static size_t final_norm_parameter_index(const lm_model *model) {
    return 2U + model->config.layer_count * block_parameter_count(&model->config);
}

static lm_model_parameter *block_parameter(lm_model *model, size_t layer, size_t offset) {
    return &model->parameters[model->blocks[layer].parameter_offset + offset];
}

static llm_status initialize_rope_tables(lm_model *model) {
    const size_t head_dimension = model->config.hidden_size / model->config.head_count;
    const size_t pair_count = head_dimension / 2U;
    if (pair_count == 0U || model->config.context_length > SIZE_MAX / pair_count ||
        model->config.context_length * pair_count > SIZE_MAX / sizeof(float)) {
        return LLM_OVERFLOW;
    }
    const size_t count = model->config.context_length * pair_count;
    float *cos_values = malloc(count * sizeof(*cos_values));
    float *sin_values = malloc(count * sizeof(*sin_values));
    if (cos_values == NULL || sin_values == NULL) {
        free(sin_values);
        free(cos_values);
        return LLM_ALLOCATION_FAILED;
    }
    for (size_t position = 0U; position < model->config.context_length; ++position) {
        for (size_t pair = 0U; pair < pair_count; ++pair) {
            const float frequency = powf(10000.0F, -(2.0F * (float)pair) / (float)head_dimension);
            const float angle = (float)position * frequency;
            cos_values[position * pair_count + pair] = cosf(angle);
            sin_values[position * pair_count + pair] = sinf(angle);
        }
    }
    llm_status status = llm_tensor_write(model->backend, &model->rope_cos_table, cos_values,
                                         count * sizeof(*cos_values));
    if (status == LLM_OK) {
        status = llm_tensor_write(model->backend, &model->rope_sin_table, sin_values,
                                  count * sizeof(*sin_values));
    }
    free(sin_values);
    free(cos_values);
    return status;
}

static llm_status add_parameter(lm_model *model, size_t *next_parameter, const char *name,
                                size_t rank, const size_t *shape, uint64_t *random_state) {
    if (model == NULL || next_parameter == NULL || name == NULL ||
        *next_parameter >= model->parameter_count) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_status status =
        model->inference_only != 0
            ? lm_model_parameter_create_inference(&model->parameters[*next_parameter],
                                                  model->backend, name, rank, shape)
            : lm_model_parameter_create(&model->parameters[*next_parameter], model->backend, name,
                                        rank, shape, random_state);
    if (status == LLM_OK) {
        ++*next_parameter;
    }
    return status;
}

static llm_status add_layer_parameter(lm_model *model, size_t *next_parameter, size_t layer,
                                      const char *component, size_t rank, const size_t *shape,
                                      uint64_t *random_state) {
    char name[96] = {0};
    const int written = snprintf(name, sizeof(name), "layers.%zu.%s", layer, component);
    if (written < 0 || (size_t)written >= sizeof(name)) {
        return LLM_OVERFLOW;
    }
    return add_parameter(model, next_parameter, name, rank, shape, random_state);
}

static llm_status create_workspace(llm_backend *backend, size_t rank, const size_t *shape,
                                   llm_tensor *tensor) {
    return llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, tensor);
}

static void destroy_block_workspace(lm_transformer_block *block) {
    if (block == NULL) {
        return;
    }
    llm_tensor_destroy(&block->mlp_norm_gradient);
    llm_tensor_destroy(&block->up_gradient);
    llm_tensor_destroy(&block->gate_gradient);
    llm_tensor_destroy(&block->silu_gate_gradient);
    llm_tensor_destroy(&block->swiglu_gradient);
    llm_tensor_destroy(&block->mlp_projection);
    llm_tensor_destroy(&block->swiglu);
    llm_tensor_destroy(&block->silu_gate);
    llm_tensor_destroy(&block->up);
    llm_tensor_destroy(&block->gate);
    llm_tensor_destroy(&block->mlp_norm);
    llm_tensor_destroy(&block->attention_residual_gradient);
    llm_tensor_destroy(&block->attention_residual);
    llm_tensor_destroy(&block->linear_input_gradient);
    llm_tensor_destroy(&block->attention_norm_gradient);
    llm_tensor_destroy(&block->value_gradient);
    llm_tensor_destroy(&block->rotated_key_gradient);
    llm_tensor_destroy(&block->rotated_query_gradient);
    llm_tensor_destroy(&block->key_gradient);
    llm_tensor_destroy(&block->query_gradient);
    llm_tensor_destroy(&block->attention_output_gradient);
    llm_tensor_destroy(&block->attention_projection);
    llm_tensor_destroy(&block->attention_output);
    llm_tensor_destroy(&block->value);
    llm_tensor_destroy(&block->rotated_key);
    llm_tensor_destroy(&block->rotated_query);
    llm_tensor_destroy(&block->key);
    llm_tensor_destroy(&block->query);
    llm_tensor_destroy(&block->attention_norm);
    llm_tensor_destroy(&block->output_gradient);
    llm_tensor_destroy(&block->output);
}

static void destroy_workspaces(lm_model *model) {
    if (model == NULL) {
        return;
    }
    for (size_t layer = 0U; layer < model->config.layer_count; ++layer) {
        destroy_block_workspace(&model->blocks[layer]);
    }
    llm_tensor_destroy(&model->final_norm_gradient);
    llm_tensor_destroy(&model->final_norm);
    llm_tensor_destroy(&model->hidden_gradient);
    llm_tensor_destroy(&model->hidden);
    model->forward_batch_size = 0U;
}

static llm_status create_block_workspace(lm_model *model, lm_transformer_block *block,
                                         const size_t *hidden_shape, const size_t *attention_shape,
                                         const size_t *mlp_shape) {
    llm_status status = create_workspace(model->backend, 3U, hidden_shape, &block->output);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->output_gradient);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->attention_norm);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 4U, attention_shape, &block->query);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 4U, attention_shape, &block->key);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 4U, attention_shape, &block->rotated_query);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 4U, attention_shape, &block->rotated_key);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 4U, attention_shape, &block->value);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 4U, attention_shape, &block->attention_output);
    if (status == LLM_OK)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->attention_projection);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 4U, attention_shape,
                                  &block->attention_output_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 4U, attention_shape, &block->query_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 4U, attention_shape, &block->key_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status =
            create_workspace(model->backend, 4U, attention_shape, &block->rotated_query_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status =
            create_workspace(model->backend, 4U, attention_shape, &block->rotated_key_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 4U, attention_shape, &block->value_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status =
            create_workspace(model->backend, 3U, hidden_shape, &block->attention_norm_gradient);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->linear_input_gradient);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->attention_residual);
    if (has_mlp(&model->config) != 0 && status == LLM_OK && model->inference_only == 0)
        status =
            create_workspace(model->backend, 3U, hidden_shape, &block->attention_residual_gradient);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->mlp_norm);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->gate);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->up);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->silu_gate);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->swiglu);
    if (has_mlp(&model->config) != 0 && status == LLM_OK)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->mlp_projection);
    if (has_mlp(&model->config) != 0 && status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->swiglu_gradient);
    if (has_mlp(&model->config) != 0 && status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->silu_gate_gradient);
    if (has_mlp(&model->config) != 0 && status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->gate_gradient);
    if (has_mlp(&model->config) != 0 && status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, mlp_shape, &block->up_gradient);
    if (has_mlp(&model->config) != 0 && status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, hidden_shape, &block->mlp_norm_gradient);
    return status;
}

static llm_status ensure_workspace(lm_model *model, size_t batch_size) {
    if (model == NULL || batch_size == 0U || batch_size > SIZE_MAX / model->config.context_length) {
        return batch_size == 0U ? LLM_INVALID_SHAPE : LLM_OVERFLOW;
    }
    if (model->forward_batch_size == batch_size && model->hidden.storage != NULL) {
        return LLM_OK;
    }
    destroy_workspaces(model);
    const size_t head_dimension =
        model->config.layer_count == 0U ? 1U : model->config.hidden_size / model->config.head_count;
    const size_t hidden_shape[] = {batch_size, model->config.context_length,
                                   model->config.hidden_size};
    const size_t attention_shape[] = {batch_size, model->config.context_length,
                                      model->config.head_count, head_dimension};
    const size_t mlp_shape[] = {batch_size, model->config.context_length,
                                model->config.feed_forward_size};
    llm_status status = create_workspace(model->backend, 3U, hidden_shape, &model->hidden);
    if (status == LLM_OK && model->inference_only == 0)
        status = create_workspace(model->backend, 3U, hidden_shape, &model->hidden_gradient);
    for (size_t layer = 0U; status == LLM_OK && layer < model->config.layer_count; ++layer) {
        status = create_block_workspace(model, &model->blocks[layer], hidden_shape, attention_shape,
                                        mlp_shape);
    }
    if (status == LLM_OK && has_mlp(&model->config) != 0) {
        status = create_workspace(model->backend, 3U, hidden_shape, &model->final_norm);
    }
    if (status == LLM_OK && has_mlp(&model->config) != 0 && model->inference_only == 0) {
        status = create_workspace(model->backend, 3U, hidden_shape, &model->final_norm_gradient);
    }
    if (status != LLM_OK) {
        destroy_workspaces(model);
        return status;
    }
    model->forward_batch_size = batch_size;
    return LLM_OK;
}

static llm_status validate_forward_inputs(const lm_model *model, const llm_tensor *input_ids,
                                          const llm_tensor *logits) {
    if (model == NULL || input_ids == NULL || logits == NULL || input_ids->storage == NULL ||
        logits->storage == NULL || input_ids->rank != 2U || logits->rank != 2U ||
        input_ids->dtype != LLM_DTYPE_U32 || logits->dtype != LLM_DTYPE_F32 ||
        input_ids->shape[0] == 0U || input_ids->shape[1] != model->config.context_length) {
        return LLM_INVALID_SHAPE;
    }
    if (input_ids->shape[0] > SIZE_MAX / model->config.context_length) {
        return LLM_OVERFLOW;
    }
    const size_t row_count = input_ids->shape[0] * model->config.context_length;
    if (logits->shape[0] != row_count || logits->shape[1] != model->config.vocabulary_size) {
        return LLM_INVALID_SHAPE;
    }
    return LLM_OK;
}

llm_status lm_model_create_internal(llm_backend *backend, const lm_model_config *config,
                                    int inference_only, lm_model **out_model) {
    if (backend == NULL || out_model == NULL || config_is_valid(config) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_model = NULL;
    lm_model *model = calloc(1U, sizeof(*model));
    if (model == NULL)
        return LLM_ALLOCATION_FAILED;
    model->backend = backend;
    model->config = *config;
    model->inference_only = inference_only != 0;
    const size_t per_block = block_parameter_count(config);
    if (config->layer_count > (SIZE_MAX - 3U) / per_block) {
        free(model);
        return LLM_OVERFLOW;
    }
    model->parameter_count = config->layer_count == 0U ? 2U
                                                       : 2U + config->layer_count * per_block +
                                                             (has_mlp(config) != 0 ? 1U : 0U);
    model->parameters = calloc(model->parameter_count, sizeof(*model->parameters));
    if (model->parameters == NULL) {
        free(model);
        return LLM_ALLOCATION_FAILED;
    }
    if (config->layer_count != 0U) {
        model->blocks = calloc(config->layer_count, sizeof(*model->blocks));
        if (model->blocks == NULL) {
            lm_model_destroy(model);
            return LLM_ALLOCATION_FAILED;
        }
    }
    uint64_t random_state = config->seed == 0U ? UINT64_C(0x8a5cd789635d2dff) : config->seed;
    const size_t embedding_shape[] = {config->vocabulary_size, config->hidden_size};
    const size_t output_shape[] = {config->hidden_size, config->vocabulary_size};
    const size_t norm_shape[] = {config->hidden_size};
    const size_t attention_weight_shape[] = {config->hidden_size, config->hidden_size};
    const size_t mlp_up_shape[] = {config->hidden_size, config->feed_forward_size};
    const size_t mlp_down_shape[] = {config->feed_forward_size, config->hidden_size};
    size_t next_parameter = 0U;
    llm_status status = add_parameter(model, &next_parameter, "token_embedding", 2U,
                                      embedding_shape, &random_state);
    if (status == LLM_OK)
        status =
            add_parameter(model, &next_parameter, "output_weight", 2U, output_shape, &random_state);
    for (size_t layer = 0U; status == LLM_OK && layer < config->layer_count; ++layer) {
        model->blocks[layer].parameter_offset = next_parameter;
        status = add_layer_parameter(model, &next_parameter, layer, "attention_norm_weight", 1U,
                                     norm_shape, &random_state);
        if (status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "query_weight", 2U,
                                         attention_weight_shape, &random_state);
        if (status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "key_weight", 2U,
                                         attention_weight_shape, &random_state);
        if (status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "value_weight", 2U,
                                         attention_weight_shape, &random_state);
        if (status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "attention_output_weight",
                                         2U, attention_weight_shape, &random_state);
        if (status == LLM_OK)
            status = llm_tensor_fill_f32(
                backend, &block_parameter(model, layer, LM_BLOCK_PARAMETER_ATTENTION_NORM)->value,
                1.0F);
        if (has_mlp(config) != 0 && status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "mlp_norm_weight", 1U,
                                         norm_shape, &random_state);
        if (has_mlp(config) != 0 && status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "gate_weight", 2U,
                                         mlp_up_shape, &random_state);
        if (has_mlp(config) != 0 && status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "up_weight", 2U,
                                         mlp_up_shape, &random_state);
        if (has_mlp(config) != 0 && status == LLM_OK)
            status = add_layer_parameter(model, &next_parameter, layer, "down_weight", 2U,
                                         mlp_down_shape, &random_state);
        if (has_mlp(config) != 0 && status == LLM_OK)
            status = llm_tensor_fill_f32(
                backend, &block_parameter(model, layer, LM_BLOCK_PARAMETER_MLP_NORM)->value, 1.0F);
    }
    if (has_mlp(config) != 0 && status == LLM_OK) {
        status = add_parameter(model, &next_parameter, "final_norm_weight", 1U, norm_shape,
                               &random_state);
        if (status == LLM_OK)
            status = llm_tensor_fill_f32(
                backend, &model->parameters[final_norm_parameter_index(model)].value, 1.0F);
    }
    if (status == LLM_OK && next_parameter != model->parameter_count)
        status = LLM_INVALID_ARGUMENT;
    if (status == LLM_OK && model->inference_only == 0)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape,
                                   &model->output_weight_gradient_workspace);
    if (status == LLM_OK && config->layer_count != 0U && model->inference_only == 0)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, attention_weight_shape,
                                   &model->attention_weight_gradient_workspace);
    if (status == LLM_OK && has_mlp(config) != 0 && model->inference_only == 0)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, mlp_up_shape,
                                   &model->mlp_up_weight_gradient_workspace);
    if (status == LLM_OK && has_mlp(config) != 0 && model->inference_only == 0)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, mlp_down_shape,
                                   &model->mlp_down_weight_gradient_workspace);
    if (status == LLM_OK && config->layer_count != 0U && model->inference_only == 0)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 1U, norm_shape,
                                   &model->norm_weight_gradient_workspace);
    if (status == LLM_OK && config->layer_count != 0U) {
        const size_t rope_shape[] = {config->context_length,
                                     config->hidden_size / config->head_count / 2U};
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, rope_shape, &model->rope_cos_table);
        if (status == LLM_OK)
            status =
                llm_tensor_create(backend, LLM_DTYPE_F32, 2U, rope_shape, &model->rope_sin_table);
        if (status == LLM_OK)
            status = initialize_rope_tables(model);
    }
    if (status != LLM_OK) {
        lm_model_destroy(model);
        return status;
    }
    *out_model = model;
    return LLM_OK;
}

llm_status lm_model_create(llm_backend *backend, const lm_model_config *config,
                           lm_model **out_model) {
    return lm_model_create_internal(backend, config, 0, out_model);
}

void lm_model_destroy(lm_model *model) {
    if (model == NULL)
        return;
    destroy_workspaces(model);
    llm_tensor_destroy(&model->rope_sin_table);
    llm_tensor_destroy(&model->rope_cos_table);
    llm_tensor_destroy(&model->norm_weight_gradient_workspace);
    llm_tensor_destroy(&model->mlp_down_weight_gradient_workspace);
    llm_tensor_destroy(&model->mlp_up_weight_gradient_workspace);
    llm_tensor_destroy(&model->attention_weight_gradient_workspace);
    llm_tensor_destroy(&model->output_weight_gradient_workspace);
    for (size_t index = 0U; index < model->parameter_count; ++index)
        lm_model_parameter_destroy(&model->parameters[index]);
    free(model->blocks);
    free(model->parameters);
    free(model);
}

llm_status lm_model_get_config(const lm_model *model, lm_model_config *out_config) {
    if (model == NULL || out_config == NULL)
        return LLM_INVALID_ARGUMENT;
    *out_config = model->config;
    return LLM_OK;
}

llm_backend *lm_model_backend(lm_model *model) { return model == NULL ? NULL : model->backend; }

const llm_tensor *lm_model_output_hidden(const lm_model *model) {
    if (model == NULL)
        return NULL;
    if (has_mlp(&model->config) != 0)
        return &model->final_norm;
    return model->config.layer_count == 0U ? &model->hidden
                                           : &model->blocks[model->config.layer_count - 1U].output;
}

llm_tensor *lm_model_output_hidden_gradient(lm_model *model) {
    if (model == NULL)
        return NULL;
    if (has_mlp(&model->config) != 0)
        return &model->final_norm_gradient;
    return model->config.layer_count == 0U
               ? &model->hidden_gradient
               : &model->blocks[model->config.layer_count - 1U].output_gradient;
}

llm_status lm_rms_norm_backward_accumulate(lm_model *model, const llm_tensor *input,
                                           lm_model_parameter *norm,
                                           const llm_tensor *output_gradient,
                                           llm_tensor *input_gradient) {
    if (model == NULL || norm == NULL || model->norm_weight_gradient_workspace.storage == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    llm_status status = llm_rms_norm_backward(model->backend, input, &norm->value, output_gradient,
                                              LM_RMS_NORM_EPSILON, input_gradient,
                                              &model->norm_weight_gradient_workspace);
    if (status == LLM_OK) {
        status =
            llm_accumulate(model->backend, &model->norm_weight_gradient_workspace, &norm->gradient);
    }
    return status;
}

llm_status lm_model_forward(lm_model *model, const llm_tensor *input_ids, llm_tensor *logits) {
    llm_status status = validate_forward_inputs(model, input_ids, logits);
    if (status == LLM_OK)
        status = ensure_workspace(model, input_ids->shape[0]);
    if (status == LLM_OK)
        status = lm_embedding_forward(model, input_ids);
    if (status == LLM_OK && model->config.layer_count != 0U)
        status = lm_transformer_forward(model);
    if (status == LLM_OK && has_mlp(&model->config) != 0) {
        status = llm_rms_norm(model->backend, &model->blocks[model->config.layer_count - 1U].output,
                              &model->parameters[final_norm_parameter_index(model)].value,
                              LM_RMS_NORM_EPSILON, &model->final_norm);
    }
    if (status == LLM_OK)
        status = lm_output_head_forward(model, logits);
    return status;
}

llm_status lm_model_backward(lm_model *model, const llm_tensor *input_ids,
                             const llm_tensor *logits_gradient) {
    if (model != NULL && model->inference_only != 0)
        return LLM_UNSUPPORTED_OPERATION;
    llm_status status = validate_forward_inputs(model, input_ids, logits_gradient);
    if (status != LLM_OK)
        return status;
    if (model->forward_batch_size != input_ids->shape[0] || model->hidden.storage == NULL ||
        model->hidden_gradient.storage == NULL)
        return LLM_INVALID_ARGUMENT;
    status = lm_output_head_backward(model, logits_gradient);
    if (status == LLM_OK && has_mlp(&model->config) != 0) {
        status = lm_rms_norm_backward_accumulate(
            model, &model->blocks[model->config.layer_count - 1U].output,
            &model->parameters[final_norm_parameter_index(model)], &model->final_norm_gradient,
            &model->blocks[model->config.layer_count - 1U].output_gradient);
    }
    if (status == LLM_OK && model->config.layer_count != 0U)
        status = lm_transformer_backward(model);
    if (status == LLM_OK)
        status = lm_embedding_backward(model, input_ids);
    return status;
}

llm_status lm_model_zero_grad(lm_model *model) {
    if (model == NULL)
        return LLM_INVALID_ARGUMENT;
    if (model->inference_only != 0)
        return LLM_UNSUPPORTED_OPERATION;
    for (size_t index = 0U; index < model->parameter_count; ++index) {
        const llm_status status =
            lm_model_parameter_zero_grad(&model->parameters[index], model->backend);
        if (status != LLM_OK)
            return status;
    }
    return LLM_OK;
}

llm_status lm_model_reset_optimizer_state(lm_model *model) {
    if (model == NULL || model->backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (model->inference_only != 0)
        return LLM_UNSUPPORTED_OPERATION;
    llm_status status = LLM_OK;
    for (size_t index = 0U; status == LLM_OK && index < model->parameter_count; ++index) {
        status = llm_tensor_zero(model->backend, &model->parameters[index].gradient);
        if (status == LLM_OK) {
            status = llm_tensor_zero(model->backend, &model->parameters[index].first_moment);
        }
        if (status == LLM_OK) {
            status = llm_tensor_zero(model->backend, &model->parameters[index].second_moment);
        }
    }
    return status;
}

llm_status lm_model_apply_adamw(lm_model *model, const llm_adamw_options *options) {
    if (model == NULL || options == NULL)
        return LLM_INVALID_ARGUMENT;
    if (model->inference_only != 0)
        return LLM_UNSUPPORTED_OPERATION;
    for (size_t index = 0U; index < model->parameter_count; ++index) {
        lm_model_parameter *parameter = &model->parameters[index];
        llm_adamw_options parameter_options = *options;
        if (strstr(parameter->name, "norm_weight") != NULL) {
            parameter_options.weight_decay = 0.0F;
        }
        const llm_status status = llm_adamw_update(model->backend, &parameter->value,
                                                   &parameter->gradient, &parameter->first_moment,
                                                   &parameter->second_moment, &parameter_options);
        if (status != LLM_OK)
            return status;
    }
    return LLM_OK;
}

size_t lm_model_parameter_count(const lm_model *model) {
    return model == NULL ? 0U : model->parameter_count;
}

const char *lm_model_parameter_name(const lm_model *model, size_t index) {
    return model == NULL || index >= lm_model_parameter_count(model)
               ? NULL
               : model->parameters[index].name;
}

llm_tensor *lm_model_parameter_value(lm_model *model, size_t index) {
    return model == NULL || index >= lm_model_parameter_count(model)
               ? NULL
               : &model->parameters[index].value;
}

const llm_tensor *lm_model_parameter_gradient(const lm_model *model, size_t index) {
    return model == NULL || model->inference_only != 0 || index >= lm_model_parameter_count(model)
               ? NULL
               : &model->parameters[index].gradient;
}
