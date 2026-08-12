#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "model_internal.h"

static int config_is_valid(const lm_model_config *config) {
    return config != NULL && config->vocabulary_size != 0U && config->context_length != 0U &&
           config->hidden_size != 0U && config->feed_forward_size == 0U &&
           ((config->layer_count == 0U && config->head_count == 0U) ||
            (config->layer_count == 1U && config->head_count == 1U &&
             config->hidden_size % 2U == 0U));
}

static llm_status initialize_rope_tables(lm_model *model) {
    const size_t pair_count = model->config.hidden_size / 2U;
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
            const float frequency = powf(10000.0F, -(2.0F * (float)pair) /
                                                       (float)model->config.hidden_size);
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

static llm_status add_parameter(lm_model *model, const char *name, size_t rank,
                                const size_t *shape, uint64_t *random_state) {
    if (model == NULL || model->parameters == NULL || model->parameter_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    size_t count = 0U;
    while (count < model->parameter_count && model->parameters[count].name != NULL) {
        ++count;
    }
    if (count == model->parameter_count) {
        return LLM_INVALID_ARGUMENT;
    }
    return lm_model_parameter_create(&model->parameters[count], model->backend, name, rank, shape,
                                     random_state);
}

static llm_status create_workspace(llm_backend *backend, size_t rank, const size_t *shape,
                                   llm_tensor *tensor) {
    return llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, tensor);
}

static llm_status ensure_workspace(lm_model *model, size_t batch_size) {
    if (model == NULL || batch_size == 0U || batch_size > SIZE_MAX / model->config.context_length) {
        return batch_size == 0U ? LLM_INVALID_SHAPE : LLM_OVERFLOW;
    }
    if (model->forward_batch_size == batch_size && model->hidden.storage != NULL &&
        model->hidden_gradient.storage != NULL &&
        (model->config.layer_count == 0U || model->transformed_hidden.storage != NULL)) {
        return LLM_OK;
    }

    llm_tensor_destroy(&model->hidden_gradient);
    llm_tensor_destroy(&model->hidden);
    llm_tensor_destroy(&model->transformer_input_gradient);
    llm_tensor_destroy(&model->attention_norm_gradient);
    llm_tensor_destroy(&model->value_gradient);
    llm_tensor_destroy(&model->key_gradient);
    llm_tensor_destroy(&model->query_gradient);
    llm_tensor_destroy(&model->attention_output_gradient);
    llm_tensor_destroy(&model->attention_projection);
    llm_tensor_destroy(&model->attention_output);
    llm_tensor_destroy(&model->value);
    llm_tensor_destroy(&model->key);
    llm_tensor_destroy(&model->query);
    llm_tensor_destroy(&model->rotated_key_gradient);
    llm_tensor_destroy(&model->rotated_query_gradient);
    llm_tensor_destroy(&model->rotated_key);
    llm_tensor_destroy(&model->rotated_query);
    llm_tensor_destroy(&model->attention_norm);
    llm_tensor_destroy(&model->transformed_hidden_gradient);
    llm_tensor_destroy(&model->transformed_hidden);
    model->forward_batch_size = 0U;
    const size_t shape[] = {batch_size, model->config.context_length, model->config.hidden_size};
    llm_status status = llm_tensor_create(model->backend, LLM_DTYPE_F32, 3U, shape, &model->hidden);
    if (status == LLM_OK) {
        status =
            llm_tensor_create(model->backend, LLM_DTYPE_F32, 3U, shape, &model->hidden_gradient);
    }
    if (status == LLM_OK && model->config.layer_count == 1U) {
        const size_t attention_shape[] = {batch_size, model->config.context_length, 1U,
                                          model->config.hidden_size};
        status = create_workspace(model->backend, 3U, shape, &model->transformed_hidden);
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 3U, shape, &model->transformed_hidden_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 3U, shape, &model->attention_norm);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->query);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->key);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->rotated_query);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->rotated_key);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->value);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->attention_output);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 3U, shape, &model->attention_projection);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape,
                                      &model->attention_output_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->query_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->key_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape,
                                      &model->rotated_query_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape,
                                      &model->rotated_key_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 4U, attention_shape, &model->value_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 3U, shape, &model->attention_norm_gradient);
        }
        if (status == LLM_OK) {
            status = create_workspace(model->backend, 3U, shape, &model->transformer_input_gradient);
        }
    }
    if (status != LLM_OK) {
        llm_tensor_destroy(&model->hidden_gradient);
        llm_tensor_destroy(&model->hidden);
        llm_tensor_destroy(&model->transformer_input_gradient);
        llm_tensor_destroy(&model->attention_norm_gradient);
        llm_tensor_destroy(&model->value_gradient);
        llm_tensor_destroy(&model->key_gradient);
        llm_tensor_destroy(&model->query_gradient);
        llm_tensor_destroy(&model->attention_output_gradient);
        llm_tensor_destroy(&model->attention_projection);
        llm_tensor_destroy(&model->attention_output);
        llm_tensor_destroy(&model->value);
        llm_tensor_destroy(&model->key);
        llm_tensor_destroy(&model->query);
        llm_tensor_destroy(&model->rotated_key_gradient);
        llm_tensor_destroy(&model->rotated_query_gradient);
        llm_tensor_destroy(&model->rotated_key);
        llm_tensor_destroy(&model->rotated_query);
        llm_tensor_destroy(&model->attention_norm);
        llm_tensor_destroy(&model->transformed_hidden_gradient);
        llm_tensor_destroy(&model->transformed_hidden);
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

llm_status lm_model_create(llm_backend *backend, const lm_model_config *config,
                           lm_model **out_model) {
    if (backend == NULL || out_model == NULL || config_is_valid(config) == 0) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_model = NULL;
    lm_model *model = calloc(1U, sizeof(*model));
    if (model == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    model->backend = backend;
    model->config = *config;
    uint64_t random_state = config->seed == 0U ? UINT64_C(0x8a5cd789635d2dff) : config->seed;
    model->parameter_count = config->layer_count == 0U ? 2U : 7U;
    model->parameters = calloc(model->parameter_count, sizeof(*model->parameters));
    if (model->parameters == NULL) {
        free(model);
        return LLM_ALLOCATION_FAILED;
    }
    const size_t embedding_shape[] = {config->vocabulary_size, config->hidden_size};
    const size_t output_shape[] = {config->hidden_size, config->vocabulary_size};
    llm_status status = lm_model_parameter_create(&model->parameters[0], backend, "token_embedding",
                                                  2U, embedding_shape, &random_state);
    if (status == LLM_OK) {
        status = add_parameter(model, "output_weight", 2U, output_shape, &random_state);
    }
    const size_t norm_shape[] = {config->hidden_size};
    const size_t attention_weight_shape[] = {config->hidden_size, config->hidden_size};
    if (status == LLM_OK && config->layer_count == 1U) {
        status = add_parameter(model, "layers.0.attention_norm_weight", 1U, norm_shape, &random_state);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        status = add_parameter(model, "layers.0.query_weight", 2U, attention_weight_shape, &random_state);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        status = add_parameter(model, "layers.0.key_weight", 2U, attention_weight_shape, &random_state);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        status = add_parameter(model, "layers.0.value_weight", 2U, attention_weight_shape, &random_state);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        status = add_parameter(model, "layers.0.attention_output_weight", 2U,
                               attention_weight_shape, &random_state);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        status = llm_tensor_fill_f32(backend, &model->parameters[2].value, 1.0F);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, output_shape,
                                   &model->output_weight_gradient_workspace);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, attention_weight_shape,
                                   &model->attention_weight_gradient_workspace);
    }
    if (status == LLM_OK && config->layer_count == 1U) {
        const size_t rope_shape[] = {config->context_length, config->hidden_size / 2U};
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, rope_shape, &model->rope_cos_table);
        if (status == LLM_OK) {
            status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, rope_shape, &model->rope_sin_table);
        }
        if (status == LLM_OK) {
            status = initialize_rope_tables(model);
        }
    }
    if (status != LLM_OK) {
        lm_model_destroy(model);
        return status;
    }
    *out_model = model;
    return LLM_OK;
}

void lm_model_destroy(lm_model *model) {
    if (model == NULL) {
        return;
    }
    llm_tensor_destroy(&model->output_weight_gradient_workspace);
    llm_tensor_destroy(&model->attention_weight_gradient_workspace);
    llm_tensor_destroy(&model->rope_sin_table);
    llm_tensor_destroy(&model->rope_cos_table);
    llm_tensor_destroy(&model->transformer_input_gradient);
    llm_tensor_destroy(&model->attention_norm_gradient);
    llm_tensor_destroy(&model->value_gradient);
    llm_tensor_destroy(&model->key_gradient);
    llm_tensor_destroy(&model->query_gradient);
    llm_tensor_destroy(&model->attention_output_gradient);
    llm_tensor_destroy(&model->attention_projection);
    llm_tensor_destroy(&model->attention_output);
    llm_tensor_destroy(&model->value);
    llm_tensor_destroy(&model->key);
    llm_tensor_destroy(&model->query);
    llm_tensor_destroy(&model->rotated_key_gradient);
    llm_tensor_destroy(&model->rotated_query_gradient);
    llm_tensor_destroy(&model->rotated_key);
    llm_tensor_destroy(&model->rotated_query);
    llm_tensor_destroy(&model->attention_norm);
    llm_tensor_destroy(&model->transformed_hidden_gradient);
    llm_tensor_destroy(&model->transformed_hidden);
    llm_tensor_destroy(&model->hidden_gradient);
    llm_tensor_destroy(&model->hidden);
    for (size_t index = 0U; index < model->parameter_count; ++index) {
        lm_model_parameter_destroy(&model->parameters[index]);
    }
    free(model->parameters);
    free(model);
}

llm_status lm_model_get_config(const lm_model *model, lm_model_config *out_config) {
    if (model == NULL || out_config == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_config = model->config;
    return LLM_OK;
}

llm_backend *lm_model_backend(lm_model *model) { return model == NULL ? NULL : model->backend; }

llm_status lm_model_forward(lm_model *model, const llm_tensor *input_ids, llm_tensor *logits) {
    llm_status status = validate_forward_inputs(model, input_ids, logits);
    if (status != LLM_OK) {
        return status;
    }
    status = ensure_workspace(model, input_ids->shape[0]);
    if (status == LLM_OK) {
        status = lm_embedding_forward(model, input_ids);
    }
    if (status == LLM_OK && model->config.layer_count == 1U) {
        status = lm_transformer_forward(model);
    }
    if (status == LLM_OK) {
        status = lm_output_head_forward(model, logits);
    }
    return status;
}

llm_status lm_model_backward(lm_model *model, const llm_tensor *input_ids,
                             const llm_tensor *logits_gradient) {
    llm_status status = validate_forward_inputs(model, input_ids, logits_gradient);
    if (status != LLM_OK) {
        return status;
    }
    if (model->forward_batch_size != input_ids->shape[0] || model->hidden.storage == NULL ||
        model->hidden_gradient.storage == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    status = lm_output_head_backward(model, logits_gradient);
    if (status == LLM_OK && model->config.layer_count == 1U) {
        status = lm_transformer_backward(model);
    }
    if (status == LLM_OK) {
        status = lm_embedding_backward(model, input_ids);
    }
    return status;
}

llm_status lm_model_zero_grad(lm_model *model) {
    if (model == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t index = 0U; index < model->parameter_count; ++index) {
        const llm_status status = lm_model_parameter_zero_grad(&model->parameters[index], model->backend);
        if (status != LLM_OK) {
            return status;
        }
    }
    return LLM_OK;
}

llm_status lm_model_apply_adamw(lm_model *model, const llm_adamw_options *options) {
    if (model == NULL || options == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    for (size_t index = 0U; index < model->parameter_count; ++index) {
        lm_model_parameter *parameter = &model->parameters[index];
        const llm_status status =
            llm_adamw_update(model->backend, &parameter->value, &parameter->gradient,
                             &parameter->first_moment, &parameter->second_moment, options);
        if (status != LLM_OK) {
            return status;
        }
    }
    return LLM_OK;
}

size_t lm_model_parameter_count(const lm_model *model) {
    return model == NULL ? 0U : model->parameter_count;
}

const char *lm_model_parameter_name(const lm_model *model, size_t index) {
    return model == NULL || index >= lm_model_parameter_count(model) ? NULL : model->parameters[index].name;
}

llm_tensor *lm_model_parameter_value(lm_model *model, size_t index) {
    return model == NULL || index >= lm_model_parameter_count(model) ? NULL : &model->parameters[index].value;
}

const llm_tensor *lm_model_parameter_gradient(const lm_model *model, size_t index) {
    return model == NULL || index >= lm_model_parameter_count(model)
               ? NULL
               : &model->parameters[index].gradient;
}
