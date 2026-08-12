#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "model_internal.h"

static int config_is_valid(const lm_trainer_config *config) {
    return config != NULL && config->batch_size != 0U && config->context_length != 0U &&
           isfinite(config->learning_rate) != 0 && config->learning_rate > 0.0F &&
           isfinite(config->beta1) != 0 && config->beta1 >= 0.0F && config->beta1 < 1.0F &&
           isfinite(config->beta2) != 0 && config->beta2 >= 0.0F && config->beta2 < 1.0F &&
           isfinite(config->epsilon) != 0 && config->epsilon > 0.0F &&
           isfinite(config->weight_decay) != 0;
}

static void destroy_tensors(lm_trainer *trainer) {
    llm_tensor_destroy(&trainer->logits_gradient);
    llm_tensor_destroy(&trainer->loss);
    llm_tensor_destroy(&trainer->logits);
    llm_tensor_destroy(&trainer->target_ids);
    llm_tensor_destroy(&trainer->input_ids);
}

llm_status lm_trainer_create(lm_model *model, lm_dataset *dataset,
                             const lm_trainer_config *config, lm_trainer **out_trainer) {
    if (model == NULL || dataset == NULL || out_trainer == NULL || config_is_valid(config) == 0 ||
        lm_dataset_get_split(dataset) != LM_DATASET_TRAIN) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_trainer = NULL;
    lm_model_config model_config = {0};
    llm_status status = lm_model_get_config(model, &model_config);
    if (status != LLM_OK) {
        return status;
    }
    if (config->context_length != model_config.context_length ||
        lm_dataset_model_vocabulary_size(dataset) != model_config.vocabulary_size ||
        config->batch_size > SIZE_MAX / config->context_length) {
        return LLM_INVALID_SHAPE;
    }

    lm_trainer *trainer = calloc(1U, sizeof(*trainer));
    if (trainer == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    trainer->model = model;
    trainer->config = *config;
    const size_t token_count = config->batch_size * config->context_length;
    if (token_count > SIZE_MAX / sizeof(*trainer->host_inputs)) {
        free(trainer);
        return LLM_OVERFLOW;
    }
    trainer->host_inputs = malloc(token_count * sizeof(*trainer->host_inputs));
    trainer->host_targets = malloc(token_count * sizeof(*trainer->host_targets));
    if (trainer->host_inputs == NULL || trainer->host_targets == NULL) {
        lm_trainer_destroy(trainer);
        return LLM_ALLOCATION_FAILED;
    }
    const size_t input_shape[] = {config->batch_size, config->context_length};
    const size_t target_shape[] = {token_count};
    const size_t logits_shape[] = {token_count, model_config.vocabulary_size};
    const lm_dataset_status dataset_status =
        lm_batcher_create(dataset, config->batch_size, config->context_length, config->seed,
                          &trainer->batcher);
    if (dataset_status == LM_DATASET_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_U32, 2U, input_shape,
                                   &trainer->input_ids);
    } else {
        lm_trainer_destroy(trainer);
        return LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_U32, 1U, target_shape,
                                   &trainer->target_ids);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_F32, 2U, logits_shape,
                                   &trainer->logits);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_F32, 0U, NULL, &trainer->loss);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_F32, 2U, logits_shape,
                                   &trainer->logits_gradient);
    }
    if (status != LLM_OK) {
        lm_trainer_destroy(trainer);
        return status;
    }
    *out_trainer = trainer;
    return LLM_OK;
}

void lm_trainer_destroy(lm_trainer *trainer) {
    if (trainer == NULL) {
        return;
    }
    destroy_tensors(trainer);
    lm_batcher_destroy(trainer->batcher);
    free(trainer->host_targets);
    free(trainer->host_inputs);
    free(trainer);
}

llm_status lm_trainer_step(lm_trainer *trainer, float *out_loss) {
    if (trainer == NULL || out_loss == NULL || trainer->step == ULLONG_MAX) {
        return trainer == NULL || out_loss == NULL ? LLM_INVALID_ARGUMENT : LLM_OVERFLOW;
    }
    const size_t token_count = trainer->config.batch_size * trainer->config.context_length;
    if (lm_batcher_next(trainer->batcher, trainer->host_inputs, trainer->host_targets) != LM_DATASET_OK) {
        return LLM_BACKEND_ERROR;
    }
    llm_backend *backend = lm_model_backend(trainer->model);
    llm_status status = llm_tensor_write(backend, &trainer->input_ids, trainer->host_inputs,
                                         token_count * sizeof(*trainer->host_inputs));
    if (status == LLM_OK) {
        status = llm_tensor_write(backend, &trainer->target_ids, trainer->host_targets,
                                  token_count * sizeof(*trainer->host_targets));
    }
    if (status == LLM_OK) {
        status = lm_model_zero_grad(trainer->model);
    }
    if (status == LLM_OK) {
        status = lm_model_forward(trainer->model, &trainer->input_ids, &trainer->logits);
    }
    if (status == LLM_OK) {
        status = llm_cross_entropy_forward(backend, &trainer->logits, &trainer->target_ids, &trainer->loss);
    }
    float loss = 0.0F;
    if (status == LLM_OK) {
        status = llm_tensor_read(backend, &trainer->loss, &loss, sizeof(loss));
    }
    if (status == LLM_OK) {
        status = llm_cross_entropy_backward(backend, &trainer->logits, &trainer->target_ids,
                                            &trainer->logits_gradient);
    }
    if (status == LLM_OK) {
        status = lm_model_backward(trainer->model, &trainer->input_ids, &trainer->logits_gradient);
    }
    const llm_adamw_options options = {.learning_rate = trainer->config.learning_rate,
                                       .beta1 = trainer->config.beta1,
                                       .beta2 = trainer->config.beta2,
                                       .epsilon = trainer->config.epsilon,
                                       .weight_decay = trainer->config.weight_decay,
                                       .gradient_scale = 1.0F,
                                       .step = trainer->step + 1U};
    if (status == LLM_OK) {
        status = lm_model_apply_adamw(trainer->model, &options);
    }
    if (status != LLM_OK) {
        return status;
    }
    ++trainer->step;
    *out_loss = loss;
    return LLM_OK;
}

unsigned long long lm_trainer_step_count(const lm_trainer *trainer) {
    return trainer == NULL ? 0U : trainer->step;
}
