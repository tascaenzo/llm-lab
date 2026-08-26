#include <float.h>
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
           isfinite(config->weight_decay) != 0 && config->weight_decay >= 0.0F &&
           config->gradient_accumulation_steps != 0U &&
           isfinite(config->minimum_learning_rate) != 0 && config->minimum_learning_rate >= 0.0F &&
           config->minimum_learning_rate <= config->learning_rate &&
           (config->sampling == LM_BATCHER_RANDOM_WINDOWS ||
            config->sampling == LM_BATCHER_SHUFFLED_WINDOWS ||
            config->sampling == LM_BATCHER_SHUFFLED_BLOCKS) &&
           isfinite(config->gradient_clip_norm) != 0 && config->gradient_clip_norm >= 0.0F &&
           (config->total_steps == 0U || config->total_steps > config->warmup_steps);
}

static float learning_rate_for_step(const lm_trainer_config *config, unsigned long long step) {
    const double maximum = (double)config->learning_rate;
    if (config->warmup_steps != 0U && step <= config->warmup_steps) {
        return (float)(maximum * (double)step / (double)config->warmup_steps);
    }
    if (config->total_steps == 0U || step >= config->total_steps) {
        return config->total_steps == 0U ? config->learning_rate : config->minimum_learning_rate;
    }
    const double progress = (double)(step - config->warmup_steps) /
                            (double)(config->total_steps - config->warmup_steps);
    const double cosine = 0.5 * (1.0 + cos(acos(-1.0) * progress));
    return (float)((double)config->minimum_learning_rate +
                   (maximum - (double)config->minimum_learning_rate) * cosine);
}

/** Reports whether the backend is an accelerator, where batching a step pays off. */
static int backend_uses_batches(const llm_backend *backend) {
    const llm_device_type device = llm_backend_device(backend);
    return device == LLM_DEVICE_METAL || device == LLM_DEVICE_CUDA;
}

static llm_status begin_device_batch(llm_backend *backend, int use_device_batch) {
    return use_device_batch != 0 ? llm_backend_begin_batch(backend) : LLM_OK;
}

/** Ends a device batch, keeping the first failure between the batch and the work. */
static llm_status end_device_batch(llm_backend *backend, int use_device_batch, llm_status status) {
    if (use_device_batch == 0) {
        return status;
    }
    const llm_status batch_status = llm_backend_end_batch(backend);
    return status == LLM_OK ? batch_status : status;
}

static void destroy_tensors(lm_trainer *trainer) {
    llm_tensor_destroy(&trainer->logits_gradient);
    llm_tensor_destroy(&trainer->loss);
    llm_tensor_destroy(&trainer->logits);
    llm_tensor_destroy(&trainer->target_ids);
    llm_tensor_destroy(&trainer->loss_mask);
    llm_tensor_destroy(&trainer->input_ids);
    llm_tensor_destroy(&trainer->gradient_norm_square);
}

llm_status lm_trainer_create(lm_model *model, lm_dataset *dataset, const lm_trainer_config *config,
                             lm_trainer **out_trainer) {
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
    trainer->learning_rate = config->learning_rate;
    trainer->gradients_are_zero = 1;
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
        lm_batcher_create_with_sampling(dataset, config->batch_size, config->context_length,
                                        config->seed, config->sampling, &trainer->batcher);
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
        status =
            llm_tensor_create(lm_model_backend(model), LLM_DTYPE_F32, 0U, NULL, &trainer->loss);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_F32, 2U, logits_shape,
                                   &trainer->logits_gradient);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(lm_model_backend(model), LLM_DTYPE_F32, 0U, NULL,
                                   &trainer->gradient_norm_square);
    }
    if (status != LLM_OK) {
        lm_trainer_destroy(trainer);
        return status;
    }
    *out_trainer = trainer;
    return LLM_OK;
}

llm_status lm_sft_trainer_create(lm_model *model, lm_sft_dataset *dataset,
                                 const lm_trainer_config *config, lm_trainer **out_trainer) {
    if (model == NULL || dataset == NULL || out_trainer == NULL || config_is_valid(config) == 0 ||
        lm_sft_dataset_get_split(dataset) != LM_DATASET_TRAIN) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_trainer = NULL;
    lm_model_config model_config = {0};
    llm_status status = lm_model_get_config(model, &model_config);
    if (status != LLM_OK) {
        return status;
    }
    if (config->context_length != model_config.context_length ||
        lm_sft_dataset_context_length(dataset) != model_config.context_length ||
        lm_sft_dataset_model_vocabulary_size(dataset) != model_config.vocabulary_size ||
        config->batch_size > SIZE_MAX / config->context_length) {
        return LLM_INVALID_SHAPE;
    }
    lm_trainer *trainer = calloc(1U, sizeof(*trainer));
    if (trainer == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    trainer->model = model;
    trainer->config = *config;
    trainer->learning_rate = config->learning_rate;
    trainer->gradients_are_zero = 1;
    trainer->sft_mode = 1;
    const size_t token_count = config->batch_size * config->context_length;
    if (token_count > SIZE_MAX / sizeof(*trainer->host_inputs)) {
        free(trainer);
        return LLM_OVERFLOW;
    }
    trainer->host_inputs = malloc(token_count * sizeof(*trainer->host_inputs));
    trainer->host_targets = malloc(token_count * sizeof(*trainer->host_targets));
    trainer->host_loss_mask = malloc(token_count * sizeof(*trainer->host_loss_mask));
    if (trainer->host_inputs == NULL || trainer->host_targets == NULL ||
        trainer->host_loss_mask == NULL) {
        lm_trainer_destroy(trainer);
        return LLM_ALLOCATION_FAILED;
    }
    if (lm_sft_batcher_create(dataset, config->batch_size, config->seed, &trainer->sft_batcher) !=
        LM_DATASET_OK) {
        lm_trainer_destroy(trainer);
        return LLM_BACKEND_ERROR;
    }
    const size_t input_shape[] = {config->batch_size, config->context_length};
    const size_t target_shape[] = {token_count};
    const size_t logits_shape[] = {token_count, model_config.vocabulary_size};
    llm_backend *backend = lm_model_backend(model);
    status = llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &trainer->input_ids);
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape,
                                   &trainer->target_ids);
    }
    if (status == LLM_OK) {
        status =
            llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &trainer->loss_mask);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &trainer->logits);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &trainer->loss);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape,
                                   &trainer->logits_gradient);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL,
                                   &trainer->gradient_norm_square);
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
    lm_sft_batcher_destroy(trainer->sft_batcher);
    free(trainer->host_loss_mask);
    free(trainer->host_targets);
    free(trainer->host_inputs);
    free(trainer);
}

llm_status lm_trainer_get_config(const lm_trainer *trainer, lm_trainer_config *out_config) {
    if (trainer == NULL || out_config == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_config = trainer->config;
    return LLM_OK;
}

static llm_status trainer_gradient_norm(lm_trainer *trainer, int use_device_batch,
                                        float normalization_divisor, float *out_norm) {
    llm_backend *backend = lm_model_backend(trainer->model);
    const size_t parameter_count = lm_model_parameter_count(trainer->model);
    llm_status status = begin_device_batch(backend, use_device_batch);
    if (status == LLM_OK) {
        status = llm_tensor_fill_f32(backend, &trainer->gradient_norm_square, 0.0F);
    }
    for (size_t index = 0U; status == LLM_OK && index < parameter_count; ++index) {
        const llm_tensor *gradient = lm_model_parameter_gradient(trainer->model, index);
        if (gradient == NULL) {
            status = LLM_INVALID_ARGUMENT;
            break;
        }
        status = llm_accumulate_sum_squares(backend, gradient, &trainer->gradient_norm_square);
    }
    status = end_device_batch(backend, use_device_batch, status);
    float sum_square = 0.0F;
    if (status == LLM_OK) {
        status = llm_tensor_read(backend, &trainer->gradient_norm_square, &sum_square,
                                 sizeof(sum_square));
    }
    if (status != LLM_OK || isfinite(sum_square) == 0 || sum_square < 0.0F) {
        return status == LLM_OK ? LLM_NUMERICAL_ERROR : status;
    }
    *out_norm = sqrtf(sum_square) / normalization_divisor;
    return isfinite(*out_norm) != 0 ? LLM_OK : LLM_NUMERICAL_ERROR;
}

llm_status lm_trainer_step(lm_trainer *trainer, float *out_loss) {
    if (trainer == NULL || out_loss == NULL || trainer->step == ULLONG_MAX) {
        return trainer == NULL || out_loss == NULL ? LLM_INVALID_ARGUMENT : LLM_OVERFLOW;
    }
    const size_t token_count = trainer->config.batch_size * trainer->config.context_length;
    llm_backend *backend = lm_model_backend(trainer->model);
    const int use_device_batch = backend_uses_batches(backend);
    llm_status status = LLM_OK;
    if (trainer->gradients_are_zero == 0) {
        status = begin_device_batch(backend, use_device_batch);
        if (status == LLM_OK) {
            status = lm_model_zero_grad(trainer->model);
        }
        status = end_device_batch(backend, use_device_batch, status);
        if (status == LLM_OK) {
            trainer->gradients_are_zero = 1;
        }
    }
    float accumulated_loss = 0.0F;
    size_t supervised_token_count = 0U;
    for (size_t micro_step = 0U;
         status == LLM_OK && micro_step < trainer->config.gradient_accumulation_steps;
         ++micro_step) {
        size_t micro_supervised_tokens = token_count;
        lm_dataset_status batch_status = LM_DATASET_OK;
        if (trainer->sft_mode != 0) {
            batch_status = lm_sft_batcher_next(
                trainer->sft_batcher, trainer->host_inputs, trainer->host_targets,
                trainer->host_loss_mask, &micro_supervised_tokens);
        } else {
            batch_status =
                lm_batcher_next(trainer->batcher, trainer->host_inputs, trainer->host_targets);
        }
        if (batch_status != LM_DATASET_OK ||
            micro_supervised_tokens > SIZE_MAX - supervised_token_count) {
            status = LLM_BACKEND_ERROR;
            break;
        }
        supervised_token_count += micro_supervised_tokens;
        status = begin_device_batch(backend, use_device_batch);
        if (status == LLM_OK) {
            status = llm_tensor_write(backend, &trainer->input_ids, trainer->host_inputs,
                                      token_count * sizeof(*trainer->host_inputs));
        }
        if (status == LLM_OK) {
            status = llm_tensor_write(backend, &trainer->target_ids, trainer->host_targets,
                                      token_count * sizeof(*trainer->host_targets));
        }
        if (status == LLM_OK && trainer->sft_mode != 0) {
            status = llm_tensor_write(backend, &trainer->loss_mask, trainer->host_loss_mask,
                                      token_count * sizeof(*trainer->host_loss_mask));
        }
        if (status == LLM_OK) {
            status = lm_model_forward(trainer->model, &trainer->input_ids, &trainer->logits);
        }
        if (status == LLM_OK) {
            status = trainer->sft_mode != 0
                         ? llm_cross_entropy_masked_forward(
                               backend, &trainer->logits, &trainer->target_ids,
                               &trainer->loss_mask, 1U, &trainer->loss)
                         : llm_cross_entropy_forward(backend, &trainer->logits,
                                                     &trainer->target_ids, &trainer->loss);
        }
        if (status == LLM_OK) {
            status = trainer->sft_mode != 0
                         ? llm_cross_entropy_masked_backward(
                               backend, &trainer->logits, &trainer->target_ids,
                               &trainer->loss_mask, 1U, &trainer->logits_gradient)
                         : llm_cross_entropy_backward(backend, &trainer->logits,
                                                      &trainer->target_ids,
                                                      &trainer->logits_gradient);
        }
        if (status == LLM_OK) {
            trainer->gradients_are_zero = 0;
            status =
                lm_model_backward(trainer->model, &trainer->input_ids, &trainer->logits_gradient);
        }
        status = end_device_batch(backend, use_device_batch, status);
        float micro_loss = 0.0F;
        if (status == LLM_OK) {
            status = llm_tensor_read(backend, &trainer->loss, &micro_loss, sizeof(micro_loss));
        }
        if (status == LLM_OK &&
            (isfinite(micro_loss) == 0 || micro_loss > FLT_MAX - accumulated_loss)) {
            status = LLM_NUMERICAL_ERROR;
        }
        if (status == LLM_OK) {
            accumulated_loss += micro_loss;
        }
    }
    const float learning_rate = learning_rate_for_step(&trainer->config, trainer->step + 1U);
    const float gradient_divisor =
        trainer->sft_mode != 0 ? (float)(supervised_token_count == 0U ? 1U
                                                                      : supervised_token_count)
                               : (float)trainer->config.gradient_accumulation_steps;
    float gradient_norm = 0.0F;
    float clipping_scale = 1.0F;
    if (status == LLM_OK && trainer->config.gradient_clip_norm > 0.0F) {
        status = trainer_gradient_norm(trainer, use_device_batch, gradient_divisor,
                                       &gradient_norm);
        if (status == LLM_OK && gradient_norm > trainer->config.gradient_clip_norm) {
            clipping_scale = trainer->config.gradient_clip_norm / gradient_norm;
        }
    }
    const llm_adamw_options options = {
        .learning_rate = learning_rate,
        .beta1 = trainer->config.beta1,
        .beta2 = trainer->config.beta2,
        .epsilon = trainer->config.epsilon,
        .weight_decay = trainer->config.weight_decay,
        .gradient_scale = clipping_scale / gradient_divisor,
        .step = trainer->step + 1U,
        .zero_gradient = 1};
    if (status == LLM_OK) {
        status = begin_device_batch(backend, use_device_batch);
        if (status == LLM_OK) {
            status = lm_model_apply_adamw(trainer->model, &options);
        }
        status = end_device_batch(backend, use_device_batch, status);
    }
    if (status != LLM_OK) {
        return status;
    }
    trainer->gradients_are_zero = 1;
    ++trainer->step;
    trainer->learning_rate = learning_rate;
    trainer->gradient_norm = gradient_norm;
    *out_loss = accumulated_loss / gradient_divisor;
    return LLM_OK;
}

unsigned long long lm_trainer_step_count(const lm_trainer *trainer) {
    return trainer == NULL ? 0U : trainer->step;
}

float lm_trainer_learning_rate(const lm_trainer *trainer) {
    return trainer == NULL ? 0.0F : trainer->learning_rate;
}

float lm_trainer_gradient_norm(const lm_trainer *trainer) {
    return trainer == NULL ? 0.0F : trainer->gradient_norm;
}

llm_status lm_model_evaluate_validation(lm_model *model, lm_dataset *dataset, size_t batch_size,
                                        size_t batch_count, uint64_t seed, float *out_loss) {
    if (model == NULL || dataset == NULL || batch_size == 0U || batch_count == 0U ||
        out_loss == NULL || lm_dataset_get_split(dataset) != LM_DATASET_VALIDATION) {
        return LLM_INVALID_ARGUMENT;
    }
    lm_model_config model_config = {0};
    llm_status status = lm_model_get_config(model, &model_config);
    if (status != LLM_OK ||
        model_config.vocabulary_size != lm_dataset_model_vocabulary_size(dataset) ||
        batch_size > SIZE_MAX / model_config.context_length) {
        return status == LLM_OK ? LLM_INVALID_SHAPE : status;
    }
    const size_t token_count = batch_size * model_config.context_length;
    if (token_count > SIZE_MAX / sizeof(token_id)) {
        return LLM_OVERFLOW;
    }
    token_id *host_inputs = malloc(token_count * sizeof(*host_inputs));
    token_id *host_targets = malloc(token_count * sizeof(*host_targets));
    const size_t input_shape[] = {batch_size, model_config.context_length};
    const size_t target_shape[] = {token_count};
    const size_t logits_shape[] = {token_count, model_config.vocabulary_size};
    llm_tensor inputs = {0};
    llm_tensor targets = {0};
    llm_tensor logits = {0};
    llm_tensor loss = {0};
    lm_batcher *batcher = NULL;
    llm_backend *backend = lm_model_backend(model);
    if (host_inputs == NULL || host_targets == NULL) {
        status = LLM_ALLOCATION_FAILED;
    }
    if (status == LLM_OK && lm_batcher_create(dataset, batch_size, model_config.context_length,
                                              seed, &batcher) != LM_DATASET_OK) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &inputs);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &targets);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss);

    double loss_sum = 0.0;
    const int use_device_batch = backend_uses_batches(backend);
    for (size_t index = 0U; status == LLM_OK && index < batch_count; ++index) {
        if (lm_batcher_next(batcher, host_inputs, host_targets) != LM_DATASET_OK) {
            status = LLM_BACKEND_ERROR;
            break;
        }
        status = begin_device_batch(backend, use_device_batch);
        if (status == LLM_OK)
            status =
                llm_tensor_write(backend, &inputs, host_inputs, token_count * sizeof(*host_inputs));
        if (status == LLM_OK)
            status = llm_tensor_write(backend, &targets, host_targets,
                                      token_count * sizeof(*host_targets));
        if (status == LLM_OK)
            status = lm_model_forward(model, &inputs, &logits);
        if (status == LLM_OK)
            status = llm_cross_entropy_forward(backend, &logits, &targets, &loss);
        status = end_device_batch(backend, use_device_batch, status);
        float batch_loss = 0.0F;
        if (status == LLM_OK)
            status = llm_tensor_read(backend, &loss, &batch_loss, sizeof(batch_loss));
        if (status == LLM_OK && isfinite(batch_loss) == 0)
            status = LLM_NUMERICAL_ERROR;
        if (status == LLM_OK)
            loss_sum += (double)batch_loss;
    }
    if (status == LLM_OK) {
        const double mean = loss_sum / (double)batch_count;
        if (isfinite(mean) == 0 || mean > FLT_MAX) {
            status = LLM_NUMERICAL_ERROR;
        } else {
            *out_loss = (float)mean;
        }
    }
    lm_batcher_destroy(batcher);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&inputs);
    free(host_targets);
    free(host_inputs);
    return status;
}

llm_status lm_model_evaluate_sft_validation(lm_model *model, lm_sft_dataset *dataset,
                                            size_t batch_size, size_t batch_count, uint64_t seed,
                                            float *out_loss) {
    if (model == NULL || dataset == NULL || batch_size == 0U || batch_count == 0U ||
        out_loss == NULL || lm_sft_dataset_get_split(dataset) != LM_DATASET_VALIDATION) {
        return LLM_INVALID_ARGUMENT;
    }
    lm_model_config config = {0};
    llm_status status = lm_model_get_config(model, &config);
    if (status != LLM_OK || config.vocabulary_size != lm_sft_dataset_model_vocabulary_size(dataset) ||
        config.context_length != lm_sft_dataset_context_length(dataset) ||
        batch_size > SIZE_MAX / config.context_length) {
        return status == LLM_OK ? LLM_INVALID_SHAPE : status;
    }
    const size_t token_count = batch_size * config.context_length;
    token_id *host_inputs = malloc(token_count * sizeof(*host_inputs));
    token_id *host_targets = malloc(token_count * sizeof(*host_targets));
    uint32_t *host_mask = malloc(token_count * sizeof(*host_mask));
    const size_t input_shape[] = {batch_size, config.context_length};
    const size_t target_shape[] = {token_count};
    const size_t logits_shape[] = {token_count, config.vocabulary_size};
    llm_tensor inputs = {0}, targets = {0}, mask = {0}, logits = {0}, loss = {0};
    lm_sft_batcher *batcher = NULL;
    llm_backend *backend = lm_model_backend(model);
    if (host_inputs == NULL || host_targets == NULL || host_mask == NULL) {
        status = LLM_ALLOCATION_FAILED;
    }
    if (status == LLM_OK && lm_sft_batcher_create(dataset, batch_size, seed, &batcher) !=
                                LM_DATASET_OK) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &inputs);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &targets);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &mask);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss);

    double loss_sum = 0.0;
    size_t supervised_count = 0U;
    const int use_device_batch = backend_uses_batches(backend);
    for (size_t index = 0U; status == LLM_OK && index < batch_count; ++index) {
        size_t batch_supervised = 0U;
        if (lm_sft_batcher_next(batcher, host_inputs, host_targets, host_mask,
                                &batch_supervised) != LM_DATASET_OK ||
            batch_supervised > SIZE_MAX - supervised_count) {
            status = LLM_BACKEND_ERROR;
            break;
        }
        supervised_count += batch_supervised;
        status = begin_device_batch(backend, use_device_batch);
        if (status == LLM_OK)
            status = llm_tensor_write(backend, &inputs, host_inputs,
                                      token_count * sizeof(*host_inputs));
        if (status == LLM_OK)
            status = llm_tensor_write(backend, &targets, host_targets,
                                      token_count * sizeof(*host_targets));
        if (status == LLM_OK)
            status =
                llm_tensor_write(backend, &mask, host_mask, token_count * sizeof(*host_mask));
        if (status == LLM_OK)
            status = lm_model_forward(model, &inputs, &logits);
        if (status == LLM_OK)
            status = llm_cross_entropy_masked_forward(backend, &logits, &targets, &mask, 1U,
                                                      &loss);
        status = end_device_batch(backend, use_device_batch, status);
        float batch_loss = 0.0F;
        if (status == LLM_OK)
            status = llm_tensor_read(backend, &loss, &batch_loss, sizeof(batch_loss));
        if (status == LLM_OK && isfinite(batch_loss) == 0)
            status = LLM_NUMERICAL_ERROR;
        if (status == LLM_OK)
            loss_sum += (double)batch_loss;
    }
    if (status == LLM_OK && supervised_count != 0U) {
        const double mean = loss_sum / (double)supervised_count;
        if (isfinite(mean) == 0 || mean > FLT_MAX) {
            status = LLM_NUMERICAL_ERROR;
        } else {
            *out_loss = (float)mean;
        }
    }
    lm_sft_batcher_destroy(batcher);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&mask);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&inputs);
    free(host_mask);
    free(host_targets);
    free(host_inputs);
    return status;
}
