#ifndef LLM_LAB_MODEL_MODEL_H
#define LLM_LAB_MODEL_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "dataset/dataset.h"
#include "runtime/runtime.h"

typedef struct lm_model lm_model;
typedef struct lm_trainer lm_trainer;

/** Configuration shared by the minimal model and its future Transformer extensions. */
typedef struct lm_model_config {
    uint32_t vocabulary_size;
    size_t context_length;
    size_t hidden_size;
    size_t layer_count;
    size_t head_count;
    size_t feed_forward_size;
    uint64_t seed;
} lm_model_config;

/** Training settings owned by one trainer instance. */
typedef struct lm_trainer_config {
    size_t batch_size;
    size_t context_length;
    uint64_t seed;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
} lm_trainer_config;

/** Creates M0: token embedding followed by an output projection. */
llm_status lm_model_create(llm_backend *backend, const lm_model_config *config,
                           lm_model **out_model);
void lm_model_destroy(lm_model *model);

/** Copies the immutable configuration into caller-owned storage. */
llm_status lm_model_get_config(const lm_model *model, lm_model_config *out_config);
llm_backend *lm_model_backend(lm_model *model);

/** Runs embedding lookup and the output head. */
llm_status lm_model_forward(lm_model *model, const llm_tensor *input_ids, llm_tensor *logits);

/** Accumulates parameter gradients from a preceding forward pass. */
llm_status lm_model_backward(lm_model *model, const llm_tensor *input_ids,
                             const llm_tensor *logits_gradient);

/** Clears every parameter gradient. */
llm_status lm_model_zero_grad(lm_model *model);

/** Applies AdamW to every registered parameter. The caller supplies a non-zero step. */
llm_status lm_model_apply_adamw(lm_model *model, const llm_adamw_options *options);

size_t lm_model_parameter_count(const lm_model *model);
const char *lm_model_parameter_name(const lm_model *model, size_t index);
llm_tensor *lm_model_parameter_value(lm_model *model, size_t index);
const llm_tensor *lm_model_parameter_gradient(const lm_model *model, size_t index);

/** Creates a trainer bound to one training dataset and model. */
llm_status lm_trainer_create(lm_model *model, lm_dataset *dataset,
                             const lm_trainer_config *config, lm_trainer **out_trainer);
void lm_trainer_destroy(lm_trainer *trainer);

/** Executes batch -> forward -> loss -> backward -> AdamW and returns the mean loss. */
llm_status lm_trainer_step(lm_trainer *trainer, float *out_loss);
unsigned long long lm_trainer_step_count(const lm_trainer *trainer);

/** Atomically saves model configuration, parameters, AdamW moments and trainer state. */
llm_status lm_trainer_save_checkpoint(const lm_trainer *trainer, lm_dataset *dataset,
                                      const char *path);

/** Restores a model from a checkpoint; with a dataset, also restores its trainer and batcher state. */
llm_status lm_trainer_load_checkpoint(llm_backend *backend, lm_dataset *dataset, const char *path,
                                      lm_model **out_model, lm_trainer **out_trainer);

#endif
