#ifndef LLM_LAB_MODEL_MODEL_H
#define LLM_LAB_MODEL_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "dataset/dataset.h"
#include "dataset/sft_dataset.h"
#include "runtime/runtime.h"

typedef struct lm_model lm_model;
typedef struct lm_trainer lm_trainer;
typedef struct lm_decode_session lm_decode_session;

/** Configuration shared by the Minimal Model and scalable decoder-only Transformer. */
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
    /** Number of micro-batches accumulated into one optimizer update. */
    size_t gradient_accumulation_steps;
    /** Number of optimizer updates for linear warmup; zero disables warmup. */
    uint64_t warmup_steps;
    /** Total optimizer updates for cosine decay; zero keeps a constant rate. */
    uint64_t total_steps;
    /** Learning-rate floor after warmup when total_steps is non-zero. */
    float minimum_learning_rate;
    /** Sampling policy for the training windows. */
    lm_batcher_sampling sampling;
    /** Global norm threshold; zero leaves gradients unclipped. */
    float gradient_clip_norm;
} lm_trainer_config;

/**
 * Optional resume-time execution overrides. Zero restores the saved value.
 * The loader accepts a changed micro-batch only when batch_size multiplied by
 * gradient_accumulation_steps remains identical to the checkpoint, preserving
 * the number of sampled sequences and optimizer semantics per update.
 */
typedef struct lm_trainer_resume_options {
    size_t batch_size;
    size_t gradient_accumulation_steps;
} lm_trainer_resume_options;

/** Creates a decoder-only model described by config. */
llm_status lm_model_create(llm_backend *backend, const lm_model_config *config,
                           lm_model **out_model);
void lm_model_destroy(lm_model *model);

/** Copies the immutable configuration into caller-owned storage. */
llm_status lm_model_get_config(const lm_model *model, lm_model_config *out_config);
llm_backend *lm_model_backend(lm_model *model);

/** Runs embedding lookup and the output head. */
llm_status lm_model_forward(lm_model *model, const llm_tensor *input_ids, llm_tensor *logits);

/**
 * Creates a stateful, inference-only decoder with one KV cache per Transformer layer.
 * The model must outlive the session. A capacity of zero uses the model context length.
 */
llm_status lm_decode_session_create(lm_model *model, size_t capacity,
                                    lm_decode_session **out_session);
void lm_decode_session_destroy(lm_decode_session *session);

/** Discards the logical cache contents without reallocating device memory. */
llm_status lm_decode_session_reset(lm_decode_session *session);

/**
 * Resets the session and consumes a prompt. Logits for its final token become available.
 */
llm_status lm_decode_session_prefill(lm_decode_session *session, const token_id *tokens,
                                     size_t token_count);

/** Appends one token and computes only the next-token logits for that position. */
llm_status lm_decode_session_decode(lm_decode_session *session, token_id token);

/** Device-resident FP32 [1, vocabulary_size] logits from the most recent token. */
const llm_tensor *lm_decode_session_logits(const lm_decode_session *session);
size_t lm_decode_session_token_count(const lm_decode_session *session);
size_t lm_decode_session_capacity(const lm_decode_session *session);

/** Accumulates parameter gradients from a preceding forward pass. */
llm_status lm_model_backward(lm_model *model, const llm_tensor *input_ids,
                             const llm_tensor *logits_gradient);

/** Clears every parameter gradient. */
llm_status lm_model_zero_grad(lm_model *model);

/** Clears gradients and AdamW moments while preserving trained parameter values. */
llm_status lm_model_reset_optimizer_state(lm_model *model);

/** Applies AdamW to every registered parameter. The caller supplies a non-zero step. */
llm_status lm_model_apply_adamw(lm_model *model, const llm_adamw_options *options);

size_t lm_model_parameter_count(const lm_model *model);
const char *lm_model_parameter_name(const lm_model *model, size_t index);
llm_tensor *lm_model_parameter_value(lm_model *model, size_t index);
const llm_tensor *lm_model_parameter_gradient(const lm_model *model, size_t index);

/** Creates a trainer bound to one training dataset and model. */
llm_status lm_trainer_create(lm_model *model, lm_dataset *dataset, const lm_trainer_config *config,
                             lm_trainer **out_trainer);
/** Creates an assistant-only-loss trainer over an SFT training split. */
llm_status lm_sft_trainer_create(lm_model *model, lm_sft_dataset *dataset,
                                 const lm_trainer_config *config, lm_trainer **out_trainer);
void lm_trainer_destroy(lm_trainer *trainer);
/** Copies the immutable training configuration into caller-owned storage. */
llm_status lm_trainer_get_config(const lm_trainer *trainer, lm_trainer_config *out_config);

/** Executes one optimizer update and returns mean loss across its micro-batches. */
llm_status lm_trainer_step(lm_trainer *trainer, float *out_loss);
unsigned long long lm_trainer_step_count(const lm_trainer *trainer);
/** Returns the learning rate used by the most recent optimizer update. */
float lm_trainer_learning_rate(const lm_trainer *trainer);
/** Returns the global norm of the averaged gradients in the most recent update. */
float lm_trainer_gradient_norm(const lm_trainer *trainer);

/** Evaluates fixed random windows from a validation split without changing trainer state. */
llm_status lm_model_evaluate_validation(lm_model *model, lm_dataset *dataset, size_t batch_size,
                                        size_t batch_count, uint64_t seed, float *out_loss);
llm_status lm_model_evaluate_sft_validation(lm_model *model, lm_sft_dataset *dataset,
                                            size_t batch_size, size_t batch_count, uint64_t seed,
                                            float *out_loss);

/** Atomically saves model configuration, parameters, AdamW moments and trainer state. */
llm_status lm_trainer_save_checkpoint(const lm_trainer *trainer, lm_dataset *dataset,
                                      const char *path);
llm_status lm_sft_trainer_save_checkpoint(const lm_trainer *trainer, lm_sft_dataset *dataset,
                                          const char *path);

/** Restores a model from a checkpoint; with a dataset, also restores its trainer and batcher state.
 */
llm_status lm_trainer_load_checkpoint(llm_backend *backend, lm_dataset *dataset, const char *path,
                                      lm_model **out_model, lm_trainer **out_trainer);

/** Restores a checkpoint with a batch-equivalent execution configuration. */
llm_status lm_trainer_load_checkpoint_with_options(llm_backend *backend, lm_dataset *dataset,
                                                   const char *path,
                                                   const lm_trainer_resume_options *options,
                                                   lm_model **out_model, lm_trainer **out_trainer);

llm_status lm_sft_trainer_load_checkpoint_with_options(llm_backend *backend,
                                                       lm_sft_dataset *dataset, const char *path,
                                                       const lm_trainer_resume_options *options,
                                                       lm_model **out_model,
                                                       lm_trainer **out_trainer);
llm_status lm_sft_trainer_load_checkpoint(llm_backend *backend, lm_sft_dataset *dataset,
                                          const char *path, lm_model **out_model,
                                          lm_trainer **out_trainer);

/**
 * Loads only parameter values from a training checkpoint. Optimizer moments,
 * gradients and backward workspaces are neither allocated nor transferred.
 */
llm_status lm_model_load_checkpoint_for_inference(llm_backend *backend, const char *path,
                                                  lm_model **out_model);

#endif
