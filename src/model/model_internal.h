#ifndef LLM_LAB_MODEL_INTERNAL_H
#define LLM_LAB_MODEL_INTERNAL_H

#include "model/model.h"

/** Epsilon shared by every RMSNorm of the decoder, forward and backward. */
#define LM_RMS_NORM_EPSILON 1.0e-5F

typedef struct lm_model_parameter {
    char *name;
    llm_tensor value;
    llm_tensor gradient;
    llm_tensor first_moment;
    llm_tensor second_moment;
} lm_model_parameter;

typedef struct lm_transformer_block {
    size_t parameter_offset;
    llm_tensor output;
    llm_tensor output_gradient;
    llm_tensor attention_norm;
    llm_tensor query;
    llm_tensor key;
    llm_tensor rotated_query;
    llm_tensor rotated_key;
    llm_tensor value;
    llm_tensor attention_output;
    llm_tensor attention_projection;
    llm_tensor attention_output_gradient;
    llm_tensor query_gradient;
    llm_tensor key_gradient;
    llm_tensor rotated_query_gradient;
    llm_tensor rotated_key_gradient;
    llm_tensor value_gradient;
    llm_tensor attention_norm_gradient;
    llm_tensor linear_input_gradient;
    llm_tensor attention_residual;
    llm_tensor attention_residual_gradient;
    llm_tensor mlp_norm;
    llm_tensor gate;
    llm_tensor up;
    llm_tensor silu_gate;
    llm_tensor swiglu;
    llm_tensor mlp_projection;
    llm_tensor swiglu_gradient;
    llm_tensor silu_gate_gradient;
    llm_tensor gate_gradient;
    llm_tensor up_gradient;
    llm_tensor mlp_norm_gradient;
} lm_transformer_block;

struct lm_model {
    llm_backend *backend;
    lm_model_config config;
    lm_model_parameter *parameters;
    size_t parameter_count;
    lm_transformer_block *blocks;
    llm_tensor hidden;
    llm_tensor hidden_gradient;
    llm_tensor final_norm;
    llm_tensor final_norm_gradient;
    llm_tensor output_weight_gradient_workspace;
    llm_tensor attention_weight_gradient_workspace;
    llm_tensor mlp_up_weight_gradient_workspace;
    llm_tensor mlp_down_weight_gradient_workspace;
    llm_tensor norm_weight_gradient_workspace;
    llm_tensor rope_cos_table;
    llm_tensor rope_sin_table;
    size_t forward_batch_size;
};

struct lm_trainer {
    lm_model *model;
    lm_batcher *batcher;
    lm_sft_batcher *sft_batcher;
    int sft_mode;
    lm_trainer_config config;
    llm_tensor input_ids;
    llm_tensor target_ids;
    llm_tensor loss_mask;
    llm_tensor logits;
    llm_tensor loss;
    llm_tensor logits_gradient;
    llm_tensor gradient_norm_square;
    token_id *host_inputs;
    token_id *host_targets;
    uint32_t *host_loss_mask;
    unsigned long long step;
    float learning_rate;
    float gradient_norm;
    int gradients_are_zero;
};

llm_status lm_model_parameter_create(lm_model_parameter *parameter, llm_backend *backend,
                                     const char *name, size_t rank, const size_t *shape,
                                     uint64_t *random_state);
void lm_model_parameter_destroy(lm_model_parameter *parameter);
llm_status lm_model_parameter_zero_grad(lm_model_parameter *parameter, llm_backend *backend);

/**
 * Backward pass of one RMSNorm that accumulates into the weight gradient.
 *
 * llm_rms_norm_backward overwrites its weight gradient output, so the norm
 * parameters need the same workspace-then-accumulate treatment as the linear
 * weights: without it only the last micro-batch of a gradient accumulation
 * would reach the norm weights.
 */
llm_status lm_rms_norm_backward_accumulate(lm_model *model, const llm_tensor *input,
                                           lm_model_parameter *norm,
                                           const llm_tensor *output_gradient,
                                           llm_tensor *input_gradient);

llm_status lm_embedding_forward(lm_model *model, const llm_tensor *input_ids);
llm_status lm_embedding_backward(lm_model *model, const llm_tensor *input_ids);
llm_status lm_output_head_forward(lm_model *model, llm_tensor *logits);
llm_status lm_output_head_backward(lm_model *model, const llm_tensor *logits_gradient);
llm_status lm_transformer_forward(lm_model *model);
llm_status lm_transformer_backward(lm_model *model);
const llm_tensor *lm_model_output_hidden(const lm_model *model);
llm_tensor *lm_model_output_hidden_gradient(lm_model *model);

#endif
