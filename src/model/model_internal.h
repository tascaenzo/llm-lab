#ifndef LLM_LAB_MODEL_INTERNAL_H
#define LLM_LAB_MODEL_INTERNAL_H

#include "model/model.h"

typedef struct lm_model_parameter {
    const char *name;
    llm_tensor value;
    llm_tensor gradient;
    llm_tensor first_moment;
    llm_tensor second_moment;
} lm_model_parameter;

struct lm_model {
    llm_backend *backend;
    lm_model_config config;
    lm_model_parameter *parameters;
    size_t parameter_count;
    llm_tensor hidden;
    llm_tensor hidden_gradient;
    llm_tensor transformed_hidden;
    llm_tensor transformed_hidden_gradient;
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
    llm_tensor transformer_input_gradient;
    llm_tensor output_weight_gradient_workspace;
    llm_tensor attention_weight_gradient_workspace;
    llm_tensor rope_cos_table;
    llm_tensor rope_sin_table;
    size_t forward_batch_size;
};

struct lm_trainer {
    lm_model *model;
    lm_batcher *batcher;
    lm_trainer_config config;
    llm_tensor input_ids;
    llm_tensor target_ids;
    llm_tensor logits;
    llm_tensor loss;
    llm_tensor logits_gradient;
    token_id *host_inputs;
    token_id *host_targets;
    unsigned long long step;
};

llm_status lm_model_parameter_create(lm_model_parameter *parameter, llm_backend *backend,
                                     const char *name, size_t rank, const size_t *shape,
                                     uint64_t *random_state);
void lm_model_parameter_destroy(lm_model_parameter *parameter);
llm_status lm_model_parameter_zero_grad(lm_model_parameter *parameter, llm_backend *backend);

llm_status lm_embedding_forward(lm_model *model, const llm_tensor *input_ids);
llm_status lm_embedding_backward(lm_model *model, const llm_tensor *input_ids);
llm_status lm_output_head_forward(lm_model *model, llm_tensor *logits);
llm_status lm_output_head_backward(lm_model *model, const llm_tensor *logits_gradient);
llm_status lm_transformer_forward(lm_model *model);
llm_status lm_transformer_backward(lm_model *model);

#endif
