#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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

typedef struct lm_decode_cache {
    llm_tensor key;
    llm_tensor value;
} lm_decode_cache;

struct lm_decode_session {
    lm_model *model;
    size_t capacity;
    size_t token_count;
    int poisoned;
    lm_decode_cache *caches;

    llm_tensor token;
    llm_tensor hidden[2];
    llm_tensor attention_norm;
    llm_tensor query;
    llm_tensor key;
    llm_tensor value;
    llm_tensor rotated_query;
    llm_tensor rotated_key;
    llm_tensor attention_output;
    llm_tensor attention_projection;
    llm_tensor attention_residual;
    llm_tensor mlp_norm;
    llm_tensor gate;
    llm_tensor up;
    llm_tensor silu_gate;
    llm_tensor swiglu;
    llm_tensor mlp_projection;
    llm_tensor final_norm;
    llm_tensor logits;

    llm_tensor query_heads;
    llm_tensor key_heads;
    llm_tensor value_heads;
    llm_tensor rotated_query_heads;
    llm_tensor rotated_key_heads;
    llm_tensor attention_output_heads;
};

static int session_has_mlp(const lm_decode_session *session) {
    return session->model->config.feed_forward_size != 0U;
}

static size_t block_parameter_count(const lm_model_config *config) {
    return config->feed_forward_size == 0U ? 5U : 9U;
}

static size_t final_norm_parameter_index(const lm_decode_session *session) {
    return 2U + session->model->config.layer_count * block_parameter_count(&session->model->config);
}

static const llm_tensor *parameter_value(const lm_decode_session *session, size_t layer,
                                         size_t offset) {
    const size_t index = session->model->blocks[layer].parameter_offset + offset;
    return &session->model->parameters[index].value;
}

static llm_status create_f32(lm_decode_session *session, size_t rank, const size_t *shape,
                             llm_tensor *tensor) {
    return llm_tensor_create(session->model->backend, LLM_DTYPE_F32, rank, shape, tensor);
}

static void destroy_session_tensors(lm_decode_session *session) {
    if (session == NULL) {
        return;
    }
    llm_tensor_destroy(&session->attention_output_heads);
    llm_tensor_destroy(&session->rotated_key_heads);
    llm_tensor_destroy(&session->rotated_query_heads);
    llm_tensor_destroy(&session->value_heads);
    llm_tensor_destroy(&session->key_heads);
    llm_tensor_destroy(&session->query_heads);
    llm_tensor_destroy(&session->logits);
    llm_tensor_destroy(&session->final_norm);
    llm_tensor_destroy(&session->mlp_projection);
    llm_tensor_destroy(&session->swiglu);
    llm_tensor_destroy(&session->silu_gate);
    llm_tensor_destroy(&session->up);
    llm_tensor_destroy(&session->gate);
    llm_tensor_destroy(&session->mlp_norm);
    llm_tensor_destroy(&session->attention_residual);
    llm_tensor_destroy(&session->attention_projection);
    llm_tensor_destroy(&session->attention_output);
    llm_tensor_destroy(&session->rotated_key);
    llm_tensor_destroy(&session->rotated_query);
    llm_tensor_destroy(&session->value);
    llm_tensor_destroy(&session->key);
    llm_tensor_destroy(&session->query);
    llm_tensor_destroy(&session->attention_norm);
    llm_tensor_destroy(&session->hidden[1]);
    llm_tensor_destroy(&session->hidden[0]);
    llm_tensor_destroy(&session->token);
    if (session->caches != NULL) {
        for (size_t layer = 0U; layer < session->model->config.layer_count; ++layer) {
            llm_tensor_destroy(&session->caches[layer].value);
            llm_tensor_destroy(&session->caches[layer].key);
        }
    }
}

static llm_status create_session_tensors(lm_decode_session *session) {
    const lm_model_config *config = &session->model->config;
    const size_t hidden_shape[] = {1U, config->hidden_size};
    const size_t head_shape[] = {
        1U, config->head_count,
        config->layer_count == 0U ? config->hidden_size : config->hidden_size / config->head_count};
    const size_t mlp_shape[] = {1U, config->feed_forward_size};
    const size_t token_shape[] = {1U};
    const size_t logits_shape[] = {1U, config->vocabulary_size};
    llm_status status =
        llm_tensor_create(session->model->backend, LLM_DTYPE_U32, 1U, token_shape, &session->token);
    if (status == LLM_OK)
        status = create_f32(session, 2U, hidden_shape, &session->hidden[0]);
    if (status == LLM_OK)
        status = create_f32(session, 2U, hidden_shape, &session->hidden[1]);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->attention_norm);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->query);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->key);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->value);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->rotated_query);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->rotated_key);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->attention_output);
    if (status == LLM_OK && config->layer_count != 0U)
        status = create_f32(session, 2U, hidden_shape, &session->attention_projection);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, hidden_shape, &session->attention_residual);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, hidden_shape, &session->mlp_norm);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, mlp_shape, &session->gate);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, mlp_shape, &session->up);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, mlp_shape, &session->silu_gate);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, mlp_shape, &session->swiglu);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, hidden_shape, &session->mlp_projection);
    if (status == LLM_OK && session_has_mlp(session) != 0)
        status = create_f32(session, 2U, hidden_shape, &session->final_norm);
    if (status == LLM_OK)
        status = create_f32(session, 2U, logits_shape, &session->logits);

    if (status == LLM_OK && config->layer_count != 0U)
        status = llm_tensor_reshape(&session->query, 3U, head_shape, &session->query_heads);
    if (status == LLM_OK && config->layer_count != 0U)
        status = llm_tensor_reshape(&session->key, 3U, head_shape, &session->key_heads);
    if (status == LLM_OK && config->layer_count != 0U)
        status = llm_tensor_reshape(&session->value, 3U, head_shape, &session->value_heads);
    if (status == LLM_OK && config->layer_count != 0U)
        status = llm_tensor_reshape(&session->rotated_query, 3U, head_shape,
                                    &session->rotated_query_heads);
    if (status == LLM_OK && config->layer_count != 0U)
        status =
            llm_tensor_reshape(&session->rotated_key, 3U, head_shape, &session->rotated_key_heads);
    if (status == LLM_OK && config->layer_count != 0U)
        status = llm_tensor_reshape(&session->attention_output, 3U, head_shape,
                                    &session->attention_output_heads);

    if (status == LLM_OK && config->layer_count != 0U) {
        const size_t cache_shape[] = {1U, session->capacity, config->head_count,
                                      config->hidden_size / config->head_count};
        for (size_t layer = 0U; status == LLM_OK && layer < config->layer_count; ++layer) {
            status = create_f32(session, 4U, cache_shape, &session->caches[layer].key);
            if (status == LLM_OK)
                status = create_f32(session, 4U, cache_shape, &session->caches[layer].value);
        }
    }
    return status;
}

static llm_status linear(lm_decode_session *session, const llm_tensor *input,
                         const llm_tensor *weight, llm_tensor *output) {
    return llm_matmul(session->model->backend, input, weight, output);
}

static llm_status execute_token(lm_decode_session *session, token_id token, int compute_logits) {
    lm_model *model = session->model;
    const lm_model_config *config = &model->config;
    if (session->poisoned != 0 || session->token_count >= session->capacity) {
        return session->poisoned != 0 ? LLM_BACKEND_ERROR : LLM_INVALID_SHAPE;
    }
    if (token >= config->vocabulary_size) {
        return LLM_INVALID_INDEX;
    }

    llm_status status = llm_tensor_write(model->backend, &session->token, &token, sizeof(token));
    int batch_open = 0;
    if (status == LLM_OK) {
        status = llm_backend_begin_batch(model->backend);
        batch_open = status == LLM_OK;
    }
    if (status == LLM_OK) {
        status = llm_gather_rows(model->backend, &model->parameters[LM_PARAMETER_EMBEDDING].value,
                                 &session->token, &session->hidden[0]);
    }

    const size_t head_dimension =
        config->layer_count == 0U ? 1U : config->hidden_size / config->head_count;
    const llm_attention_options attention_options = {.scale = 1.0F / sqrtf((float)head_dimension)};
    const llm_tensor *layer_input = &session->hidden[0];
    for (size_t layer = 0U; status == LLM_OK && layer < config->layer_count; ++layer) {
        llm_tensor *layer_output = &session->hidden[(layer + 1U) % 2U];
        status = llm_rms_norm(model->backend, layer_input,
                              parameter_value(session, layer, LM_BLOCK_PARAMETER_ATTENTION_NORM),
                              LM_RMS_NORM_EPSILON, &session->attention_norm);
        if (status == LLM_OK)
            status =
                linear(session, &session->attention_norm,
                       parameter_value(session, layer, LM_BLOCK_PARAMETER_QUERY), &session->query);
        if (status == LLM_OK)
            status = linear(session, &session->attention_norm,
                            parameter_value(session, layer, LM_BLOCK_PARAMETER_KEY), &session->key);
        if (status == LLM_OK)
            status =
                linear(session, &session->attention_norm,
                       parameter_value(session, layer, LM_BLOCK_PARAMETER_VALUE), &session->value);
        if (status == LLM_OK)
            status = llm_rope_position(model->backend, &session->query_heads,
                                       &model->rope_cos_table, &model->rope_sin_table,
                                       session->token_count, &session->rotated_query_heads);
        if (status == LLM_OK)
            status = llm_rope_position(model->backend, &session->key_heads, &model->rope_cos_table,
                                       &model->rope_sin_table, session->token_count,
                                       &session->rotated_key_heads);
        if (status == LLM_OK)
            status = llm_attention_decode(
                model->backend, &session->rotated_query_heads, &session->rotated_key_heads,
                &session->value_heads, &session->caches[layer].key, &session->caches[layer].value,
                session->token_count, &attention_options, &session->attention_output_heads);
        if (status == LLM_OK)
            status = linear(session, &session->attention_output,
                            parameter_value(session, layer, LM_BLOCK_PARAMETER_ATTENTION_OUTPUT),
                            &session->attention_projection);
        if (session_has_mlp(session) != 0) {
            if (status == LLM_OK)
                status = llm_add(model->backend, layer_input, &session->attention_projection,
                                 &session->attention_residual);
            if (status == LLM_OK)
                status = llm_rms_norm(model->backend, &session->attention_residual,
                                      parameter_value(session, layer, LM_BLOCK_PARAMETER_MLP_NORM),
                                      LM_RMS_NORM_EPSILON, &session->mlp_norm);
            if (status == LLM_OK)
                status = linear(session, &session->mlp_norm,
                                parameter_value(session, layer, LM_BLOCK_PARAMETER_GATE),
                                &session->gate);
            if (status == LLM_OK)
                status =
                    linear(session, &session->mlp_norm,
                           parameter_value(session, layer, LM_BLOCK_PARAMETER_UP), &session->up);
            if (status == LLM_OK)
                status = llm_silu(model->backend, &session->gate, &session->silu_gate);
            if (status == LLM_OK)
                status = llm_multiply(model->backend, &session->silu_gate, &session->up,
                                      &session->swiglu);
            if (status == LLM_OK)
                status = linear(session, &session->swiglu,
                                parameter_value(session, layer, LM_BLOCK_PARAMETER_DOWN),
                                &session->mlp_projection);
            if (status == LLM_OK)
                status = llm_add(model->backend, &session->attention_residual,
                                 &session->mlp_projection, layer_output);
        } else if (status == LLM_OK) {
            status =
                llm_add(model->backend, layer_input, &session->attention_projection, layer_output);
        }
        layer_input = layer_output;
    }

    const llm_tensor *output_hidden = layer_input;
    if (status == LLM_OK && compute_logits != 0 && session_has_mlp(session) != 0) {
        status = llm_rms_norm(model->backend, layer_input,
                              &model->parameters[final_norm_parameter_index(session)].value,
                              LM_RMS_NORM_EPSILON, &session->final_norm);
        output_hidden = &session->final_norm;
    }
    if (status == LLM_OK && compute_logits != 0) {
        status = linear(session, output_hidden, &model->parameters[LM_PARAMETER_OUTPUT].value,
                        &session->logits);
    }

    if (batch_open != 0) {
        const llm_status end_status = llm_backend_end_batch(model->backend);
        if (status == LLM_OK)
            status = end_status;
    }
    if (status != LLM_OK) {
        session->poisoned = 1;
        return status;
    }
    ++session->token_count;
    return LLM_OK;
}

llm_status lm_decode_session_create(lm_model *model, size_t capacity,
                                    lm_decode_session **out_session) {
    if (model == NULL || out_session == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_session = NULL;
    if (capacity == 0U)
        capacity = model->config.context_length;
    if (capacity > model->config.context_length) {
        return LLM_INVALID_SHAPE;
    }
    lm_decode_session *session = calloc(1U, sizeof(*session));
    if (session == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    session->model = model;
    session->capacity = capacity;
    if (model->config.layer_count != 0U) {
        session->caches = calloc(model->config.layer_count, sizeof(*session->caches));
        if (session->caches == NULL) {
            free(session);
            return LLM_ALLOCATION_FAILED;
        }
    }
    const llm_status status = create_session_tensors(session);
    if (status != LLM_OK) {
        lm_decode_session_destroy(session);
        return status;
    }
    *out_session = session;
    return LLM_OK;
}

void lm_decode_session_destroy(lm_decode_session *session) {
    if (session == NULL)
        return;
    destroy_session_tensors(session);
    free(session->caches);
    free(session);
}

llm_status lm_decode_session_reset(lm_decode_session *session) {
    if (session == NULL)
        return LLM_INVALID_ARGUMENT;
    session->token_count = 0U;
    session->poisoned = 0;
    return LLM_OK;
}

llm_status lm_decode_session_prefill(lm_decode_session *session, const token_id *tokens,
                                     size_t token_count) {
    if (session == NULL || tokens == NULL || token_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    if (token_count > session->capacity) {
        return LLM_INVALID_SHAPE;
    }
    llm_status status = lm_decode_session_reset(session);
    for (size_t index = 0U; status == LLM_OK && index < token_count; ++index)
        status = execute_token(session, tokens[index], index + 1U == token_count);
    return status;
}

llm_status lm_decode_session_decode(lm_decode_session *session, token_id token) {
    if (session == NULL)
        return LLM_INVALID_ARGUMENT;
    return execute_token(session, token, 1);
}

const llm_tensor *lm_decode_session_logits(const lm_decode_session *session) {
    return session == NULL || session->token_count == 0U || session->poisoned != 0
               ? NULL
               : &session->logits;
}

size_t lm_decode_session_token_count(const lm_decode_session *session) {
    return session == NULL ? 0U : session->token_count;
}

size_t lm_decode_session_capacity(const lm_decode_session *session) {
    return session == NULL ? 0U : session->capacity;
}
