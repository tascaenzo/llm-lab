#include "model_internal.h"

llm_status lm_embedding_forward(lm_model *model, const llm_tensor *input_ids) {
    if (model == NULL || input_ids == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    return llm_gather_rows(model->backend, &model->parameters[0].value, input_ids, &model->hidden);
}

llm_status lm_embedding_backward(lm_model *model, const llm_tensor *input_ids) {
    if (model == NULL || input_ids == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    return llm_scatter_add_rows(model->backend, &model->hidden_gradient, input_ids,
                                &model->parameters[0].gradient);
}
