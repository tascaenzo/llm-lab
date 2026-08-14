#include <stdint.h>
#include <stdlib.h>

#include "model_internal.h"

static uint64_t next_random(uint64_t *state) {
    uint64_t value = *state;
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
    *state = value;
    return value;
}

static char *duplicate_name(const char *name) {
    if (name == NULL) {
        return NULL;
    }
    size_t length = 0U;
    while (name[length] != '\0') {
        if (length == SIZE_MAX - 1U) {
            return NULL;
        }
        ++length;
    }
    char *result = malloc(length + 1U);
    if (result != NULL) {
        for (size_t index = 0U; index <= length; ++index) {
            result[index] = name[index];
        }
    }
    return result;
}

static llm_status initialize_values(lm_model_parameter *parameter, llm_backend *backend,
                                    uint64_t *random_state) {
    if (parameter == NULL || backend == NULL || random_state == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (parameter->value.element_count > SIZE_MAX / sizeof(float)) {
        return LLM_OVERFLOW;
    }
    float *values = malloc(parameter->value.element_count * sizeof(*values));
    if (values == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    for (size_t index = 0U; index < parameter->value.element_count; ++index) {
        const uint32_t bits = (uint32_t)(next_random(random_state) >> 40U);
        const float unit = (float)bits / 16777215.0F;
        values[index] = (unit * 2.0F - 1.0F) * 0.02F;
    }
    const llm_status status = llm_tensor_write(backend, &parameter->value, values,
                                               parameter->value.element_count * sizeof(*values));
    free(values);
    return status;
}

llm_status lm_model_parameter_create(lm_model_parameter *parameter, llm_backend *backend,
                                     const char *name, size_t rank, const size_t *shape,
                                     uint64_t *random_state) {
    if (parameter == NULL || backend == NULL || name == NULL || random_state == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *parameter = (lm_model_parameter){0};
    parameter->name = duplicate_name(name);
    if (parameter->name == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    llm_status status = llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, &parameter->value);
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, &parameter->gradient);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, &parameter->first_moment);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, rank, shape, &parameter->second_moment);
    }
    if (status == LLM_OK) {
        status = llm_tensor_zero(backend, &parameter->gradient);
    }
    if (status == LLM_OK) {
        status = llm_tensor_zero(backend, &parameter->first_moment);
    }
    if (status == LLM_OK) {
        status = llm_tensor_zero(backend, &parameter->second_moment);
    }
    if (status == LLM_OK) {
        status = initialize_values(parameter, backend, random_state);
    }
    if (status != LLM_OK) {
        lm_model_parameter_destroy(parameter);
    }
    return status;
}

void lm_model_parameter_destroy(lm_model_parameter *parameter) {
    if (parameter == NULL) {
        return;
    }
    llm_tensor_destroy(&parameter->second_moment);
    llm_tensor_destroy(&parameter->first_moment);
    llm_tensor_destroy(&parameter->gradient);
    llm_tensor_destroy(&parameter->value);
    free(parameter->name);
    *parameter = (lm_model_parameter){0};
}

llm_status lm_model_parameter_zero_grad(lm_model_parameter *parameter, llm_backend *backend) {
    if (parameter == NULL || backend == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    return llm_tensor_zero(backend, &parameter->gradient);
}
