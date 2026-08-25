#include <math.h>

#include "cpu_internal.h"

#define LLM_CPU_OPTIMIZER_VALUES_PER_TASK 16384U

typedef struct cpu_adamw_job {
    float *parameter;
    float *gradient;
    float *first_moment;
    float *second_moment;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float gradient_scale;
    float inverse_first_bias;
    float inverse_second_bias;
    int zero_gradient;
} cpu_adamw_job;

static llm_status adamw_range(void *context, size_t begin, size_t end) {
    cpu_adamw_job *job = context;
    for (size_t index = begin; index < end; ++index) {
        const float parameter = job->parameter[index];
        const float gradient = job->gradient[index] * job->gradient_scale;
        const float first_moment =
            job->beta1 * job->first_moment[index] + (1.0F - job->beta1) * gradient;
        const float second_moment =
            job->beta2 * job->second_moment[index] + (1.0F - job->beta2) * gradient * gradient;
        const float corrected_first = first_moment * job->inverse_first_bias;
        const float corrected_second = second_moment * job->inverse_second_bias;
        const float updated =
            parameter -
            job->learning_rate * (corrected_first / (sqrtf(corrected_second) + job->epsilon) +
                                  job->weight_decay * parameter);
        if (!isfinite(parameter) || !isfinite(gradient) || !isfinite(first_moment) ||
            !isfinite(second_moment) || !isfinite(updated)) {
            return LLM_NUMERICAL_ERROR;
        }
        job->first_moment[index] = first_moment;
        job->second_moment[index] = second_moment;
        job->parameter[index] = updated;
        if (job->zero_gradient != 0) {
            job->gradient[index] = 0.0F;
        }
    }
    return LLM_OK;
}

llm_status llm_cpu_execute_adamw_update_f32(void *context, float *parameter, float *gradient,
                                            float *first_moment, float *second_moment,
                                            size_t value_count, float learning_rate, float beta1,
                                            float beta2, float epsilon, float weight_decay,
                                            float gradient_scale, unsigned long long step,
                                            int zero_gradient) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || parameter == NULL || gradient == NULL || first_moment == NULL ||
        second_moment == NULL || value_count == 0U || !isfinite(learning_rate) ||
        learning_rate < 0.0F || !isfinite(beta1) || beta1 < 0.0F || beta1 >= 1.0F ||
        !isfinite(beta2) || beta2 < 0.0F || beta2 >= 1.0F || !isfinite(epsilon) ||
        epsilon <= 0.0F || !isfinite(weight_decay) || weight_decay < 0.0F ||
        !isfinite(gradient_scale) || step == 0ULL) {
        return LLM_INVALID_ARGUMENT;
    }
    const float first_bias = 1.0F - (float)pow((double)beta1, (double)step);
    const float second_bias = 1.0F - (float)pow((double)beta2, (double)step);
    if (!isfinite(first_bias) || !isfinite(second_bias) || first_bias <= 0.0F ||
        second_bias <= 0.0F) {
        return LLM_NUMERICAL_ERROR;
    }
    cpu_adamw_job job = {
        .parameter = parameter,
        .gradient = gradient,
        .first_moment = first_moment,
        .second_moment = second_moment,
        .learning_rate = learning_rate,
        .beta1 = beta1,
        .beta2 = beta2,
        .epsilon = epsilon,
        .weight_decay = weight_decay,
        .gradient_scale = gradient_scale,
        .inverse_first_bias = 1.0F / first_bias,
        .inverse_second_bias = 1.0F / second_bias,
        .zero_gradient = zero_gradient,
    };
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_OPTIMIZER_VALUES_PER_TASK,
                                adamw_range, &job);
}
