#include <math.h>
#include <string.h>

#include "cpu_internal.h"

#define LLM_CPU_TRANSFORMER_VALUES_PER_TASK 16384U
#define LLM_CPU_TRANSFORMER_ROWS_PER_TASK 16U

typedef struct cpu_silu_job {
    const float *input;
    const float *output_gradient;
    float *output;
} cpu_silu_job;

typedef struct cpu_rms_norm_job {
    const float *input;
    const float *weight;
    float epsilon;
    float *output;
    size_t row_width;
} cpu_rms_norm_job;

typedef struct cpu_rope_job {
    const float *input;
    const float *cos_table;
    const float *sin_table;
    float *output;
    size_t sequence_length;
    size_t head_count;
    size_t head_dimension;
    size_t table_position_count;
    size_t position_offset;
    int backward;
} cpu_rope_job;

static float stable_sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + expf(-value));
    }
    const float exponential = expf(value);
    return exponential / (1.0F + exponential);
}

static llm_status silu_range(void *context, size_t begin, size_t end) {
    cpu_silu_job *job = context;
    for (size_t index = begin; index < end; ++index) {
        const float input = job->input[index];
        if (!isfinite(input)) {
            return LLM_NUMERICAL_ERROR;
        }
        const float sigmoid = stable_sigmoid(input);
        const float value = input * sigmoid;
        if (!isfinite(value)) {
            return LLM_NUMERICAL_ERROR;
        }
        job->output[index] = value;
    }
    return LLM_OK;
}

static llm_status silu_backward_range(void *context, size_t begin, size_t end) {
    cpu_silu_job *job = context;
    for (size_t index = begin; index < end; ++index) {
        const float input = job->input[index];
        const float output_gradient = job->output_gradient[index];
        if (!isfinite(input) || !isfinite(output_gradient)) {
            return LLM_NUMERICAL_ERROR;
        }
        const float sigmoid = stable_sigmoid(input);
        const float value = output_gradient * sigmoid * (1.0F + input * (1.0F - sigmoid));
        if (!isfinite(value)) {
            return LLM_NUMERICAL_ERROR;
        }
        job->output[index] = value;
    }
    return LLM_OK;
}

static llm_status rms_norm_range(void *context, size_t begin, size_t end) {
    cpu_rms_norm_job *job = context;
    for (size_t row = begin; row < end; ++row) {
        const float *input = job->input + row * job->row_width;
        float *output = job->output + row * job->row_width;
        float square_sum = 0.0F;
        for (size_t column = 0U; column < job->row_width; ++column) {
            if (!isfinite(input[column]) || !isfinite(job->weight[column])) {
                return LLM_NUMERICAL_ERROR;
            }
            square_sum += input[column] * input[column];
        }
        const float inverse_rms = 1.0F / sqrtf(square_sum / (float)job->row_width + job->epsilon);
        if (!isfinite(inverse_rms)) {
            return LLM_NUMERICAL_ERROR;
        }
        for (size_t column = 0U; column < job->row_width; ++column) {
            output[column] = input[column] * inverse_rms * job->weight[column];
            if (!isfinite(output[column])) {
                return LLM_NUMERICAL_ERROR;
            }
        }
    }
    return LLM_OK;
}

static llm_status rope_range(void *context, size_t begin, size_t end) {
    cpu_rope_job *job = context;
    const size_t pairs_per_head = job->head_dimension / 2U;
    const size_t pairs_per_position = job->head_count * pairs_per_head;
    for (size_t pair_index = begin; pair_index < end; ++pair_index) {
        const size_t position_index = pair_index / pairs_per_position;
        const size_t pair_within_position = pair_index % pairs_per_position;
        const size_t pair = pair_within_position % pairs_per_head;
        const size_t sequence = position_index % job->sequence_length;
        const size_t table_position = job->position_offset + sequence;
        if (table_position >= job->table_position_count) {
            return LLM_INVALID_SHAPE;
        }
        const size_t value_index = pair_index * 2U;
        const size_t table_index = table_position * pairs_per_head + pair;
        const float first = job->input[value_index];
        const float second = job->input[value_index + 1U];
        const float cosine = job->cos_table[table_index];
        const float sine = job->sin_table[table_index];
        if (!isfinite(first) || !isfinite(second) || !isfinite(cosine) || !isfinite(sine)) {
            return LLM_NUMERICAL_ERROR;
        }
        if (job->backward == 0) {
            job->output[value_index] = first * cosine - second * sine;
            job->output[value_index + 1U] = first * sine + second * cosine;
        } else {
            job->output[value_index] = first * cosine + second * sine;
            job->output[value_index + 1U] = -first * sine + second * cosine;
        }
        if (!isfinite(job->output[value_index]) || !isfinite(job->output[value_index + 1U])) {
            return LLM_NUMERICAL_ERROR;
        }
    }
    return LLM_OK;
}

static size_t attention_offset(size_t batch, size_t sequence, size_t head, size_t sequence_length,
                               size_t head_count, size_t head_dimension) {
    return ((batch * sequence_length + sequence) * head_count + head) * head_dimension;
}

static float attention_score(const float *query, const float *key, size_t head_dimension,
                             float scale, llm_status *status) {
    float dot = 0.0F;
    for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
        if (!isfinite(query[dimension]) || !isfinite(key[dimension])) {
            *status = LLM_NUMERICAL_ERROR;
            return 0.0F;
        }
        dot += query[dimension] * key[dimension];
    }
    dot *= scale;
    if (!isfinite(dot)) {
        *status = LLM_NUMERICAL_ERROR;
    }
    return dot;
}

llm_status llm_cpu_execute_silu_f32(void *context, const float *input, float *output,
                                    size_t value_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || input == NULL || output == NULL || value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_silu_job job = {.input = input, .output_gradient = NULL, .output = output};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_TRANSFORMER_VALUES_PER_TASK,
                                silu_range, &job);
}

llm_status llm_cpu_execute_silu_backward_f32(void *context, const float *input,
                                             const float *output_gradient, float *input_gradient,
                                             size_t value_count) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || input == NULL || output_gradient == NULL || input_gradient == NULL ||
        value_count == 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_silu_job job = {
        .input = input, .output_gradient = output_gradient, .output = input_gradient};
    return llm_cpu_parallel_for(cpu->executor, value_count, LLM_CPU_TRANSFORMER_VALUES_PER_TASK,
                                silu_backward_range, &job);
}

llm_status llm_cpu_execute_rms_norm_f32(void *context, const float *input, const float *weight,
                                        float epsilon, float *output, size_t outer_count,
                                        size_t row_width) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || input == NULL || weight == NULL || output == NULL || outer_count == 0U ||
        row_width == 0U || !isfinite(epsilon) || epsilon <= 0.0F) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_rms_norm_job job = {.input = input,
                            .weight = weight,
                            .epsilon = epsilon,
                            .output = output,
                            .row_width = row_width};
    return llm_cpu_parallel_for(cpu->executor, outer_count, LLM_CPU_TRANSFORMER_ROWS_PER_TASK,
                                rms_norm_range, &job);
}

llm_status llm_cpu_execute_rms_norm_backward_f32(void *context, const float *input,
                                                 const float *weight, const float *output_gradient,
                                                 float epsilon, float *input_gradient,
                                                 float *weight_gradient, size_t outer_count,
                                                 size_t row_width) {
    if (context == NULL || input == NULL || weight == NULL || output_gradient == NULL ||
        input_gradient == NULL || weight_gradient == NULL || outer_count == 0U || row_width == 0U ||
        !isfinite(epsilon) || epsilon <= 0.0F) {
        return LLM_INVALID_ARGUMENT;
    }
    (void)memset(weight_gradient, 0, row_width * sizeof(*weight_gradient));
    for (size_t row = 0U; row < outer_count; ++row) {
        const float *row_input = input + row * row_width;
        const float *row_output_gradient = output_gradient + row * row_width;
        float *row_input_gradient = input_gradient + row * row_width;
        float square_sum = 0.0F;
        float projected_gradient = 0.0F;
        for (size_t column = 0U; column < row_width; ++column) {
            if (!isfinite(row_input[column]) || !isfinite(weight[column]) ||
                !isfinite(row_output_gradient[column])) {
                return LLM_NUMERICAL_ERROR;
            }
            square_sum += row_input[column] * row_input[column];
            projected_gradient += row_output_gradient[column] * weight[column] * row_input[column];
        }
        const float inverse_rms = 1.0F / sqrtf(square_sum / (float)row_width + epsilon);
        const float correction =
            projected_gradient * inverse_rms * inverse_rms * inverse_rms / (float)row_width;
        if (!isfinite(inverse_rms) || !isfinite(correction)) {
            return LLM_NUMERICAL_ERROR;
        }
        for (size_t column = 0U; column < row_width; ++column) {
            row_input_gradient[column] =
                row_output_gradient[column] * weight[column] * inverse_rms -
                row_input[column] * correction;
            weight_gradient[column] +=
                row_output_gradient[column] * row_input[column] * inverse_rms;
            if (!isfinite(row_input_gradient[column]) || !isfinite(weight_gradient[column])) {
                return LLM_NUMERICAL_ERROR;
            }
        }
    }
    return LLM_OK;
}

static llm_status execute_rope(void *context, const float *input, const float *cos_table,
                               const float *sin_table, size_t batch_count, size_t sequence_length,
                               size_t head_count, size_t head_dimension,
                               size_t table_position_count, size_t position_offset, float *output,
                               int backward) {
    llm_cpu_context *cpu = context;
    if (cpu == NULL || input == NULL || cos_table == NULL || sin_table == NULL || output == NULL ||
        batch_count == 0U || sequence_length == 0U || head_count == 0U || head_dimension == 0U ||
        head_dimension % 2U != 0U || table_position_count == 0U ||
        position_offset > table_position_count ||
        sequence_length > table_position_count - position_offset) {
        return LLM_INVALID_ARGUMENT;
    }
    cpu_rope_job job = {
        .input = input,
        .cos_table = cos_table,
        .sin_table = sin_table,
        .output = output,
        .sequence_length = sequence_length,
        .head_count = head_count,
        .head_dimension = head_dimension,
        .table_position_count = table_position_count,
        .position_offset = position_offset,
        .backward = backward,
    };
    const size_t pair_count = batch_count * sequence_length * head_count * head_dimension / 2U;
    return llm_cpu_parallel_for(cpu->executor, pair_count, LLM_CPU_TRANSFORMER_VALUES_PER_TASK,
                                rope_range, &job);
}

llm_status llm_cpu_execute_rope_f32(void *context, const float *input, const float *cos_table,
                                    const float *sin_table, size_t batch_count,
                                    size_t sequence_length, size_t head_count,
                                    size_t head_dimension, size_t table_position_count,
                                    size_t position_offset, float *output) {
    return execute_rope(context, input, cos_table, sin_table, batch_count, sequence_length,
                        head_count, head_dimension, table_position_count, position_offset, output,
                        0);
}

llm_status llm_cpu_execute_rope_backward_f32(void *context, const float *output_gradient,
                                             const float *cos_table, const float *sin_table,
                                             size_t batch_count, size_t sequence_length,
                                             size_t head_count, size_t head_dimension,
                                             size_t table_position_count, size_t position_offset,
                                             float *input_gradient) {
    return execute_rope(context, output_gradient, cos_table, sin_table, batch_count,
                        sequence_length, head_count, head_dimension, table_position_count,
                        position_offset, input_gradient, 1);
}

llm_status llm_cpu_execute_attention_forward_f32(
    void *context, const float *query, const float *key, const float *value, float scale,
    size_t query_position_offset, size_t batch_count, size_t query_length, size_t key_length,
    size_t query_head_count, size_t key_value_head_count, size_t head_dimension, float *output) {
    if (context == NULL || query == NULL || key == NULL || value == NULL || output == NULL ||
        !isfinite(scale) || scale <= 0.0F || batch_count == 0U || query_length == 0U ||
        key_length == 0U || query_head_count == 0U || key_value_head_count == 0U ||
        head_dimension == 0U || query_head_count % key_value_head_count != 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    const size_t heads_per_group = query_head_count / key_value_head_count;
    for (size_t batch = 0U; batch < batch_count; ++batch) {
        for (size_t query_position = 0U; query_position < query_length; ++query_position) {
            size_t visible_keys = query_position_offset + query_position + 1U;
            if (visible_keys > key_length) {
                visible_keys = key_length;
            }
            for (size_t query_head = 0U; query_head < query_head_count; ++query_head) {
                const size_t key_value_head = query_head / heads_per_group;
                const float *query_row =
                    query + attention_offset(batch, query_position, query_head, query_length,
                                             query_head_count, head_dimension);
                float *output_row =
                    output + attention_offset(batch, query_position, query_head, query_length,
                                              query_head_count, head_dimension);
                float maximum = -INFINITY;
                llm_status status = LLM_OK;
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const float *key_row =
                        key + attention_offset(batch, key_position, key_value_head, key_length,
                                               key_value_head_count, head_dimension);
                    const float score =
                        attention_score(query_row, key_row, head_dimension, scale, &status);
                    if (status != LLM_OK) {
                        return status;
                    }
                    if (score > maximum) {
                        maximum = score;
                    }
                }
                float denominator = 0.0F;
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const float *key_row =
                        key + attention_offset(batch, key_position, key_value_head, key_length,
                                               key_value_head_count, head_dimension);
                    denominator +=
                        expf(attention_score(query_row, key_row, head_dimension, scale, &status) -
                             maximum);
                }
                if (status != LLM_OK || !isfinite(denominator) || denominator <= 0.0F) {
                    return LLM_NUMERICAL_ERROR;
                }
                for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                    output_row[dimension] = 0.0F;
                }
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const float *key_row =
                        key + attention_offset(batch, key_position, key_value_head, key_length,
                                               key_value_head_count, head_dimension);
                    const float *value_row =
                        value + attention_offset(batch, key_position, key_value_head, key_length,
                                                 key_value_head_count, head_dimension);
                    const float probability =
                        expf(attention_score(query_row, key_row, head_dimension, scale, &status) -
                             maximum) /
                        denominator;
                    for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                        if (!isfinite(value_row[dimension])) {
                            return LLM_NUMERICAL_ERROR;
                        }
                        output_row[dimension] += probability * value_row[dimension];
                    }
                }
                for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                    if (!isfinite(output_row[dimension])) {
                        return LLM_NUMERICAL_ERROR;
                    }
                }
            }
        }
    }
    return LLM_OK;
}

llm_status llm_cpu_execute_attention_backward_f32(
    void *context, const float *query, const float *key, const float *value,
    const float *output_gradient, float scale, size_t query_position_offset, size_t batch_count,
    size_t query_length, size_t key_length, size_t query_head_count, size_t key_value_head_count,
    size_t head_dimension, float *query_gradient, float *key_gradient, float *value_gradient) {
    if (context == NULL || query == NULL || key == NULL || value == NULL ||
        output_gradient == NULL || query_gradient == NULL || key_gradient == NULL ||
        value_gradient == NULL || !isfinite(scale) || scale <= 0.0F || batch_count == 0U ||
        query_length == 0U || key_length == 0U || query_head_count == 0U ||
        key_value_head_count == 0U || head_dimension == 0U ||
        query_head_count % key_value_head_count != 0U) {
        return LLM_INVALID_ARGUMENT;
    }
    (void)memset(query_gradient, 0,
                 batch_count * query_length * query_head_count * head_dimension * sizeof(float));
    (void)memset(key_gradient, 0,
                 batch_count * key_length * key_value_head_count * head_dimension * sizeof(float));
    (void)memset(value_gradient, 0,
                 batch_count * key_length * key_value_head_count * head_dimension * sizeof(float));
    const size_t heads_per_group = query_head_count / key_value_head_count;
    for (size_t batch = 0U; batch < batch_count; ++batch) {
        for (size_t query_position = 0U; query_position < query_length; ++query_position) {
            size_t visible_keys = query_position_offset + query_position + 1U;
            if (visible_keys > key_length) {
                visible_keys = key_length;
            }
            for (size_t query_head = 0U; query_head < query_head_count; ++query_head) {
                const size_t key_value_head = query_head / heads_per_group;
                const size_t query_index =
                    attention_offset(batch, query_position, query_head, query_length,
                                     query_head_count, head_dimension);
                const float *query_row = query + query_index;
                const float *output_gradient_row = output_gradient + query_index;
                float *query_gradient_row = query_gradient + query_index;
                float maximum = -INFINITY;
                llm_status status = LLM_OK;
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const float *key_row =
                        key + attention_offset(batch, key_position, key_value_head, key_length,
                                               key_value_head_count, head_dimension);
                    const float score =
                        attention_score(query_row, key_row, head_dimension, scale, &status);
                    if (status != LLM_OK) {
                        return status;
                    }
                    if (score > maximum) {
                        maximum = score;
                    }
                }
                float denominator = 0.0F;
                float weighted_probability_gradient = 0.0F;
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const size_t key_index =
                        attention_offset(batch, key_position, key_value_head, key_length,
                                         key_value_head_count, head_dimension);
                    const float score =
                        attention_score(query_row, key + key_index, head_dimension, scale, &status);
                    denominator += expf(score - maximum);
                }
                if (status != LLM_OK || !isfinite(denominator) || denominator <= 0.0F) {
                    return LLM_NUMERICAL_ERROR;
                }
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const size_t key_index =
                        attention_offset(batch, key_position, key_value_head, key_length,
                                         key_value_head_count, head_dimension);
                    const float probability = expf(attention_score(query_row, key + key_index,
                                                                   head_dimension, scale, &status) -
                                                   maximum) /
                                              denominator;
                    float probability_gradient = 0.0F;
                    for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                        if (!isfinite(output_gradient_row[dimension]) ||
                            !isfinite(value[key_index + dimension])) {
                            return LLM_NUMERICAL_ERROR;
                        }
                        probability_gradient +=
                            output_gradient_row[dimension] * value[key_index + dimension];
                    }
                    weighted_probability_gradient += probability * probability_gradient;
                }
                for (size_t key_position = 0U; key_position < visible_keys; ++key_position) {
                    const size_t key_index =
                        attention_offset(batch, key_position, key_value_head, key_length,
                                         key_value_head_count, head_dimension);
                    const float probability = expf(attention_score(query_row, key + key_index,
                                                                   head_dimension, scale, &status) -
                                                   maximum) /
                                              denominator;
                    float probability_gradient = 0.0F;
                    for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                        probability_gradient +=
                            output_gradient_row[dimension] * value[key_index + dimension];
                    }
                    const float score_gradient =
                        probability * (probability_gradient - weighted_probability_gradient);
                    for (size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                        query_gradient_row[dimension] +=
                            scale * score_gradient * key[key_index + dimension];
                        key_gradient[key_index + dimension] +=
                            scale * score_gradient * query_row[dimension];
                        value_gradient[key_index + dimension] +=
                            probability * output_gradient_row[dimension];
                    }
                }
            }
        }
    }
    const size_t query_value_count = batch_count * query_length * query_head_count * head_dimension;
    const size_t key_value_count = batch_count * key_length * key_value_head_count * head_dimension;
    for (size_t index = 0U; index < query_value_count; ++index) {
        if (!isfinite(query_gradient[index])) {
            return LLM_NUMERICAL_ERROR;
        }
    }
    for (size_t index = 0U; index < key_value_count; ++index) {
        if (!isfinite(key_gradient[index]) || !isfinite(value_gradient[index])) {
            return LLM_NUMERICAL_ERROR;
        }
    }
    return LLM_OK;
}
