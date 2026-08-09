#include <stdint.h>
#include <stdlib.h>

#include "tokenizer_internal.h"

typedef struct token_buffer {
    token_id *data;
    size_t length;
    size_t capacity;
} token_buffer;

typedef struct byte_buffer {
    unsigned char *data;
    size_t length;
    size_t capacity;
} byte_buffer;

typedef struct encode_context {
    const tokenizer *tokenizer;
    token_buffer *output;
} encode_context;

static tokenizer_status token_buffer_reserve(token_buffer *buffer, size_t required) {
    if (required <= buffer->capacity) {
        return TOKENIZER_OK;
    }

    size_t new_capacity = buffer->capacity == 0U ? 64U : buffer->capacity;
    while (new_capacity < required) {
        if (new_capacity > (SIZE_MAX / 2U)) {
            return TOKENIZER_OVERFLOW;
        }
        new_capacity *= 2U;
    }

    if (tokenizer_allocation_would_overflow(new_capacity, sizeof(*buffer->data))) {
        return TOKENIZER_OVERFLOW;
    }

    token_id *new_data = realloc(buffer->data, new_capacity * sizeof(*new_data));
    if (new_data == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    buffer->data = new_data;
    buffer->capacity = new_capacity;
    return TOKENIZER_OK;
}

static tokenizer_status token_buffer_append(token_buffer *buffer, token_id id) {
    const tokenizer_status status = token_buffer_reserve(buffer, buffer->length + 1U);
    if (status != TOKENIZER_OK) {
        return status;
    }

    buffer->data[buffer->length] = id;
    ++buffer->length;
    return TOKENIZER_OK;
}

static tokenizer_status byte_buffer_append(byte_buffer *buffer, unsigned char byte) {
    if (buffer->length == buffer->capacity) {
        const size_t new_capacity = buffer->capacity == 0U ? 64U : buffer->capacity * 2U;
        if (new_capacity < buffer->capacity ||
            tokenizer_allocation_would_overflow(new_capacity, sizeof(*buffer->data))) {
            return TOKENIZER_OVERFLOW;
        }

        unsigned char *new_data = realloc(buffer->data, new_capacity * sizeof(*new_data));
        if (new_data == NULL) {
            return TOKENIZER_ALLOCATION_FAILED;
        }

        buffer->data = new_data;
        buffer->capacity = new_capacity;
    }

    buffer->data[buffer->length] = byte;
    ++buffer->length;
    return TOKENIZER_OK;
}

static void apply_merge(token_id *ids, size_t *length, bpe_merge merge, token_id merged_id) {
    size_t read_index = 0U;
    size_t write_index = 0U;

    while (read_index < *length) {
        if (read_index + 1U < *length && ids[read_index] == merge.left_id &&
            ids[read_index + 1U] == merge.right_id) {
            ids[write_index] = merged_id;
            ++write_index;
            read_index += 2U;
        } else {
            ids[write_index] = ids[read_index];
            ++write_index;
            ++read_index;
        }
    }

    *length = write_index;
}

static tokenizer_status encode_pretoken(const unsigned char *bytes, size_t length, void *context) {
    encode_context *encode = context;
    if (length == 0U) {
        return TOKENIZER_OK;
    }

    if (tokenizer_allocation_would_overflow(length, sizeof(token_id))) {
        return TOKENIZER_OVERFLOW;
    }

    token_id *ids = malloc(length * sizeof(*ids));
    if (ids == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    for (size_t index = 0U; index < length; ++index) {
        ids[index] = bytes[index];
    }

    size_t token_length = length;
    for (;;) {
        token_id selected_id = 0U;
        int has_selected = 0;
        for (size_t index = 0U; index + 1U < token_length; ++index) {
            token_id candidate_id = 0U;
            if (tokenizer_find_merged_id(encode->tokenizer, ids[index], ids[index + 1U],
                                         &candidate_id) != 0 &&
                (has_selected == 0 || candidate_id < selected_id)) {
                selected_id = candidate_id;
                has_selected = 1;
            }
        }
        if (has_selected == 0) {
            break;
        }

        const size_t merge_index = (size_t)(selected_id - TOKENIZER_BYTE_VOCABULARY_SIZE);
        apply_merge(ids, &token_length, encode->tokenizer->merges[merge_index], selected_id);
    }

    if (token_length > SIZE_MAX - encode->output->length) {
        free(ids);
        return TOKENIZER_OVERFLOW;
    }

    const tokenizer_status status =
        token_buffer_reserve(encode->output, encode->output->length + token_length);
    if (status == TOKENIZER_OK) {
        for (size_t index = 0U; index < token_length; ++index) {
            encode->output->data[encode->output->length + index] = ids[index];
        }
        encode->output->length += token_length;
    }

    free(ids);
    return status;
}

tokenizer_status tokenizer_encode(const tokenizer *tokenizer, const unsigned char *input,
                                  size_t input_length, token_sequence *out_tokens) {
    if (out_tokens == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    out_tokens->ids = NULL;
    out_tokens->length = 0U;

    if (tokenizer == NULL || (input == NULL && input_length != 0U)) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    token_buffer output = {0};
    encode_context context = {.tokenizer = tokenizer, .output = &output};
    const tokenizer_status status =
        tokenizer_pretokenize_bytes(input, input_length, encode_pretoken, &context);
    if (status != TOKENIZER_OK) {
        free(output.data);
        return status;
    }

    out_tokens->ids = output.data;
    out_tokens->length = output.length;
    return TOKENIZER_OK;
}

static tokenizer_status push_id(token_buffer *stack, token_id id) {
    return token_buffer_append(stack, id);
}

static tokenizer_status decode_id(const tokenizer *tokenizer, token_id id, byte_buffer *output) {
    token_buffer stack = {0};
    tokenizer_status status = push_id(&stack, id);

    while (status == TOKENIZER_OK && stack.length != 0U) {
        --stack.length;
        const token_id current = stack.data[stack.length];

        if (current < TOKENIZER_BYTE_VOCABULARY_SIZE) {
            status = byte_buffer_append(output, (unsigned char)current);
            continue;
        }

        const uint64_t merge_index = (uint64_t)current - TOKENIZER_BYTE_VOCABULARY_SIZE;
        if (merge_index >= tokenizer->merge_count) {
            status = TOKENIZER_INVALID_TOKEN;
            continue;
        }

        const bpe_merge merge = tokenizer->merges[merge_index];
        status = push_id(&stack, merge.right_id);
        if (status == TOKENIZER_OK) {
            status = push_id(&stack, merge.left_id);
        }
    }

    free(stack.data);
    return status;
}

tokenizer_status tokenizer_decode(const tokenizer *tokenizer, const token_sequence *tokens,
                                  unsigned char **out_bytes, size_t *out_length) {
    if (out_bytes == NULL || out_length == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    *out_bytes = NULL;
    *out_length = 0U;

    if (tokenizer == NULL || tokens == NULL || (tokens->ids == NULL && tokens->length != 0U)) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    byte_buffer output = {0};
    tokenizer_status status = TOKENIZER_OK;

    for (size_t index = 0U; index < tokens->length && status == TOKENIZER_OK; ++index) {
        status = decode_id(tokenizer, tokens->ids[index], &output);
    }

    if (status != TOKENIZER_OK) {
        free(output.data);
        return status;
    }

    *out_bytes = output.data;
    *out_length = output.length;
    return TOKENIZER_OK;
}

void token_sequence_destroy(token_sequence *tokens) {
    if (tokens == NULL) {
        return;
    }

    free(tokens->ids);
    tokens->ids = NULL;
    tokens->length = 0U;
}

void tokenizer_bytes_destroy(unsigned char *bytes) { free(bytes); }

const char *tokenizer_status_string(tokenizer_status status) {
    switch (status) {
    case TOKENIZER_OK:
        return "ok";
    case TOKENIZER_INVALID_ARGUMENT:
        return "invalid argument";
    case TOKENIZER_ALLOCATION_FAILED:
        return "allocation failed";
    case TOKENIZER_OVERFLOW:
        return "overflow";
    case TOKENIZER_INVALID_TOKEN:
        return "invalid token";
    case TOKENIZER_IO_ERROR:
        return "I/O error";
    case TOKENIZER_INVALID_MODEL:
        return "invalid model";
    }

    return "unknown status";
}
