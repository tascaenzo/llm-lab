#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tokenizer_internal.h"

typedef enum pretoken_kind {
    PRETOKEN_WHITESPACE,
    PRETOKEN_WORD,
    PRETOKEN_PUNCTUATION
} pretoken_kind;

typedef struct byte_buffer {
    unsigned char *data;
    size_t length;
    size_t capacity;
} byte_buffer;

static pretoken_kind classify_byte(unsigned char byte) {
    if (byte == ' ' || (byte >= '\t' && byte <= '\r')) {
        return PRETOKEN_WHITESPACE;
    }

    if ((byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') ||
        (byte >= 'a' && byte <= 'z') || byte == '_' || byte == '\'' || byte >= 0x80U) {
        return PRETOKEN_WORD;
    }

    return PRETOKEN_PUNCTUATION;
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

static tokenizer_status flush_buffer(byte_buffer *buffer, tokenizer_pretoken_callback callback,
                                     void *context) {
    if (buffer->length == 0U) {
        return TOKENIZER_OK;
    }

    const tokenizer_status status = callback(buffer->data, buffer->length, context);
    buffer->length = 0U;
    return status;
}

static tokenizer_status consume_byte(byte_buffer *buffer, int *has_kind,
                                     pretoken_kind *current_kind, unsigned char byte,
                                     tokenizer_pretoken_callback callback, void *context) {
    const pretoken_kind next_kind = classify_byte(byte);
    tokenizer_status status = TOKENIZER_OK;

    if (next_kind == PRETOKEN_PUNCTUATION) {
        status = flush_buffer(buffer, callback, context);
        if (status != TOKENIZER_OK) {
            return status;
        }

        *has_kind = 0;
        status = byte_buffer_append(buffer, byte);
        if (status != TOKENIZER_OK) {
            return status;
        }

        return flush_buffer(buffer, callback, context);
    }

    if (*has_kind != 0 && *current_kind != next_kind) {
        status = flush_buffer(buffer, callback, context);
        if (status != TOKENIZER_OK) {
            return status;
        }
    }

    *current_kind = next_kind;
    *has_kind = 1;
    return byte_buffer_append(buffer, byte);
}

static tokenizer_status finalize_buffer(byte_buffer *buffer, tokenizer_pretoken_callback callback,
                                        void *context) {
    const tokenizer_status status = flush_buffer(buffer, callback, context);
    free(buffer->data);
    return status;
}

tokenizer_status tokenizer_pretokenize_bytes(const unsigned char *input, size_t input_length,
                                             tokenizer_pretoken_callback callback, void *context) {
    if ((input == NULL && input_length != 0U) || callback == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    byte_buffer buffer = {0};
    int has_kind = 0;
    pretoken_kind current_kind = PRETOKEN_PUNCTUATION;

    for (size_t index = 0U; index < input_length; ++index) {
        const tokenizer_status status =
            consume_byte(&buffer, &has_kind, &current_kind, input[index], callback, context);
        if (status != TOKENIZER_OK) {
            free(buffer.data);
            return status;
        }
    }

    return finalize_buffer(&buffer, callback, context);
}

tokenizer_status tokenizer_pretokenize_file(const char *path, tokenizer_pretoken_callback callback,
                                            tokenizer_file_progress_callback progress_callback,
                                            void *progress_context, void *context) {
    if (path == NULL || callback == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return TOKENIZER_IO_ERROR;
    }

    byte_buffer buffer = {0};
    int has_kind = 0;
    pretoken_kind current_kind = PRETOKEN_PUNCTUATION;
    tokenizer_status status = TOKENIZER_OK;
    unsigned char input_buffer[64U * 1024U];
    uint64_t bytes_read = 0U;

    for (;;) {
        if (feof(file) != 0 || ferror(file) != 0) {
            break;
        }
        const size_t read_count = fread(input_buffer, 1U, sizeof(input_buffer), file);
        if (read_count == 0U) {
            break;
        }

        for (size_t index = 0U; index < read_count; ++index) {
            status = consume_byte(&buffer, &has_kind, &current_kind, input_buffer[index], callback,
                                  context);
            if (status != TOKENIZER_OK) {
                break;
            }
        }
        if (status != TOKENIZER_OK) {
            break;
        }

        if ((uint64_t)read_count > UINT64_MAX - bytes_read) {
            status = TOKENIZER_OVERFLOW;
            break;
        }
        bytes_read += (uint64_t)read_count;
        if (progress_callback != NULL) {
            progress_callback(bytes_read, progress_context);
        }
    }

    if (status == TOKENIZER_OK && ferror(file) != 0) {
        status = TOKENIZER_IO_ERROR;
    }

    if (fclose(file) != 0 && status == TOKENIZER_OK) {
        status = TOKENIZER_IO_ERROR;
    }

    if (status == TOKENIZER_OK) {
        return finalize_buffer(&buffer, callback, context);
    }

    free(buffer.data);
    return status;
}
