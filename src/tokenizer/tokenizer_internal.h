#ifndef TOKENIZER_INTERNAL_H
#define TOKENIZER_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "tokenizer/tokenizer.h"

typedef struct bpe_merge {
    token_id left_id;
    token_id right_id;
} bpe_merge;

struct tokenizer {
    bpe_merge *merges;
    size_t merge_count;
    size_t merge_capacity;
    uint32_t vocabulary_size;
};

typedef tokenizer_status (*tokenizer_pretoken_callback)(const unsigned char *bytes, size_t length,
                                                        void *context);

typedef void (*tokenizer_file_progress_callback)(uint64_t bytes_read, void *context);

tokenizer_status tokenizer_add_merge(tokenizer *tokenizer, token_id left_id, token_id right_id);
tokenizer_status tokenizer_append_merge(tokenizer *tokenizer, token_id left_id, token_id right_id);
tokenizer_status tokenizer_pretokenize_bytes(const unsigned char *input, size_t input_length,
                                             tokenizer_pretoken_callback callback, void *context);
tokenizer_status tokenizer_pretokenize_file(const char *path, tokenizer_pretoken_callback callback,
                                            tokenizer_file_progress_callback progress_callback,
                                            void *progress_context, void *context);

#endif
