#include <stdint.h>
#include <stdlib.h>

#include "tokenizer_internal.h"

static int allocation_would_overflow(size_t count, size_t element_size) {
    return element_size != 0U && count > (SIZE_MAX / element_size);
}

tokenizer_status tokenizer_create_byte_level(tokenizer **out_tokenizer) {
    if (out_tokenizer == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    *out_tokenizer = NULL;

    tokenizer *new_tokenizer = calloc(1U, sizeof(*new_tokenizer));
    if (new_tokenizer == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    new_tokenizer->vocabulary_size = TOKENIZER_BYTE_VOCABULARY_SIZE;
    *out_tokenizer = new_tokenizer;
    return TOKENIZER_OK;
}

void tokenizer_destroy(tokenizer *tokenizer) {
    if (tokenizer == NULL) {
        return;
    }

    free(tokenizer->merges);
    free(tokenizer);
}

uint32_t tokenizer_vocabulary_size(const tokenizer *tokenizer) {
    return tokenizer == NULL ? 0U : tokenizer->vocabulary_size;
}

size_t tokenizer_merge_count(const tokenizer *tokenizer) {
    return tokenizer == NULL ? 0U : tokenizer->merge_count;
}

tokenizer_status tokenizer_append_merge(tokenizer *tokenizer, token_id left_id, token_id right_id) {
    if (tokenizer == NULL || left_id >= tokenizer->vocabulary_size ||
        right_id >= tokenizer->vocabulary_size || tokenizer->vocabulary_size == UINT32_MAX) {
        return TOKENIZER_INVALID_MODEL;
    }

    if (tokenizer->merge_count == tokenizer->merge_capacity) {
        const size_t new_capacity =
            tokenizer->merge_capacity == 0U ? 64U : tokenizer->merge_capacity * 2U;
        if (new_capacity < tokenizer->merge_capacity ||
            allocation_would_overflow(new_capacity, sizeof(*tokenizer->merges))) {
            return TOKENIZER_OVERFLOW;
        }

        bpe_merge *new_merges = realloc(tokenizer->merges, new_capacity * sizeof(*new_merges));
        if (new_merges == NULL) {
            return TOKENIZER_ALLOCATION_FAILED;
        }

        tokenizer->merges = new_merges;
        tokenizer->merge_capacity = new_capacity;
    }

    tokenizer->merges[tokenizer->merge_count].left_id = left_id;
    tokenizer->merges[tokenizer->merge_count].right_id = right_id;
    ++tokenizer->merge_count;
    ++tokenizer->vocabulary_size;
    return TOKENIZER_OK;
}

tokenizer_status tokenizer_add_merge(tokenizer *tokenizer, token_id left_id, token_id right_id) {
    if (tokenizer == NULL) {
        return TOKENIZER_INVALID_MODEL;
    }

    for (size_t index = 0U; index < tokenizer->merge_count; ++index) {
        if (tokenizer->merges[index].left_id == left_id &&
            tokenizer->merges[index].right_id == right_id) {
            return TOKENIZER_INVALID_MODEL;
        }
    }

    return tokenizer_append_merge(tokenizer, left_id, right_id);
}
