#include <stdint.h>
#include <stdlib.h>

#include "tokenizer_internal.h"

static uint64_t pair_key(token_id left_id, token_id right_id) {
    return ((uint64_t)left_id << 32U) | right_id;
}

static uint64_t mix_hash(uint64_t value) {
    value ^= value >> 30U;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27U;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31U;
    return value;
}

static tokenizer_status rank_table_rehash(tokenizer *tokenizer, size_t new_capacity) {
    if (tokenizer_allocation_would_overflow(new_capacity, sizeof(*tokenizer->rank_entries))) {
        return TOKENIZER_OVERFLOW;
    }

    bpe_rank_entry *new_entries = calloc(new_capacity, sizeof(*new_entries));
    if (new_entries == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    for (size_t index = 0U; index < tokenizer->rank_capacity; ++index) {
        const bpe_rank_entry entry = tokenizer->rank_entries[index];
        if (entry.occupied == 0) {
            continue;
        }

        size_t destination = (size_t)(mix_hash(entry.key) & (uint64_t)(new_capacity - 1U));
        while (new_entries[destination].occupied != 0) {
            destination = (destination + 1U) & (new_capacity - 1U);
        }
        new_entries[destination] = entry;
    }

    free(tokenizer->rank_entries);
    tokenizer->rank_entries = new_entries;
    tokenizer->rank_capacity = new_capacity;
    return TOKENIZER_OK;
}

static tokenizer_status rank_table_reserve(tokenizer *tokenizer) {
    if (tokenizer->rank_capacity == 0U) {
        return rank_table_rehash(tokenizer, 128U);
    }
    if (tokenizer->merge_count + 1U < tokenizer->rank_capacity - tokenizer->rank_capacity / 3U) {
        return TOKENIZER_OK;
    }
    if (tokenizer->rank_capacity > SIZE_MAX / 2U) {
        return TOKENIZER_OVERFLOW;
    }
    return rank_table_rehash(tokenizer, tokenizer->rank_capacity * 2U);
}

int tokenizer_find_merged_id(const tokenizer *tokenizer, token_id left_id, token_id right_id,
                             token_id *out_merged_id) {
    if (tokenizer == NULL || tokenizer->rank_capacity == 0U) {
        return 0;
    }

    const uint64_t key = pair_key(left_id, right_id);
    size_t index = (size_t)(mix_hash(key) & (uint64_t)(tokenizer->rank_capacity - 1U));
    for (;;) {
        const bpe_rank_entry *entry = &tokenizer->rank_entries[index];
        if (entry->occupied == 0) {
            return 0;
        }
        if (entry->key == key) {
            if (out_merged_id != NULL) {
                *out_merged_id = entry->merged_id;
            }
            return 1;
        }
        index = (index + 1U) & (tokenizer->rank_capacity - 1U);
    }
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
    free(tokenizer->rank_entries);
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
    if (tokenizer_find_merged_id(tokenizer, left_id, right_id, NULL) != 0) {
        return TOKENIZER_INVALID_MODEL;
    }

    if (tokenizer->merge_count == tokenizer->merge_capacity) {
        const size_t new_capacity =
            tokenizer->merge_capacity == 0U ? 64U : tokenizer->merge_capacity * 2U;
        if (new_capacity < tokenizer->merge_capacity ||
            tokenizer_allocation_would_overflow(new_capacity, sizeof(*tokenizer->merges))) {
            return TOKENIZER_OVERFLOW;
        }

        bpe_merge *new_merges = realloc(tokenizer->merges, new_capacity * sizeof(*new_merges));
        if (new_merges == NULL) {
            return TOKENIZER_ALLOCATION_FAILED;
        }

        tokenizer->merges = new_merges;
        tokenizer->merge_capacity = new_capacity;
    }

    const tokenizer_status rank_status = rank_table_reserve(tokenizer);
    if (rank_status != TOKENIZER_OK) {
        return rank_status;
    }

    tokenizer->merges[tokenizer->merge_count].left_id = left_id;
    tokenizer->merges[tokenizer->merge_count].right_id = right_id;
    const uint64_t key = pair_key(left_id, right_id);
    size_t rank_index = (size_t)(mix_hash(key) & (uint64_t)(tokenizer->rank_capacity - 1U));
    while (tokenizer->rank_entries[rank_index].occupied != 0) {
        rank_index = (rank_index + 1U) & (tokenizer->rank_capacity - 1U);
    }
    tokenizer->rank_entries[rank_index] = (bpe_rank_entry){
        .key = key,
        .merged_id = tokenizer->vocabulary_size,
        .occupied = 1,
    };
    ++tokenizer->merge_count;
    ++tokenizer->vocabulary_size;
    return TOKENIZER_OK;
}

tokenizer_status tokenizer_add_merge(tokenizer *tokenizer, token_id left_id, token_id right_id) {
    return tokenizer_append_merge(tokenizer, left_id, right_id);
}
