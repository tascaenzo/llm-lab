#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tokenizer_internal.h"

typedef struct train_word {
    unsigned char *bytes;
    size_t byte_length;
    token_id *ids;
    size_t token_length;
    uint64_t frequency;
    uint64_t hash;
    uint64_t selection_generation;
    int occupied;
} train_word;

typedef struct word_table {
    train_word *entries;
    size_t capacity;
    size_t size;
} word_table;

typedef struct pair_entry {
    uint64_t key;
    uint64_t count;
    uint64_t generation;
    size_t *word_indices;
    size_t word_count;
    size_t word_capacity;
    size_t last_word_index;
    int has_last_word_index;
    int occupied;
} pair_entry;

typedef struct pair_table {
    pair_entry *entries;
    size_t capacity;
    size_t size;
} pair_table;

typedef struct pair_candidate {
    uint64_t key;
    uint64_t count;
    uint64_t generation;
} pair_candidate;

typedef struct pair_heap {
    pair_candidate *entries;
    size_t size;
    size_t capacity;
} pair_heap;

typedef struct train_progress {
    tokenizer_train_progress_callback callback;
    void *context;
    uint64_t bytes_before_current_file;
    uint64_t total_input_bytes;
} train_progress;

static void report_progress(const train_progress *progress, tokenizer_train_phase phase,
                            uint64_t completed, uint64_t total) {
    if (progress->callback != NULL) {
        progress->callback(phase, completed, total, progress->context);
    }
}

static void report_file_progress(uint64_t bytes_read, void *context) {
    train_progress *progress = context;
    report_progress(progress, TOKENIZER_TRAIN_READING_INPUT,
                    progress->bytes_before_current_file + bytes_read, progress->total_input_bytes);
}

static tokenizer_status input_file_size(const char *path, uint64_t *out_size) {
    struct stat file_status;
    if (stat(path, &file_status) != 0 || file_status.st_size < 0) {
        return TOKENIZER_IO_ERROR;
    }

    *out_size = (uint64_t)file_status.st_size;
    return TOKENIZER_OK;
}

static int allocation_would_overflow(size_t count, size_t element_size) {
    return element_size != 0U && count > (SIZE_MAX / element_size);
}

static uint64_t mix_hash(uint64_t value) {
    value ^= value >> 30U;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27U;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31U;
    return value;
}

static uint64_t hash_bytes(const unsigned char *bytes, size_t length) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0U; index < length; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return mix_hash(hash);
}

static tokenizer_status word_table_rehash(word_table *table, size_t new_capacity) {
    if (allocation_would_overflow(new_capacity, sizeof(*table->entries))) {
        return TOKENIZER_OVERFLOW;
    }

    train_word *new_entries = calloc(new_capacity, sizeof(*new_entries));
    if (new_entries == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    for (size_t index = 0U; index < table->capacity; ++index) {
        train_word *entry = &table->entries[index];
        if (entry->occupied == 0) {
            continue;
        }

        size_t destination = (size_t)(entry->hash & (uint64_t)(new_capacity - 1U));
        while (new_entries[destination].occupied != 0) {
            destination = (destination + 1U) & (new_capacity - 1U);
        }

        new_entries[destination] = *entry;
    }

    free(table->entries);
    table->entries = new_entries;
    table->capacity = new_capacity;
    return TOKENIZER_OK;
}

static tokenizer_status word_table_reserve(word_table *table) {
    if (table->capacity == 0U) {
        return word_table_rehash(table, 1024U);
    }

    if ((table->size + 1U) * 10U < table->capacity * 7U) {
        return TOKENIZER_OK;
    }

    if (table->capacity > (SIZE_MAX / 2U)) {
        return TOKENIZER_OVERFLOW;
    }

    return word_table_rehash(table, table->capacity * 2U);
}

static train_word *word_table_find(word_table *table, const unsigned char *bytes, size_t length,
                                   uint64_t hash) {
    size_t index = (size_t)(hash & (uint64_t)(table->capacity - 1U));

    for (;;) {
        train_word *entry = &table->entries[index];
        if (entry->occupied == 0 || (entry->hash == hash && entry->byte_length == length &&
                                     memcmp(entry->bytes, bytes, length) == 0)) {
            return entry;
        }

        index = (index + 1U) & (table->capacity - 1U);
    }
}

static tokenizer_status word_table_add(word_table *table, const unsigned char *bytes,
                                       size_t length) {
    const tokenizer_status reserve_status = word_table_reserve(table);
    if (reserve_status != TOKENIZER_OK) {
        return reserve_status;
    }

    const uint64_t hash = hash_bytes(bytes, length);
    train_word *entry = word_table_find(table, bytes, length, hash);
    if (entry->occupied != 0) {
        if (entry->frequency == UINT64_MAX) {
            return TOKENIZER_OVERFLOW;
        }

        ++entry->frequency;
        return TOKENIZER_OK;
    }

    if (allocation_would_overflow(length, sizeof(*entry->ids))) {
        return TOKENIZER_OVERFLOW;
    }

    unsigned char *stored_bytes = malloc(length);
    token_id *ids = malloc(length * sizeof(*ids));
    if (stored_bytes == NULL || ids == NULL) {
        free(stored_bytes);
        free(ids);
        return TOKENIZER_ALLOCATION_FAILED;
    }

    memcpy(stored_bytes, bytes, length);
    for (size_t index = 0U; index < length; ++index) {
        ids[index] = bytes[index];
    }

    entry->bytes = stored_bytes;
    entry->byte_length = length;
    entry->ids = ids;
    entry->token_length = length;
    entry->frequency = 1U;
    entry->hash = hash;
    entry->occupied = 1;
    ++table->size;
    return TOKENIZER_OK;
}

static void word_table_destroy(word_table *table) {
    for (size_t index = 0U; index < table->capacity; ++index) {
        if (table->entries[index].occupied != 0) {
            free(table->entries[index].bytes);
            free(table->entries[index].ids);
        }
    }

    free(table->entries);
}

static tokenizer_status collect_word(const unsigned char *bytes, size_t length, void *context) {
    return word_table_add(context, bytes, length);
}

static tokenizer_status pair_table_rehash(pair_table *table, size_t new_capacity) {
    if (allocation_would_overflow(new_capacity, sizeof(*table->entries))) {
        return TOKENIZER_OVERFLOW;
    }

    pair_entry *new_entries = calloc(new_capacity, sizeof(*new_entries));
    if (new_entries == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    for (size_t index = 0U; index < table->capacity; ++index) {
        const pair_entry entry = table->entries[index];
        if (entry.occupied == 0) {
            continue;
        }

        size_t destination = (size_t)(mix_hash(entry.key) & (uint64_t)(new_capacity - 1U));
        while (new_entries[destination].occupied != 0) {
            destination = (destination + 1U) & (new_capacity - 1U);
        }

        new_entries[destination] = entry;
    }

    free(table->entries);
    table->entries = new_entries;
    table->capacity = new_capacity;
    return TOKENIZER_OK;
}

static tokenizer_status pair_table_reserve(pair_table *table) {
    if (table->capacity == 0U) {
        return pair_table_rehash(table, 1024U);
    }

    if ((table->size + 1U) * 10U < table->capacity * 7U) {
        return TOKENIZER_OK;
    }

    if (table->capacity > (SIZE_MAX / 2U)) {
        return TOKENIZER_OVERFLOW;
    }

    return pair_table_rehash(table, table->capacity * 2U);
}

static pair_entry *pair_table_find(pair_table *table, uint64_t key) {
    if (table->capacity == 0U) {
        return NULL;
    }

    size_t index = (size_t)(mix_hash(key) & (uint64_t)(table->capacity - 1U));
    for (;;) {
        pair_entry *entry = &table->entries[index];
        if (entry->occupied == 0) {
            return NULL;
        }
        if (entry->key == key) {
            return entry;
        }

        index = (index + 1U) & (table->capacity - 1U);
    }
}

static int candidate_is_higher_priority(const pair_candidate *left, const pair_candidate *right) {
    return left->count > right->count || (left->count == right->count && left->key < right->key);
}

static tokenizer_status pair_heap_reserve(pair_heap *heap) {
    if (heap->size < heap->capacity) {
        return TOKENIZER_OK;
    }

    const size_t new_capacity = heap->capacity == 0U ? 1024U : heap->capacity * 2U;
    if (new_capacity < heap->capacity ||
        allocation_would_overflow(new_capacity, sizeof(*heap->entries))) {
        return TOKENIZER_OVERFLOW;
    }

    pair_candidate *new_entries = realloc(heap->entries, new_capacity * sizeof(*new_entries));
    if (new_entries == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    heap->entries = new_entries;
    heap->capacity = new_capacity;
    return TOKENIZER_OK;
}

static tokenizer_status pair_heap_push(pair_heap *heap, uint64_t key, uint64_t count,
                                       uint64_t generation) {
    if (count == 0U) {
        return TOKENIZER_OK;
    }

    const tokenizer_status reserve_status = pair_heap_reserve(heap);
    if (reserve_status != TOKENIZER_OK) {
        return reserve_status;
    }

    size_t index = heap->size;
    ++heap->size;
    const pair_candidate candidate = {key, count, generation};
    while (index > 0U) {
        const size_t parent = (index - 1U) / 2U;
        if (candidate_is_higher_priority(&heap->entries[parent], &candidate) != 0) {
            break;
        }

        heap->entries[index] = heap->entries[parent];
        index = parent;
    }

    heap->entries[index] = candidate;
    return TOKENIZER_OK;
}

static int pair_heap_pop(pair_heap *heap, pair_candidate *out_candidate) {
    if (heap->size == 0U) {
        return 0;
    }

    *out_candidate = heap->entries[0];
    --heap->size;
    if (heap->size == 0U) {
        return 1;
    }

    const pair_candidate last = heap->entries[heap->size];
    size_t index = 0U;
    while (index * 2U + 1U < heap->size) {
        size_t child = index * 2U + 1U;
        if (child + 1U < heap->size &&
            candidate_is_higher_priority(&heap->entries[child + 1U], &heap->entries[child]) != 0) {
            ++child;
        }

        if (candidate_is_higher_priority(&last, &heap->entries[child]) != 0) {
            break;
        }

        heap->entries[index] = heap->entries[child];
        index = child;
    }

    heap->entries[index] = last;
    return 1;
}

static tokenizer_status pair_heap_rebuild(pair_heap *heap, const pair_table *pairs) {
    heap->size = 0U;
    for (size_t index = 0U; index < pairs->capacity; ++index) {
        const pair_entry *entry = &pairs->entries[index];
        if (entry->occupied == 0 || entry->count == 0U) {
            continue;
        }

        const tokenizer_status status =
            pair_heap_push(heap, entry->key, entry->count, entry->generation);
        if (status != TOKENIZER_OK) {
            return status;
        }
    }

    return TOKENIZER_OK;
}

static void pair_heap_destroy(pair_heap *heap) { free(heap->entries); }

static tokenizer_status pair_table_adjust(pair_table *table, pair_heap *heap, uint64_t key,
                                          uint64_t amount, int add) {
    pair_entry *entry = pair_table_find(table, key);
    if (entry == NULL) {
        if (add == 0) {
            return TOKENIZER_INVALID_MODEL;
        }

        const tokenizer_status reserve_status = pair_table_reserve(table);
        if (reserve_status != TOKENIZER_OK) {
            return reserve_status;
        }

        size_t index = (size_t)(mix_hash(key) & (uint64_t)(table->capacity - 1U));
        while (table->entries[index].occupied != 0) {
            index = (index + 1U) & (table->capacity - 1U);
        }

        entry = &table->entries[index];
        entry->key = key;
        entry->occupied = 1;
        ++table->size;
    }

    if (add != 0) {
        if (amount > UINT64_MAX - entry->count) {
            return TOKENIZER_OVERFLOW;
        }
        entry->count += amount;
    } else {
        if (amount > entry->count) {
            return TOKENIZER_INVALID_MODEL;
        }
        entry->count -= amount;
    }

    if (entry->generation == UINT64_MAX) {
        return TOKENIZER_OVERFLOW;
    }
    ++entry->generation;

    return heap == NULL ? TOKENIZER_OK
                        : pair_heap_push(heap, entry->key, entry->count, entry->generation);
}

static tokenizer_status pair_entry_add_word(pair_entry *entry, size_t word_index) {
    if (entry->has_last_word_index != 0 && entry->last_word_index == word_index) {
        return TOKENIZER_OK;
    }

    if (entry->word_count == entry->word_capacity) {
        const size_t new_capacity = entry->word_capacity == 0U ? 4U : entry->word_capacity * 2U;
        if (new_capacity < entry->word_capacity ||
            allocation_would_overflow(new_capacity, sizeof(*entry->word_indices))) {
            return TOKENIZER_OVERFLOW;
        }

        size_t *new_indices =
            realloc(entry->word_indices, new_capacity * sizeof(*entry->word_indices));
        if (new_indices == NULL) {
            return TOKENIZER_ALLOCATION_FAILED;
        }

        entry->word_indices = new_indices;
        entry->word_capacity = new_capacity;
    }

    entry->word_indices[entry->word_count] = word_index;
    ++entry->word_count;
    entry->last_word_index = word_index;
    entry->has_last_word_index = 1;
    return TOKENIZER_OK;
}

static tokenizer_status pair_table_add_occurrence(pair_table *table, pair_heap *heap, uint64_t key,
                                                  uint64_t frequency, size_t word_index) {
    tokenizer_status status = pair_table_adjust(table, heap, key, frequency, 1);
    if (status != TOKENIZER_OK) {
        return status;
    }

    pair_entry *entry = pair_table_find(table, key);
    return entry == NULL ? TOKENIZER_INVALID_MODEL : pair_entry_add_word(entry, word_index);
}

static void pair_table_destroy(pair_table *table) {
    for (size_t index = 0U; index < table->capacity; ++index) {
        free(table->entries[index].word_indices);
    }
    free(table->entries);
}

static uint64_t pair_key(token_id left_id, token_id right_id) {
    return ((uint64_t)left_id << 32U) | right_id;
}

static tokenizer_status add_word_pairs(const train_word *word, size_t word_index, pair_table *pairs,
                                       pair_heap *heap) {
    for (size_t index = 0U; index + 1U < word->token_length; ++index) {
        const tokenizer_status status = pair_table_add_occurrence(
            pairs, heap, pair_key(word->ids[index], word->ids[index + 1U]), word->frequency,
            word_index);
        if (status != TOKENIZER_OK) {
            return status;
        }
    }

    return TOKENIZER_OK;
}

static tokenizer_status remove_word_pairs(const train_word *word, pair_table *pairs,
                                          pair_heap *heap) {
    for (size_t index = 0U; index + 1U < word->token_length; ++index) {
        const tokenizer_status status = pair_table_adjust(
            pairs, heap, pair_key(word->ids[index], word->ids[index + 1U]), word->frequency, 0);
        if (status != TOKENIZER_OK) {
            return status;
        }
    }

    return TOKENIZER_OK;
}

static tokenizer_status collect_pairs(const word_table *words, pair_table *pairs,
                                      const train_progress *progress) {
    for (size_t word_index = 0U; word_index < words->capacity; ++word_index) {
        if (word_index % 4096U == 0U) {
            report_progress(progress, TOKENIZER_TRAIN_COLLECTING_PAIRS, word_index,
                            words->capacity);
        }

        const train_word *word = &words->entries[word_index];
        if (word->occupied == 0 || word->token_length < 2U) {
            continue;
        }

        const tokenizer_status status = add_word_pairs(word, word_index, pairs, NULL);
        if (status != TOKENIZER_OK) {
            return status;
        }
    }

    report_progress(progress, TOKENIZER_TRAIN_COLLECTING_PAIRS, words->capacity, words->capacity);

    return TOKENIZER_OK;
}

static int pair_heap_find_best(pair_heap *heap, pair_table *pairs, uint64_t *out_key) {
    pair_candidate candidate;
    while (pair_heap_pop(heap, &candidate) != 0) {
        pair_entry *entry = pair_table_find(pairs, candidate.key);
        if (entry != NULL && entry->count != 0U && entry->count == candidate.count &&
            entry->generation == candidate.generation) {
            *out_key = candidate.key;
            return 1;
        }
    }

    return 0;
}

static int word_has_pair(const train_word *word, token_id left_id, token_id right_id) {
    for (size_t index = 0U; index + 1U < word->token_length; ++index) {
        if (word->ids[index] == left_id && word->ids[index + 1U] == right_id) {
            return 1;
        }
    }

    return 0;
}

static void apply_merge(train_word *word, token_id left_id, token_id right_id, token_id merged_id) {
    size_t read_index = 0U;
    size_t write_index = 0U;

    while (read_index < word->token_length) {
        if (read_index + 1U < word->token_length && word->ids[read_index] == left_id &&
            word->ids[read_index + 1U] == right_id) {
            word->ids[write_index] = merged_id;
            ++write_index;
            read_index += 2U;
        } else {
            word->ids[write_index] = word->ids[read_index];
            ++write_index;
            ++read_index;
        }
    }

    word->token_length = write_index;
}

static tokenizer_status apply_merge_to_affected_words(word_table *words, pair_table *pairs,
                                                      pair_heap *heap, uint64_t selected_key,
                                                      token_id left_id, token_id right_id,
                                                      token_id merged_id,
                                                      uint64_t selection_generation) {
    pair_entry *selected_entry = pair_table_find(pairs, selected_key);
    if (selected_entry == NULL) {
        return TOKENIZER_INVALID_MODEL;
    }

    size_t *candidate_indices = selected_entry->word_indices;
    const size_t candidate_count = selected_entry->word_count;
    selected_entry->word_indices = NULL;
    selected_entry->word_count = 0U;
    selected_entry->word_capacity = 0U;

    tokenizer_status status = TOKENIZER_OK;
    for (size_t index = 0U; index < candidate_count && status == TOKENIZER_OK; ++index) {
        const size_t word_index = candidate_indices[index];
        if (word_index >= words->capacity) {
            status = TOKENIZER_INVALID_MODEL;
            break;
        }

        train_word *word = &words->entries[word_index];
        if (word->occupied == 0 || word->selection_generation == selection_generation) {
            continue;
        }
        word->selection_generation = selection_generation;

        if (word_has_pair(word, left_id, right_id) == 0) {
            continue;
        }

        status = remove_word_pairs(word, pairs, heap);
        if (status == TOKENIZER_OK) {
            apply_merge(word, left_id, right_id, merged_id);
            status = add_word_pairs(word, word_index, pairs, heap);
        }
    }
    free(candidate_indices);

    selected_entry = pair_table_find(pairs, selected_key);
    if (status == TOKENIZER_OK && (selected_entry == NULL || selected_entry->count != 0U)) {
        return TOKENIZER_INVALID_MODEL;
    }

    return status;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *left_path = left;
    const char *const *right_path = right;
    return strcmp(*left_path, *right_path);
}

tokenizer_status tokenizer_train_with_progress(const char *const *input_paths, size_t input_count,
                                               uint32_t target_vocabulary_size,
                                               tokenizer_train_progress_callback progress_callback,
                                               void *progress_context, tokenizer **out_tokenizer) {
    if (input_paths == NULL || input_count == 0U ||
        target_vocabulary_size < TOKENIZER_BYTE_VOCABULARY_SIZE || out_tokenizer == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    *out_tokenizer = NULL;

    if (allocation_would_overflow(input_count, sizeof(*input_paths))) {
        return TOKENIZER_OVERFLOW;
    }

    const char **ordered_paths = malloc(input_count * sizeof(*ordered_paths));
    if (ordered_paths == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }

    for (size_t index = 0U; index < input_count; ++index) {
        if (input_paths[index] == NULL) {
            free(ordered_paths);
            return TOKENIZER_INVALID_ARGUMENT;
        }
        ordered_paths[index] = input_paths[index];
    }
    qsort(ordered_paths, input_count, sizeof(*ordered_paths), compare_paths);

    train_progress progress = {
        .callback = progress_callback,
        .context = progress_context,
        .bytes_before_current_file = 0U,
        .total_input_bytes = 0U,
    };
    word_table words = {0};
    tokenizer_status status = TOKENIZER_OK;
    for (size_t index = 0U; index < input_count && status == TOKENIZER_OK; ++index) {
        uint64_t file_size = 0U;
        status = input_file_size(ordered_paths[index], &file_size);
        if (status == TOKENIZER_OK) {
            if (file_size > UINT64_MAX - progress.total_input_bytes) {
                status = TOKENIZER_OVERFLOW;
            } else {
                progress.total_input_bytes += file_size;
            }
        }
    }
    report_progress(&progress, TOKENIZER_TRAIN_READING_INPUT, 0U, progress.total_input_bytes);
    for (size_t index = 0U; index < input_count && status == TOKENIZER_OK; ++index) {
        status = tokenizer_pretokenize_file(ordered_paths[index], collect_word,
                                            report_file_progress, &progress, &words);
        if (status == TOKENIZER_OK) {
            uint64_t file_size = 0U;
            status = input_file_size(ordered_paths[index], &file_size);
            if (status == TOKENIZER_OK) {
                progress.bytes_before_current_file += file_size;
            }
        }
    }
    free(ordered_paths);

    tokenizer *tokenizer = NULL;
    pair_table pairs = {0};
    pair_heap heap = {0};
    if (status == TOKENIZER_OK) {
        status = tokenizer_create_byte_level(&tokenizer);
    }
    if (status == TOKENIZER_OK) {
        status = collect_pairs(&words, &pairs, &progress);
    }
    if (status == TOKENIZER_OK) {
        report_progress(&progress, TOKENIZER_TRAIN_BUILDING_HEAP, 0U, 1U);
        status = pair_heap_rebuild(&heap, &pairs);
        if (status == TOKENIZER_OK) {
            report_progress(&progress, TOKENIZER_TRAIN_BUILDING_HEAP, 1U, 1U);
        }
    }

    uint64_t selection_generation = 0U;
    const uint64_t target_merges =
        (uint64_t)target_vocabulary_size - TOKENIZER_BYTE_VOCABULARY_SIZE;
    report_progress(&progress, TOKENIZER_TRAIN_MERGING, 0U, target_merges);
    while (status == TOKENIZER_OK && tokenizer->vocabulary_size < target_vocabulary_size) {
        if (pairs.size != 0U && heap.size > pairs.size * 4U) {
            status = pair_heap_rebuild(&heap, &pairs);
            if (status != TOKENIZER_OK) {
                break;
            }
        }

        uint64_t best_pair = 0U;
        if (pair_heap_find_best(&heap, &pairs, &best_pair) == 0) {
            break;
        }

        if (selection_generation == UINT64_MAX) {
            status = TOKENIZER_OVERFLOW;
            break;
        }
        ++selection_generation;

        const token_id left_id = (token_id)(best_pair >> 32U);
        const token_id right_id = (token_id)best_pair;
        const token_id merged_id = tokenizer->vocabulary_size;
        status = tokenizer_append_merge(tokenizer, left_id, right_id);
        if (status == TOKENIZER_OK) {
            status = apply_merge_to_affected_words(&words, &pairs, &heap, best_pair, left_id,
                                                   right_id, merged_id, selection_generation);
            if (status == TOKENIZER_OK) {
                report_progress(&progress, TOKENIZER_TRAIN_MERGING,
                                tokenizer->vocabulary_size - TOKENIZER_BYTE_VOCABULARY_SIZE,
                                target_merges);
            }
        }
    }

    pair_heap_destroy(&heap);
    pair_table_destroy(&pairs);
    word_table_destroy(&words);

    if (status != TOKENIZER_OK) {
        tokenizer_destroy(tokenizer);
        return status;
    }

    *out_tokenizer = tokenizer;
    return TOKENIZER_OK;
}

tokenizer_status tokenizer_train(const char *const *input_paths, size_t input_count,
                                 uint32_t target_vocabulary_size, tokenizer **out_tokenizer) {
    return tokenizer_train_with_progress(input_paths, input_count, target_vocabulary_size, NULL,
                                         NULL, out_tokenizer);
}
