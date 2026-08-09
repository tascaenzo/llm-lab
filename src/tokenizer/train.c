#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tokenizer_internal.h"
#include "train_data.h"

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

static int compare_paths(const void *left, const void *right) {
    const char *const *left_path = left;
    const char *const *right_path = right;
    return strcmp(*left_path, *right_path);
}

static tokenizer_status order_input_paths(const char *const *input_paths, size_t input_count,
                                          const char ***out_ordered_paths) {
    if (tokenizer_allocation_would_overflow(input_count, sizeof(**out_ordered_paths))) {
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
    *out_ordered_paths = ordered_paths;
    return TOKENIZER_OK;
}

static tokenizer_status measure_input_files(const char *const *paths, size_t path_count,
                                            train_progress *progress) {
    for (size_t index = 0U; index < path_count; ++index) {
        uint64_t file_size = 0U;
        const tokenizer_status status = input_file_size(paths[index], &file_size);
        if (status != TOKENIZER_OK) {
            return status;
        }
        if (file_size > UINT64_MAX - progress->total_input_bytes) {
            return TOKENIZER_OVERFLOW;
        }
        progress->total_input_bytes += file_size;
    }
    return TOKENIZER_OK;
}

static tokenizer_status read_input_files(const char *const *paths, size_t path_count,
                                         tokenizer_training_data *data, train_progress *progress) {
    report_progress(progress, TOKENIZER_TRAIN_READING_INPUT, 0U, progress->total_input_bytes);
    for (size_t index = 0U; index < path_count; ++index) {
        const tokenizer_status status =
            tokenizer_pretokenize_file(paths[index], tokenizer_training_collect_pretoken,
                                       report_file_progress, progress, data);
        if (status != TOKENIZER_OK) {
            return status;
        }

        uint64_t file_size = 0U;
        const tokenizer_status size_status = input_file_size(paths[index], &file_size);
        if (size_status != TOKENIZER_OK) {
            return size_status;
        }
        progress->bytes_before_current_file += file_size;
    }
    return TOKENIZER_OK;
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

    const char **ordered_paths = NULL;
    tokenizer_status status = order_input_paths(input_paths, input_count, &ordered_paths);
    tokenizer_training_data *data = NULL;
    if (status == TOKENIZER_OK) {
        status = tokenizer_training_data_create(&data);
    }

    train_progress progress = {
        .callback = progress_callback,
        .context = progress_context,
    };
    if (status == TOKENIZER_OK) {
        status = measure_input_files(ordered_paths, input_count, &progress);
    }
    if (status == TOKENIZER_OK) {
        status = read_input_files(ordered_paths, input_count, data, &progress);
    }
    free(ordered_paths);

    tokenizer *tokenizer = NULL;
    if (status == TOKENIZER_OK) {
        status = tokenizer_create_byte_level(&tokenizer);
    }
    if (status == TOKENIZER_OK) {
        status = tokenizer_training_prepare(data, progress_callback, progress_context);
    }

    const uint64_t target_merges =
        (uint64_t)target_vocabulary_size - TOKENIZER_BYTE_VOCABULARY_SIZE;
    report_progress(&progress, TOKENIZER_TRAIN_MERGING, 0U, target_merges);
    while (status == TOKENIZER_OK && tokenizer->vocabulary_size < target_vocabulary_size) {
        int did_merge = 0;
        status = tokenizer_training_merge_next(data, tokenizer, &did_merge);
        if (status != TOKENIZER_OK || did_merge == 0) {
            break;
        }
        report_progress(&progress, TOKENIZER_TRAIN_MERGING,
                        tokenizer->vocabulary_size - TOKENIZER_BYTE_VOCABULARY_SIZE, target_merges);
    }

    tokenizer_training_data_destroy(data);
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
