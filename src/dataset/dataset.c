#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <io.h>
#endif

#include "dataset_internal.h"

static lm_dataset_status sha256_file(const char *path, unsigned char checksum[32]) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return LM_DATASET_IO_ERROR;
    }
    tokenizer_sha256_context sha256;
    tokenizer_sha256_init(&sha256);
    unsigned char buffer[64U * 1024U];
    for (;;) {
        const size_t read_count = fread(buffer, 1U, sizeof(buffer), file);
        if (read_count != 0U) {
            tokenizer_sha256_update(&sha256, buffer, read_count);
        }
        if (read_count < sizeof(buffer)) {
            break;
        }
    }
    lm_dataset_status status = ferror(file) == 0 ? LM_DATASET_OK : LM_DATASET_IO_ERROR;
    if (fclose(file) != 0) {
        status = LM_DATASET_IO_ERROR;
    }
    if (status == LM_DATASET_OK) {
        tokenizer_sha256_final(&sha256, checksum);
    }
    return status;
}

static lm_dataset_status file_size_and_rewind(FILE *file, uint64_t *out_size) {
#ifdef _WIN32
    if (_fseeki64(file, 0, SEEK_END) != 0) {
        return LM_DATASET_IO_ERROR;
    }
    const __int64 size = _ftelli64(file);
    if (size < 0 || _fseeki64(file, 0, SEEK_SET) != 0) {
        return LM_DATASET_IO_ERROR;
    }
    *out_size = (uint64_t)size;
#else
    if (fseek(file, 0L, SEEK_END) != 0) {
        return LM_DATASET_IO_ERROR;
    }
    const long size = ftell(file);
    if (size < 0 || fseek(file, 0L, SEEK_SET) != 0) {
        return LM_DATASET_IO_ERROR;
    }
    *out_size = (uint64_t)size;
#endif
    return LM_DATASET_OK;
}

lm_dataset_status lm_dataset_prepare_jsonl(const char *tokenizer_path,
                                           const char *documents_jsonl_path,
                                           const char *output_prefix,
                                           lm_dataset_prepare_report *out_report) {
    return lm_dataset_prepare_jsonl_with_progress(tokenizer_path, documents_jsonl_path,
                                                  output_prefix, NULL, NULL, out_report);
}

lm_dataset_status lm_dataset_prepare_jsonl_with_progress(
    const char *tokenizer_path, const char *documents_jsonl_path, const char *output_prefix,
    lm_dataset_progress_callback progress_callback, void *progress_context,
    lm_dataset_prepare_report *out_report) {
    if (tokenizer_path == NULL || documents_jsonl_path == NULL || output_prefix == NULL ||
        out_report == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_report = (lm_dataset_prepare_report){0};

    tokenizer *tokenizer = NULL;
    const tokenizer_status tokenizer_result = tokenizer_load(tokenizer_path, &tokenizer);
    if (tokenizer_result != TOKENIZER_OK) {
        return tokenizer_result == TOKENIZER_ALLOCATION_FAILED ? LM_DATASET_ALLOCATION_FAILED
                                                               : LM_DATASET_TOKENIZER_ERROR;
    }
    const uint32_t vocabulary_size = tokenizer_vocabulary_size(tokenizer);
    if (vocabulary_size == UINT32_MAX) {
        tokenizer_destroy(tokenizer);
        return LM_DATASET_OVERFLOW;
    }

    unsigned char tokenizer_checksum[32] = {0};
    lm_dataset_status status = sha256_file(tokenizer_path, tokenizer_checksum);
    FILE *documents = NULL;
    uint64_t total_bytes = 0U;
    if (status == LM_DATASET_OK) {
        documents = fopen(documents_jsonl_path, "rb");
        if (documents == NULL) {
            status = LM_DATASET_IO_ERROR;
        } else {
            status = file_size_and_rewind(documents, &total_bytes);
        }
    }

    lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT] = {0};
    for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT && status == LM_DATASET_OK; ++index) {
        status = lm_dataset_writer_open(&writers[index], output_prefix, (lm_dataset_split)index);
    }
    if (status == LM_DATASET_OK) {
        status = lm_dataset_parse_documents(documents, tokenizer, writers, total_bytes,
                                            progress_callback, progress_context);
    }
    if (documents != NULL && fclose(documents) != 0 && status == LM_DATASET_OK) {
        status = LM_DATASET_IO_ERROR;
    }
    if (status == LM_DATASET_OK) {
        status = lm_dataset_writers_publish(writers, vocabulary_size, tokenizer_checksum);
    }
    if (status == LM_DATASET_OK) {
        for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT; ++index) {
            out_report->document_counts[index] = writers[index].document_count;
            out_report->token_counts[index] = writers[index].token_count;
        }
        out_report->tokenizer_vocabulary_size = vocabulary_size;
        out_report->model_vocabulary_size = vocabulary_size + 1U;
        out_report->end_of_document_token = vocabulary_size;
        for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT; ++index) {
            writers[index].published = 0;
        }
    }
    lm_dataset_writers_abort(writers);
    tokenizer_destroy(tokenizer);
    return status;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12U;
    value ^= value << 25U;
    value ^= value >> 27U;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

lm_dataset_status lm_batcher_create(lm_dataset *dataset, size_t batch_size, size_t context_length,
                                    uint64_t seed, lm_batcher **out_batcher) {
    if (dataset == NULL || batch_size == 0U || context_length == 0U || out_batcher == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_batcher = NULL;
    if ((uint64_t)context_length >= dataset->token_count) {
        return LM_DATASET_INSUFFICIENT_DATA;
    }
    if (batch_size > SIZE_MAX / context_length || context_length == SIZE_MAX ||
        context_length + 1U > SIZE_MAX / sizeof(token_id)) {
        return LM_DATASET_OVERFLOW;
    }

    lm_batcher *batcher = calloc(1U, sizeof(*batcher));
    if (batcher == NULL) {
        return LM_DATASET_ALLOCATION_FAILED;
    }
    batcher->window = malloc((context_length + 1U) * sizeof(*batcher->window));
    if (batcher->window == NULL) {
        free(batcher);
        return LM_DATASET_ALLOCATION_FAILED;
    }
    batcher->dataset = dataset;
    batcher->batch_size = batch_size;
    batcher->context_length = context_length;
    batcher->random_state = seed == 0U ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    *out_batcher = batcher;
    return LM_DATASET_OK;
}

void lm_batcher_destroy(lm_batcher *batcher) {
    if (batcher == NULL) {
        return;
    }
    free(batcher->window);
    free(batcher);
}

size_t lm_batcher_batch_size(const lm_batcher *batcher) {
    return batcher == NULL ? 0U : batcher->batch_size;
}

size_t lm_batcher_context_length(const lm_batcher *batcher) {
    return batcher == NULL ? 0U : batcher->context_length;
}

lm_dataset_status lm_batcher_next(lm_batcher *batcher, token_id *out_inputs,
                                  token_id *out_targets) {
    if (batcher == NULL || out_inputs == NULL || out_targets == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    const uint64_t possible_offsets =
        batcher->dataset->token_count - (uint64_t)batcher->context_length;
    for (size_t row = 0U; row < batcher->batch_size; ++row) {
        const uint64_t offset = next_random(&batcher->random_state) % possible_offsets;
        lm_dataset_status status = lm_dataset_read_tokens(
            batcher->dataset, offset, batcher->context_length + 1U, batcher->window);
        if (status != LM_DATASET_OK) {
            return status;
        }
        const size_t row_offset = row * batcher->context_length;
        for (size_t column = 0U; column < batcher->context_length; ++column) {
            out_inputs[row_offset + column] = batcher->window[column];
            out_targets[row_offset + column] = batcher->window[column + 1U];
        }
    }
    return LM_DATASET_OK;
}
