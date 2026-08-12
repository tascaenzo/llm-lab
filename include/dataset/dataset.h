#ifndef LLM_LAB_DATASET_DATASET_H
#define LLM_LAB_DATASET_DATASET_H

#include <stddef.h>
#include <stdint.h>

#include "tokenizer/tokenizer.h"

typedef enum lm_dataset_status {
    LM_DATASET_OK = 0,
    LM_DATASET_INVALID_ARGUMENT,
    LM_DATASET_ALLOCATION_FAILED,
    LM_DATASET_OVERFLOW,
    LM_DATASET_IO_ERROR,
    LM_DATASET_OUTPUT_EXISTS,
    LM_DATASET_INVALID_JSONL,
    LM_DATASET_DUPLICATE_DOCUMENT,
    LM_DATASET_INVALID_FORMAT,
    LM_DATASET_INSUFFICIENT_DATA,
    LM_DATASET_TOKENIZER_ERROR
} lm_dataset_status;

typedef enum lm_dataset_split {
    LM_DATASET_TRAIN = 0,
    LM_DATASET_VALIDATION = 1,
    LM_DATASET_TEST = 2
} lm_dataset_split;

typedef struct lm_dataset_prepare_report {
    uint64_t document_counts[3];
    uint64_t token_counts[3];
    uint32_t tokenizer_vocabulary_size;
    uint32_t model_vocabulary_size;
    token_id end_of_document_token;
} lm_dataset_prepare_report;

typedef struct lm_dataset lm_dataset;
typedef struct lm_batcher lm_batcher;

typedef void (*lm_dataset_progress_callback)(uint64_t bytes_read, uint64_t total_bytes,
                                             uint64_t documents_processed,
                                             const uint64_t token_counts[3], void *context);

/** Reports payload bytes checked while an existing .llmdat artifact is opened. */
typedef void (*lm_dataset_open_progress_callback)(uint64_t bytes_read, uint64_t total_bytes,
                                                  void *context);

/**
 * Converts a documents.jsonl file into PREFIX.train.llmdat,
 * PREFIX.validation.llmdat and PREFIX.test.llmdat.
 */
lm_dataset_status lm_dataset_prepare_jsonl(const char *tokenizer_path,
                                           const char *documents_jsonl_path,
                                           const char *output_prefix,
                                           lm_dataset_prepare_report *out_report);

/** Like lm_dataset_prepare_jsonl, with optional progress notifications. */
lm_dataset_status lm_dataset_prepare_jsonl_with_progress(
    const char *tokenizer_path, const char *documents_jsonl_path, const char *output_prefix,
    lm_dataset_progress_callback progress_callback, void *progress_context,
    lm_dataset_prepare_report *out_report);

/** Opens and fully validates an .llmdat artifact. */
lm_dataset_status lm_dataset_open(const char *path, lm_dataset **out_dataset);

/** Like lm_dataset_open, with optional progress notifications during full payload validation. */
lm_dataset_status lm_dataset_open_with_progress(const char *path,
                                                lm_dataset_open_progress_callback progress_callback,
                                                void *progress_context, lm_dataset **out_dataset);

void lm_dataset_close(lm_dataset *dataset);

uint64_t lm_dataset_token_count(const lm_dataset *dataset);
uint64_t lm_dataset_document_count(const lm_dataset *dataset);
uint32_t lm_dataset_tokenizer_vocabulary_size(const lm_dataset *dataset);
uint32_t lm_dataset_model_vocabulary_size(const lm_dataset *dataset);
token_id lm_dataset_end_of_document_token(const lm_dataset *dataset);
lm_dataset_split lm_dataset_get_split(const lm_dataset *dataset);

/** Reads count tokens starting at token_offset. */
lm_dataset_status lm_dataset_read_tokens(lm_dataset *dataset, uint64_t token_offset, size_t count,
                                         token_id *out_tokens);

/** Creates a deterministic random-window batcher. The dataset must outlive it. */
lm_dataset_status lm_batcher_create(lm_dataset *dataset, size_t batch_size, size_t context_length,
                                    uint64_t seed, lm_batcher **out_batcher);

void lm_batcher_destroy(lm_batcher *batcher);

size_t lm_batcher_batch_size(const lm_batcher *batcher);
size_t lm_batcher_context_length(const lm_batcher *batcher);

/** Returns/restores the PRNG state used for deterministic batch selection. */
uint64_t lm_batcher_random_state(const lm_batcher *batcher);
lm_dataset_status lm_batcher_set_random_state(lm_batcher *batcher, uint64_t state);

/** Fills batch_size * context_length input and target tokens. */
lm_dataset_status lm_batcher_next(lm_batcher *batcher, token_id *out_inputs, token_id *out_targets);

const char *lm_dataset_status_string(lm_dataset_status status);

#endif
