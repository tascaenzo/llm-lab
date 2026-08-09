#ifndef LLM_LAB_DATASET_INTERNAL_H
#define LLM_LAB_DATASET_INTERNAL_H

#include <stdio.h>

#include "dataset/dataset.h"
#include "sha256.h"

#define LM_DATASET_FORMAT_VERSION UINT32_C(1)
#define LM_DATASET_HEADER_SIZE 128U
#define LM_DATASET_TOKEN_SIZE 4U
#define LM_DATASET_SPLIT_COUNT 3U

typedef struct lm_dataset_writer {
    FILE *file;
    char *final_path;
    char *temporary_path;
    uint64_t token_count;
    uint64_t document_count;
    lm_dataset_split split;
    tokenizer_sha256_context payload_sha256;
    int temporary_created;
    int published;
} lm_dataset_writer;

struct lm_dataset {
    FILE *file;
    uint64_t token_count;
    uint64_t document_count;
    uint32_t tokenizer_vocabulary_size;
    uint32_t model_vocabulary_size;
    token_id end_of_document_token;
    lm_dataset_split split;
};

struct lm_batcher {
    lm_dataset *dataset;
    size_t batch_size;
    size_t context_length;
    uint64_t random_state;
    token_id *window;
};

void lm_dataset_store_u32(unsigned char *bytes, uint32_t value);
void lm_dataset_store_u64(unsigned char *bytes, uint64_t value);
uint32_t lm_dataset_load_u32(const unsigned char *bytes);
uint64_t lm_dataset_load_u64(const unsigned char *bytes);

lm_dataset_status lm_dataset_writer_open(lm_dataset_writer *writer, const char *output_prefix,
                                         lm_dataset_split split);
lm_dataset_status lm_dataset_writer_append(lm_dataset_writer *writer, const token_id *tokens,
                                           size_t token_count, token_id end_of_document_token);
lm_dataset_status
lm_dataset_writers_publish(lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT],
                           uint32_t tokenizer_vocabulary_size,
                           const unsigned char tokenizer_checksum[TOKENIZER_SHA256_DIGEST_SIZE]);
void lm_dataset_writers_abort(lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT]);

lm_dataset_status lm_dataset_parse_documents(FILE *input, const tokenizer *tokenizer,
                                             lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT],
                                             uint64_t total_bytes,
                                             lm_dataset_progress_callback progress_callback,
                                             void *progress_context);

#endif
