#ifndef LLM_LAB_DATASET_SFT_DATASET_H
#define LLM_LAB_DATASET_SFT_DATASET_H

#include <stddef.h>
#include <stdint.h>

#include "dataset/dataset.h"

#define LM_CHAT_RESERVED_TOKEN_COUNT UINT32_C(7)

typedef struct lm_chat_protocol {
    token_id system_token;
    token_id user_token;
    token_id assistant_token;
    token_id end_token;
    token_id padding_token;
} lm_chat_protocol;

typedef struct lm_sft_prepare_report {
    uint64_t example_counts[3];
    uint64_t supervised_token_counts[3];
    uint32_t tokenizer_vocabulary_size;
    uint32_t model_vocabulary_size;
    size_t context_length;
    lm_chat_protocol protocol;
} lm_sft_prepare_report;

typedef struct lm_sft_dataset lm_sft_dataset;
typedef struct lm_sft_batcher lm_sft_batcher;

/** Resolves the stable v1 role-token mapping above <EOD>. */
lm_dataset_status lm_chat_protocol_v1(uint32_t tokenizer_vocabulary_size,
                                      uint32_t model_vocabulary_size,
                                      lm_chat_protocol *out_protocol);

/**
 * Converts conversational JSONL to fixed-size SFT artifacts. Each line must
 * contain non-empty `id`, `source`, `license` fields and a `messages` array of
 * `{role,content}` objects.
 */
lm_dataset_status lm_sft_dataset_prepare_jsonl(const char *tokenizer_path,
                                               const char *conversations_jsonl_path,
                                               const char *output_prefix, size_t context_length,
                                               lm_sft_prepare_report *out_report);

lm_dataset_status lm_sft_dataset_open(const char *path, lm_sft_dataset **out_dataset);
void lm_sft_dataset_close(lm_sft_dataset *dataset);

lm_dataset_split lm_sft_dataset_get_split(const lm_sft_dataset *dataset);
uint64_t lm_sft_dataset_example_count(const lm_sft_dataset *dataset);
uint64_t lm_sft_dataset_supervised_token_count(const lm_sft_dataset *dataset);
uint32_t lm_sft_dataset_model_vocabulary_size(const lm_sft_dataset *dataset);
size_t lm_sft_dataset_context_length(const lm_sft_dataset *dataset);
lm_chat_protocol lm_sft_dataset_protocol(const lm_sft_dataset *dataset);
lm_dataset_status lm_sft_dataset_tokenizer_matches(const lm_sft_dataset *dataset,
                                                   const char *tokenizer_path, int *out_matches);

lm_dataset_status lm_sft_batcher_create(lm_sft_dataset *dataset, size_t batch_size, uint64_t seed,
                                        lm_sft_batcher **out_batcher);
void lm_sft_batcher_destroy(lm_sft_batcher *batcher);

/** Fills B*T inputs, targets and a zero/one loss mask. */
lm_dataset_status lm_sft_batcher_next(lm_sft_batcher *batcher, token_id *out_inputs,
                                      token_id *out_targets, uint32_t *out_loss_mask,
                                      size_t *out_active_target_count);
lm_dataset_status lm_sft_batcher_get_state(const lm_sft_batcher *batcher,
                                           lm_batcher_state *out_state);
lm_dataset_status lm_sft_batcher_set_state(lm_sft_batcher *batcher, const lm_batcher_state *state);

#endif
