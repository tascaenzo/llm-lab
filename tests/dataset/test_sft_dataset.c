#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "dataset/sft_dataset.h"
#include "test_support.h"
#include "tokenizer/tokenizer.h"

static uint64_t fnv1a(const char *text) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; text[index] != '\0'; ++index) {
        hash ^= (unsigned char)text[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int write_split_examples(const char *path) {
    FILE *file = fopen(path, "wb");
    TEST_ASSERT(file != NULL);
    int found[3] = {0};
    for (unsigned int number = 0U; number < 10000U; ++number) {
        char id[64] = {0};
        (void)snprintf(id, sizeof(id), "sft:%u", number);
        const uint64_t bucket = fnv1a(id) % UINT64_C(10000);
        const size_t split = bucket < UINT64_C(9000) ? 0U : (bucket < UINT64_C(9500) ? 1U : 2U);
        if (found[split] != 0) {
            continue;
        }
        const char *system = split == 0U ? "{\"role\":\"system\",\"content\":\"S\"}," : "";
        TEST_ASSERT(fprintf(file,
                            "{\"id\":\"%s\",\"source\":\"fixture\","
                            "\"license\":\"CC0-1.0\",\"messages\":[%s"
                            "{\"role\":\"user\",\"content\":\"Ciao\"},"
                            "{\"role\":\"assistant\",\"content\":\"Salve\"}]}\n",
                            id, system) > 0);
        found[split] = 1;
        if (found[0] != 0 && found[1] != 0 && found[2] != 0) {
            break;
        }
    }
    TEST_ASSERT(fclose(file) == 0);
    return found[0] != 0 && found[1] != 0 && found[2] != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int test_sft_round_trip(void) {
    char tokenizer_path[512] = {0};
    char jsonl_path[512] = {0};
    char prefix[512] = {0};
    (void)snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/sft-%ld.llmtok",
                   LLM_LAB_TEST_BINARY_DIR, (long)getpid());
    (void)snprintf(jsonl_path, sizeof(jsonl_path), "%s/sft-%ld.jsonl", LLM_LAB_TEST_BINARY_DIR,
                   (long)getpid());
    (void)snprintf(prefix, sizeof(prefix), "%s/sft-%ld", LLM_LAB_TEST_BINARY_DIR,
                   (long)getpid());

    tokenizer *text_tokenizer = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&text_tokenizer) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_save(text_tokenizer, tokenizer_path) == TOKENIZER_OK);
    tokenizer_destroy(text_tokenizer);
    TEST_ASSERT(write_split_examples(jsonl_path) == EXIT_SUCCESS);

    lm_sft_prepare_report report = {0};
    TEST_ASSERT(lm_sft_dataset_prepare_jsonl(tokenizer_path, jsonl_path, prefix, 32U, &report) ==
                LM_DATASET_OK);
    TEST_ASSERT(report.model_vocabulary_size == 264U);
    TEST_ASSERT(report.protocol.system_token == 257U);
    TEST_ASSERT(report.protocol.padding_token == 261U);
    for (size_t split = 0U; split < 3U; ++split) {
        TEST_ASSERT(report.example_counts[split] == 1U);
        TEST_ASSERT(report.supervised_token_counts[split] == 6U);
    }

    char train_path[540] = {0};
    (void)snprintf(train_path, sizeof(train_path), "%s.train.llmsft", prefix);
    lm_sft_dataset *dataset = NULL;
    TEST_ASSERT(lm_sft_dataset_open(train_path, &dataset) == LM_DATASET_OK);
    TEST_ASSERT(lm_sft_dataset_context_length(dataset) == 32U);
    int tokenizer_matches = 0;
    TEST_ASSERT(lm_sft_dataset_tokenizer_matches(dataset, tokenizer_path, &tokenizer_matches) ==
                LM_DATASET_OK);
    TEST_ASSERT(tokenizer_matches != 0);

    lm_sft_batcher *batcher = NULL;
    TEST_ASSERT(lm_sft_batcher_create(dataset, 1U, 7U, &batcher) == LM_DATASET_OK);
    token_id inputs[32] = {0};
    token_id targets[32] = {0};
    uint32_t mask[32] = {0};
    size_t active_count = 0U;
    TEST_ASSERT(lm_sft_batcher_next(batcher, inputs, targets, mask, &active_count) ==
                LM_DATASET_OK);
    TEST_ASSERT(active_count == 6U);
    TEST_ASSERT(inputs[0] == report.protocol.system_token);
    TEST_ASSERT(inputs[1] == (token_id)'S');
    TEST_ASSERT(targets[0] == (token_id)'S');
    TEST_ASSERT(mask[0] == 0U);
    size_t mask_sum = 0U;
    for (size_t index = 0U; index < 32U; ++index) {
        mask_sum += mask[index];
    }
    TEST_ASSERT(mask_sum == active_count);

    lm_sft_batcher_destroy(batcher);
    lm_sft_dataset_close(dataset);
    (void)remove(train_path);
    char path[540] = {0};
    (void)snprintf(path, sizeof(path), "%s.validation.llmsft", prefix);
    (void)remove(path);
    (void)snprintf(path, sizeof(path), "%s.test.llmsft", prefix);
    (void)remove(path);
    (void)remove(jsonl_path);
    (void)remove(tokenizer_path);
    return EXIT_SUCCESS;
}

int main(void) { return test_sft_round_trip(); }
