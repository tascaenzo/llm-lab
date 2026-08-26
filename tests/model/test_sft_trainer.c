#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include "dataset/sft_dataset.h"
#include "model/model.h"
#include "runtime/backend.h"
#include "test_support.h"
#include "tokenizer/tokenizer.h"

static uint64_t hash_id(const char *text) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; text[index] != '\0'; ++index) {
        hash ^= (unsigned char)text[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int write_examples(const char *path) {
    FILE *file = fopen(path, "wb");
    TEST_ASSERT(file != NULL);
    int counts[3] = {0};
    for (unsigned int number = 0U; number < 50000U; ++number) {
        char id[32] = {0};
        (void)snprintf(id, sizeof(id), "trainer:%u", number);
        const uint64_t bucket = hash_id(id) % UINT64_C(10000);
        const size_t split = bucket < UINT64_C(9000) ? 0U : (bucket < UINT64_C(9500) ? 1U : 2U);
        const int required = split == 0U ? 3 : 1;
        if (counts[split] >= required) {
            continue;
        }
        TEST_ASSERT(fprintf(file,
                            "{\"id\":\"%s\",\"source\":\"fixture\","
                            "\"license\":\"CC0-1.0\",\"messages\":["
                            "{\"role\":\"user\",\"content\":\"a\"},"
                            "{\"role\":\"assistant\",\"content\":\"b\"}]}\n",
                            id) > 0);
        ++counts[split];
        if (counts[0] == 3 && counts[1] == 1 && counts[2] == 1) {
            break;
        }
    }
    TEST_ASSERT(fclose(file) == 0);
    return counts[0] == 3 && counts[1] == 1 && counts[2] == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static void remove_artifacts(const char *prefix, const char *tokenizer_path,
                             const char *jsonl_path, const char *checkpoint_path) {
    char path[600] = {0};
    (void)snprintf(path, sizeof(path), "%s.train.llmsft", prefix);
    (void)remove(path);
    (void)snprintf(path, sizeof(path), "%s.validation.llmsft", prefix);
    (void)remove(path);
    (void)snprintf(path, sizeof(path), "%s.test.llmsft", prefix);
    (void)remove(path);
    (void)remove(tokenizer_path);
    (void)remove(jsonl_path);
    (void)remove(checkpoint_path);
}

static int test_sft_train_save_resume(void) {
    char tokenizer_path[512] = {0}, jsonl_path[512] = {0}, prefix[512] = {0};
    char checkpoint_path[512] = {0}, train_path[560] = {0}, validation_path[560] = {0};
    const long process = (long)getpid();
    (void)snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/sft-model-%ld.llmtok",
                   LLM_LAB_TEST_BINARY_DIR, process);
    (void)snprintf(jsonl_path, sizeof(jsonl_path), "%s/sft-model-%ld.jsonl",
                   LLM_LAB_TEST_BINARY_DIR, process);
    (void)snprintf(prefix, sizeof(prefix), "%s/sft-model-%ld", LLM_LAB_TEST_BINARY_DIR, process);
    (void)snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/sft-model-%ld.llmckpt",
                   LLM_LAB_TEST_BINARY_DIR, process);
    tokenizer *text_tokenizer = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&text_tokenizer) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_save(text_tokenizer, tokenizer_path) == TOKENIZER_OK);
    tokenizer_destroy(text_tokenizer);
    TEST_ASSERT(write_examples(jsonl_path) == EXIT_SUCCESS);
    lm_sft_prepare_report report = {0};
    TEST_ASSERT(lm_sft_dataset_prepare_jsonl(tokenizer_path, jsonl_path, prefix, 16U, &report) ==
                LM_DATASET_OK);
    (void)snprintf(train_path, sizeof(train_path), "%s.train.llmsft", prefix);
    (void)snprintf(validation_path, sizeof(validation_path), "%s.validation.llmsft", prefix);
    lm_sft_dataset *train = NULL, *validation = NULL;
    TEST_ASSERT(lm_sft_dataset_open(train_path, &train) == LM_DATASET_OK);
    TEST_ASSERT(lm_sft_dataset_open(validation_path, &validation) == LM_DATASET_OK);

    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create(&backend) == LLM_OK);
    const lm_model_config model_config = {.vocabulary_size = report.model_vocabulary_size,
                                          .context_length = 16U,
                                          .hidden_size = 16U,
                                          .layer_count = 1U,
                                          .head_count = 1U,
                                          .feed_forward_size = 32U,
                                          .seed = 3U};
    lm_model *model = NULL;
    TEST_ASSERT(lm_model_create(backend, &model_config, &model) == LLM_OK);
    const lm_trainer_config trainer_config = {.batch_size = 1U,
                                              .context_length = 16U,
                                              .seed = 5U,
                                              .learning_rate = 1.0e-3F,
                                              .beta1 = 0.9F,
                                              .beta2 = 0.95F,
                                              .epsilon = 1.0e-8F,
                                              .weight_decay = 0.0F,
                                              .gradient_accumulation_steps = 2U,
                                              .warmup_steps = 0U,
                                              .total_steps = 0U,
                                              .minimum_learning_rate = 1.0e-3F,
                                              .sampling = LM_BATCHER_SHUFFLED_BLOCKS,
                                              .gradient_clip_norm = 1.0F};
    lm_trainer *trainer = NULL;
    TEST_ASSERT(lm_sft_trainer_create(model, train, &trainer_config, &trainer) == LLM_OK);
    float loss = 0.0F;
    TEST_ASSERT(lm_trainer_step(trainer, &loss) == LLM_OK);
    TEST_ASSERT(isfinite(loss) != 0 && loss > 0.0F);
    float validation_loss = 0.0F;
    TEST_ASSERT(lm_model_evaluate_sft_validation(model, validation, 1U, 1U, 11U,
                                                 &validation_loss) == LLM_OK);
    TEST_ASSERT(isfinite(validation_loss) != 0 && validation_loss > 0.0F);
    TEST_ASSERT(lm_sft_trainer_save_checkpoint(trainer, train, checkpoint_path) == LLM_OK);
    lm_trainer_destroy(trainer);
    lm_model_destroy(model);

    lm_model *resumed_model = NULL;
    lm_trainer *resumed_trainer = NULL;
    TEST_ASSERT(lm_sft_trainer_load_checkpoint(backend, train, checkpoint_path, &resumed_model,
                                               &resumed_trainer) == LLM_OK);
    TEST_ASSERT(lm_trainer_step_count(resumed_trainer) == 1U);
    TEST_ASSERT(lm_trainer_step(resumed_trainer, &loss) == LLM_OK);
    TEST_ASSERT(lm_trainer_step_count(resumed_trainer) == 2U);

    lm_trainer_destroy(resumed_trainer);
    lm_model_destroy(resumed_model);
    llm_backend_destroy(backend);
    lm_sft_dataset_close(validation);
    lm_sft_dataset_close(train);
    remove_artifacts(prefix, tokenizer_path, jsonl_path, checkpoint_path);
    return EXIT_SUCCESS;
}

int main(void) { return test_sft_train_save_resume(); }
