#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset/dataset.h"
#include "test_support.h"

typedef struct progress_observation {
    size_t calls;
    uint64_t bytes_read;
    uint64_t total_bytes;
    uint64_t documents;
    uint64_t tokens;
} progress_observation;

static void observe_progress(uint64_t bytes_read, uint64_t total_bytes,
                             uint64_t documents_processed, const uint64_t token_counts[3],
                             void *context) {
    progress_observation *observation = context;
    ++observation->calls;
    observation->bytes_read = bytes_read;
    observation->total_bytes = total_bytes;
    observation->documents = documents_processed;
    observation->tokens = token_counts[0] + token_counts[1] + token_counts[2];
}

static void build_path(char *buffer, size_t capacity, const char *name) {
    (void)snprintf(buffer, capacity, "%s/%s", LLM_LAB_TEST_BINARY_DIR, name);
}

static void cleanup_outputs(const char *prefix) {
    static const char *const suffixes[] = {
        ".train.llmdat",      ".validation.llmdat",      ".test.llmdat",
        ".train.llmdat.part", ".validation.llmdat.part", ".test.llmdat.part",
    };
    char path[1024] = {0};
    for (size_t index = 0U; index < sizeof(suffixes) / sizeof(suffixes[0]); ++index) {
        (void)snprintf(path, sizeof(path), "%s%s", prefix, suffixes[index]);
        (void)remove(path);
    }
}

static int verify_split(const char *path, lm_dataset_split expected_split,
                        uint64_t expected_documents) {
    lm_dataset *dataset = NULL;
    TEST_ASSERT(lm_dataset_open(path, &dataset) == LM_DATASET_OK);
    TEST_ASSERT(lm_dataset_get_split(dataset) == expected_split);
    TEST_ASSERT(lm_dataset_document_count(dataset) == expected_documents);
    TEST_ASSERT(lm_dataset_token_count(dataset) >= expected_documents);
    TEST_ASSERT(lm_dataset_tokenizer_vocabulary_size(dataset) == 256U);
    TEST_ASSERT(lm_dataset_model_vocabulary_size(dataset) == 257U);
    TEST_ASSERT(lm_dataset_end_of_document_token(dataset) == 256U);

    const uint64_t token_count = lm_dataset_token_count(dataset);
    TEST_ASSERT(token_count <= SIZE_MAX / sizeof(token_id));
    token_id *tokens = malloc((size_t)token_count * sizeof(*tokens));
    TEST_ASSERT(tokens != NULL);
    TEST_ASSERT(lm_dataset_read_tokens(dataset, 0U, (size_t)token_count, tokens) == LM_DATASET_OK);
    uint64_t end_tokens = 0U;
    for (size_t index = 0U; index < (size_t)token_count; ++index) {
        TEST_ASSERT(tokens[index] <= 256U);
        if (tokens[index] == 256U) {
            ++end_tokens;
        }
    }
    TEST_ASSERT(end_tokens == expected_documents);
    free(tokens);
    lm_dataset_close(dataset);
    return EXIT_SUCCESS;
}

static int verify_batcher(const char *train_path) {
    lm_dataset *dataset = NULL;
    TEST_ASSERT(lm_dataset_open(train_path, &dataset) == LM_DATASET_OK);
    lm_batcher *first = NULL;
    lm_batcher *second = NULL;
    TEST_ASSERT(lm_batcher_create(dataset, 3U, 4U, UINT64_C(42), &first) == LM_DATASET_OK);
    TEST_ASSERT(lm_batcher_create(dataset, 3U, 4U, UINT64_C(42), &second) == LM_DATASET_OK);
    TEST_ASSERT(lm_batcher_batch_size(first) == 3U);
    TEST_ASSERT(lm_batcher_context_length(first) == 4U);

    token_id first_inputs[12] = {0};
    token_id first_targets[12] = {0};
    token_id second_inputs[12] = {0};
    token_id second_targets[12] = {0};
    TEST_ASSERT(lm_batcher_next(first, first_inputs, first_targets) == LM_DATASET_OK);
    TEST_ASSERT(lm_batcher_next(second, second_inputs, second_targets) == LM_DATASET_OK);
    TEST_ASSERT(memcmp(first_inputs, second_inputs, sizeof(first_inputs)) == 0);
    TEST_ASSERT(memcmp(first_targets, second_targets, sizeof(first_targets)) == 0);
    for (size_t row = 0U; row < 3U; ++row) {
        for (size_t column = 0U; column + 1U < 4U; ++column) {
            TEST_ASSERT(first_targets[row * 4U + column] == first_inputs[row * 4U + column + 1U]);
        }
    }

    lm_batcher_destroy(first);
    lm_batcher_destroy(second);
    lm_dataset_close(dataset);
    return EXIT_SUCCESS;
}

static int verify_corruption_is_detected(const char *source_path, const char *corrupt_path) {
    FILE *source = fopen(source_path, "rb");
    TEST_ASSERT(source != NULL);
    FILE *destination = fopen(corrupt_path, "wb");
    TEST_ASSERT(destination != NULL);
    unsigned char buffer[4096] = {0};
    for (;;) {
        const size_t count = fread(buffer, 1U, sizeof(buffer), source);
        if (count != 0U) {
            TEST_ASSERT(fwrite(buffer, 1U, count, destination) == count);
        }
        if (count < sizeof(buffer)) {
            break;
        }
    }
    TEST_ASSERT(ferror(source) == 0);
    TEST_ASSERT(fclose(source) == 0);
    TEST_ASSERT(fclose(destination) == 0);

    destination = fopen(corrupt_path, "rb+");
    TEST_ASSERT(destination != NULL);
    TEST_ASSERT(fseek(destination, 128L, SEEK_SET) == 0);
    const int original = fgetc(destination);
    TEST_ASSERT(original != EOF);
    TEST_ASSERT(fseek(destination, 128L, SEEK_SET) == 0);
    TEST_ASSERT(fputc(original ^ 0xff, destination) != EOF);
    TEST_ASSERT(fclose(destination) == 0);

    lm_dataset *dataset = NULL;
    TEST_ASSERT(lm_dataset_open(corrupt_path, &dataset) == LM_DATASET_INVALID_FORMAT);
    TEST_ASSERT(dataset == NULL);
    return EXIT_SUCCESS;
}

int main(void) {
    char model_path[1024] = {0};
    char output_prefix[1024] = {0};
    char train_path[1024] = {0};
    char validation_path[1024] = {0};
    char test_path[1024] = {0};
    char corrupt_path[1024] = {0};
    build_path(model_path, sizeof(model_path), "dataset-test.llmtok");
    build_path(output_prefix, sizeof(output_prefix), "dataset-test");
    (void)snprintf(train_path, sizeof(train_path), "%s.train.llmdat", output_prefix);
    (void)snprintf(validation_path, sizeof(validation_path), "%s.validation.llmdat", output_prefix);
    (void)snprintf(test_path, sizeof(test_path), "%s.test.llmdat", output_prefix);
    build_path(corrupt_path, sizeof(corrupt_path), "dataset-test-corrupt.llmdat");
    (void)remove(model_path);
    (void)remove(corrupt_path);
    cleanup_outputs(output_prefix);

    tokenizer *tokenizer = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&tokenizer) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_save(tokenizer, model_path) == TOKENIZER_OK);
    tokenizer_destroy(tokenizer);

    lm_dataset_prepare_report report = {0};
    progress_observation progress = {0};
    TEST_ASSERT(lm_dataset_prepare_jsonl_with_progress(model_path, LLM_LAB_TEST_DOCUMENTS_PATH,
                                                       output_prefix, observe_progress, &progress,
                                                       &report) == LM_DATASET_OK);
    TEST_ASSERT(progress.calls >= 2U);
    TEST_ASSERT(progress.bytes_read == progress.total_bytes);
    TEST_ASSERT(progress.total_bytes > 0U);
    TEST_ASSERT(progress.documents == 6U);
    TEST_ASSERT(progress.tokens ==
                report.token_counts[0] + report.token_counts[1] + report.token_counts[2]);
    TEST_ASSERT(report.tokenizer_vocabulary_size == 256U);
    TEST_ASSERT(report.model_vocabulary_size == 257U);
    TEST_ASSERT(report.end_of_document_token == 256U);
    for (size_t index = 0U; index < 3U; ++index) {
        TEST_ASSERT(report.document_counts[index] == 2U);
    }

    TEST_ASSERT(verify_split(train_path, LM_DATASET_TRAIN, 2U) == EXIT_SUCCESS);
    TEST_ASSERT(verify_split(validation_path, LM_DATASET_VALIDATION, 2U) == EXIT_SUCCESS);
    TEST_ASSERT(verify_split(test_path, LM_DATASET_TEST, 2U) == EXIT_SUCCESS);
    TEST_ASSERT(verify_batcher(train_path) == EXIT_SUCCESS);
    TEST_ASSERT(verify_corruption_is_detected(train_path, corrupt_path) == EXIT_SUCCESS);

    TEST_ASSERT(lm_dataset_prepare_jsonl(model_path, LLM_LAB_TEST_DOCUMENTS_PATH, output_prefix,
                                         &report) == LM_DATASET_OUTPUT_EXISTS);

    (void)remove(model_path);
    (void)remove(corrupt_path);
    cleanup_outputs(output_prefix);
    return EXIT_SUCCESS;
}
