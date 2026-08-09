#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_support.h"
#include "tokenizer/tokenizer.h"

static void remove_model_files(const char *path) {
    char temporary_path[1024] = {0};
    (void)remove(path);
    if (snprintf(temporary_path, sizeof(temporary_path), "%s.tmp", path) > 0) {
        (void)remove(temporary_path);
    }
}

static int files_match(const char *left_path, const char *right_path) {
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    if (left == NULL || right == NULL) {
        if (left != NULL) {
            (void)fclose(left);
        }
        if (right != NULL) {
            (void)fclose(right);
        }
        return 0;
    }

    int matches = 1;
    for (;;) {
        const int left_byte = fgetc(left);
        const int right_byte = fgetc(right);
        if (left_byte != right_byte) {
            matches = 0;
            break;
        }
        if (left_byte == EOF) {
            break;
        }
    }
    if (ferror(left) != 0 || ferror(right) != 0 || fclose(left) != 0 || fclose(right) != 0) {
        matches = 0;
    }
    return matches;
}

int main(void) {
    static const char *const ordered_inputs[] = {
        LLM_LAB_TEST_FIXTURE_DIR "/tiny-corpus-a.txt",
        LLM_LAB_TEST_FIXTURE_DIR "/tiny-corpus-b.txt",
    };
    static const char *const reversed_inputs[] = {
        LLM_LAB_TEST_FIXTURE_DIR "/tiny-corpus-b.txt",
        LLM_LAB_TEST_FIXTURE_DIR "/tiny-corpus-a.txt",
    };

    tokenizer *ordered = NULL;
    tokenizer *reversed = NULL;
    TEST_ASSERT(tokenizer_train(ordered_inputs, 2U, 264U, &ordered) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_train(reversed_inputs, 2U, 264U, &reversed) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_vocabulary_size(ordered) == tokenizer_vocabulary_size(reversed));
    TEST_ASSERT(tokenizer_vocabulary_size(ordered) > TOKENIZER_BYTE_VOCABULARY_SIZE);
    TEST_ASSERT(tokenizer_vocabulary_size(ordered) <= 264U);

    char ordered_path[1024] = {0};
    char reversed_path[1024] = {0};
    TEST_ASSERT(snprintf(ordered_path, sizeof(ordered_path), "%s/ordered.llmtok",
                         LLM_LAB_TEST_BINARY_DIR) > 0);
    TEST_ASSERT(snprintf(reversed_path, sizeof(reversed_path), "%s/reversed.llmtok",
                         LLM_LAB_TEST_BINARY_DIR) > 0);
    remove_model_files(ordered_path);
    remove_model_files(reversed_path);
    TEST_ASSERT(tokenizer_save(ordered, ordered_path) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_save(reversed, reversed_path) == TOKENIZER_OK);
    TEST_ASSERT(files_match(ordered_path, reversed_path) != 0);

    static const unsigned char sample[] = "abab abc";
    token_sequence tokens = {0};
    TEST_ASSERT(tokenizer_encode(ordered, sample, sizeof(sample) - 1U, &tokens) == TOKENIZER_OK);
    unsigned char *decoded = NULL;
    size_t decoded_length = 0U;
    TEST_ASSERT(tokenizer_decode(ordered, &tokens, &decoded, &decoded_length) == TOKENIZER_OK);
    TEST_ASSERT(decoded_length == sizeof(sample) - 1U);
    TEST_ASSERT(memcmp(decoded, sample, decoded_length) == 0);
    tokenizer_bytes_destroy(decoded);
    token_sequence_destroy(&tokens);
    tokenizer_destroy(ordered);
    tokenizer_destroy(reversed);
    remove_model_files(ordered_path);
    remove_model_files(reversed_path);

    char empty_path[1024] = {0};
    TEST_ASSERT(snprintf(empty_path, sizeof(empty_path), "%s/empty.txt", LLM_LAB_TEST_BINARY_DIR) >
                0);
    FILE *empty = fopen(empty_path, "wb");
    TEST_ASSERT(empty != NULL);
    TEST_ASSERT(fclose(empty) == 0);
    const char *empty_inputs[] = {empty_path};
    tokenizer *empty_model = NULL;
    TEST_ASSERT(tokenizer_train(empty_inputs, 1U, 260U, &empty_model) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_vocabulary_size(empty_model) == TOKENIZER_BYTE_VOCABULARY_SIZE);
    tokenizer_destroy(empty_model);
    TEST_ASSERT(remove(empty_path) == 0);

    tokenizer *invalid = NULL;
    TEST_ASSERT(tokenizer_train(NULL, 0U, 256U, &invalid) == TOKENIZER_INVALID_ARGUMENT);
    TEST_ASSERT(invalid == NULL);
    return EXIT_SUCCESS;
}
