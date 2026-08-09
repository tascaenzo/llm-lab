#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include "test_support.h"
#include "tokenizer_internal.h"

#define TEST_MODEL_HEADER_SIZE 64U
#define TEST_MODEL_MERGE_SIZE 8U
#define TEST_MODEL_SIZE (TEST_MODEL_HEADER_SIZE + 2U * TEST_MODEL_MERGE_SIZE)

static int write_bytes(const char *path, const unsigned char *bytes, size_t length) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    const int wrote_all = fwrite(bytes, 1U, length, file) == length;
    return fclose(file) == 0 && wrote_all != 0;
}

static int read_bytes(const char *path, unsigned char *bytes, size_t length) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    const int read_all = fread(bytes, 1U, length, file) == length;
    const int reached_end = fgetc(file) == EOF && ferror(file) == 0;
    return fclose(file) == 0 && read_all != 0 && reached_end != 0;
}

static int file_contains(const char *path, const char *expected) {
    unsigned char buffer[64] = {0};
    const size_t expected_length = strlen(expected);
    if (expected_length > sizeof(buffer)) {
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    const size_t length = fread(buffer, 1U, sizeof(buffer), file);
    const int read_ok = ferror(file) == 0 && fclose(file) == 0;
    return read_ok != 0 && length == expected_length && memcmp(buffer, expected, length) == 0;
}

static void remove_model_files(const char *path) {
    char temporary_path[1024] = {0};
    (void)remove(path);
    const int length = snprintf(temporary_path, sizeof(temporary_path), "%s.tmp", path);
    if (length > 0 && (size_t)length < sizeof(temporary_path)) {
        (void)remove(temporary_path);
    }
}

static void store_little_endian_u32(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8U);
    bytes[2] = (unsigned char)(value >> 16U);
    bytes[3] = (unsigned char)(value >> 24U);
}

static void refresh_payload_checksum(unsigned char *model, size_t length) {
    tokenizer_sha256_context sha256;
    tokenizer_sha256_init(&sha256);
    tokenizer_sha256_update(&sha256, model + TEST_MODEL_HEADER_SIZE,
                            length - TEST_MODEL_HEADER_SIZE);
    tokenizer_sha256_final(&sha256, model + 32U);
}

static int expect_invalid_model(const char *name, const unsigned char *bytes, size_t length) {
    char path[1024] = {0};
    const int path_length =
        snprintf(path, sizeof(path), "%s/%s.llmtok", LLM_LAB_TEST_BINARY_DIR, name);
    if (path_length <= 0 || (size_t)path_length >= sizeof(path) ||
        write_bytes(path, bytes, length) == 0) {
        return 0;
    }
    tokenizer *model = NULL;
    const tokenizer_status status = tokenizer_load(path, &model);
    tokenizer_destroy(model);
    const int removed = remove(path) == 0;
    return status == TOKENIZER_INVALID_MODEL && removed != 0;
}

int main(void) {
    char valid_path[1024] = {0};
    const int valid_path_length =
        snprintf(valid_path, sizeof(valid_path), "%s/valid.llmtok", LLM_LAB_TEST_BINARY_DIR);
    TEST_ASSERT(valid_path_length > 0 && (size_t)valid_path_length < sizeof(valid_path));
    remove_model_files(valid_path);

    tokenizer *known_model = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&known_model) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_add_merge(known_model, 97U, 98U) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_add_merge(known_model, 256U, 99U) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_save(known_model, valid_path) == TOKENIZER_OK);
    tokenizer_destroy(known_model);

    unsigned char valid_model[TEST_MODEL_SIZE] = {0};
    TEST_ASSERT(read_bytes(valid_path, valid_model, sizeof(valid_model)) != 0);
    static const unsigned char expected_magic[8] = {'L', 'L', 'M', 'T', 'O', 'K', '\r', '\n'};
    TEST_ASSERT(memcmp(valid_model, expected_magic, sizeof(expected_magic)) == 0);

    tokenizer *loaded = NULL;
    TEST_ASSERT(tokenizer_load(valid_path, &loaded) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_vocabulary_size(loaded) == 258U);
    TEST_ASSERT(tokenizer_merge_count(loaded) == 2U);
    tokenizer_destroy(loaded);
    TEST_ASSERT(remove(valid_path) == 0);

    unsigned char invalid[TEST_MODEL_SIZE + 1U] = {0};
    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    invalid[0] ^= 0xFFU;
    TEST_ASSERT(expect_invalid_model("bad-magic", invalid, TEST_MODEL_SIZE) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    store_little_endian_u32(invalid + 8U, 2U);
    TEST_ASSERT(expect_invalid_model("bad-version", invalid, TEST_MODEL_SIZE) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    store_little_endian_u32(invalid + 12U, 63U);
    TEST_ASSERT(expect_invalid_model("bad-header-size", invalid, TEST_MODEL_SIZE) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    store_little_endian_u32(invalid + 16U, 255U);
    TEST_ASSERT(expect_invalid_model("bad-base-vocabulary", invalid, TEST_MODEL_SIZE) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    invalid[24U] = 15U;
    TEST_ASSERT(expect_invalid_model("bad-payload-size", invalid, TEST_MODEL_SIZE) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    invalid[32U] ^= 0xFFU;
    TEST_ASSERT(expect_invalid_model("bad-checksum", invalid, TEST_MODEL_SIZE) != 0);

    TEST_ASSERT(expect_invalid_model("truncated", valid_model, TEST_MODEL_SIZE - 1U) != 0);
    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    invalid[TEST_MODEL_SIZE] = 0U;
    TEST_ASSERT(expect_invalid_model("trailing-data", invalid, TEST_MODEL_SIZE + 1U) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    store_little_endian_u32(invalid + TEST_MODEL_HEADER_SIZE, 999U);
    refresh_payload_checksum(invalid, TEST_MODEL_SIZE);
    TEST_ASSERT(expect_invalid_model("future-token-reference", invalid, TEST_MODEL_SIZE) != 0);

    memcpy(invalid, valid_model, TEST_MODEL_SIZE);
    memcpy(invalid + TEST_MODEL_HEADER_SIZE + TEST_MODEL_MERGE_SIZE,
           invalid + TEST_MODEL_HEADER_SIZE, TEST_MODEL_MERGE_SIZE);
    refresh_payload_checksum(invalid, TEST_MODEL_SIZE);
    TEST_ASSERT(expect_invalid_model("duplicate-merge", invalid, TEST_MODEL_SIZE) != 0);

    static const unsigned char text_model[] = "llm-tokenizer 1\nbase-vocab 256\nmerges 0\n";
    TEST_ASSERT(expect_invalid_model("legacy-text", text_model, sizeof(text_model) - 1U) != 0);

    tokenizer *byte_model = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&byte_model) == TOKENIZER_OK);
    char protected_path[1024] = {0};
    char protected_temporary_path[1024] = {0};
    const int protected_length = snprintf(protected_path, sizeof(protected_path),
                                          "%s/protected.llmtok", LLM_LAB_TEST_BINARY_DIR);
    TEST_ASSERT(protected_length > 0 && (size_t)protected_length < sizeof(protected_path));
    const int temporary_length = snprintf(
        protected_temporary_path, sizeof(protected_temporary_path), "%s.tmp", protected_path);
    TEST_ASSERT(temporary_length > 0 &&
                (size_t)temporary_length < sizeof(protected_temporary_path));
    remove_model_files(protected_path);
    TEST_ASSERT(write_bytes(protected_path, (const unsigned char *)"original", 8U) != 0);
    TEST_ASSERT(write_bytes(protected_temporary_path, (const unsigned char *)"busy", 4U) != 0);
    TEST_ASSERT(tokenizer_save(byte_model, protected_path) == TOKENIZER_IO_ERROR);
    TEST_ASSERT(file_contains(protected_path, "original") != 0);
    remove_model_files(protected_path);
    tokenizer_destroy(byte_model);
    return EXIT_SUCCESS;
}
