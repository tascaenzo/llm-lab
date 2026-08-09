#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include "tokenizer_internal.h"

#define MODEL_FORMAT_VERSION UINT32_C(1)
#define MODEL_HEADER_SIZE 64U
#define MODEL_MERGE_SIZE 8U
#define MODEL_CHECKSUM_OFFSET 32U

static const unsigned char model_magic[8] = {'L', 'L', 'M', 'T', 'O', 'K', '\r', '\n'};

static void store_little_endian_u32(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8U);
    bytes[2] = (unsigned char)(value >> 16U);
    bytes[3] = (unsigned char)(value >> 24U);
}

static void store_little_endian_u64(unsigned char *bytes, uint64_t value) {
    for (size_t index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
}

static uint32_t load_little_endian_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) | ((uint32_t)bytes[2] << 16U) |
           ((uint32_t)bytes[3] << 24U);
}

static uint64_t load_little_endian_u64(const unsigned char *bytes) {
    uint64_t value = 0U;
    for (size_t index = 0U; index < 8U; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static void encode_merge(unsigned char bytes[MODEL_MERGE_SIZE], bpe_merge merge) {
    store_little_endian_u32(bytes, merge.left_id);
    store_little_endian_u32(bytes + 4U, merge.right_id);
}

static tokenizer_status read_exact(FILE *file, unsigned char *bytes, size_t length) {
    const size_t read_count = fread(bytes, 1U, length, file);
    if (read_count == length) {
        return TOKENIZER_OK;
    }
    return ferror(file) != 0 ? TOKENIZER_IO_ERROR : TOKENIZER_INVALID_MODEL;
}

static tokenizer_status write_exact(FILE *file, const unsigned char *bytes, size_t length) {
    return fwrite(bytes, 1U, length, file) == length ? TOKENIZER_OK : TOKENIZER_IO_ERROR;
}

static void payload_checksum(const tokenizer *tokenizer,
                             unsigned char checksum[TOKENIZER_SHA256_DIGEST_SIZE]) {
    tokenizer_sha256_context sha256;
    tokenizer_sha256_init(&sha256);
    for (size_t index = 0U; index < tokenizer->merge_count; ++index) {
        unsigned char encoded[MODEL_MERGE_SIZE] = {0};
        encode_merge(encoded, tokenizer->merges[index]);
        tokenizer_sha256_update(&sha256, encoded, sizeof(encoded));
    }
    tokenizer_sha256_final(&sha256, checksum);
}

static tokenizer_status write_model(const tokenizer *tokenizer, FILE *file) {
    if (tokenizer->merge_count > (size_t)UINT32_MAX - TOKENIZER_BYTE_VOCABULARY_SIZE) {
        return TOKENIZER_OVERFLOW;
    }

    const uint32_t merge_count = (uint32_t)tokenizer->merge_count;
    const uint64_t payload_size = (uint64_t)merge_count * MODEL_MERGE_SIZE;
    unsigned char header[MODEL_HEADER_SIZE] = {0};
    memcpy(header, model_magic, sizeof(model_magic));
    store_little_endian_u32(header + 8U, MODEL_FORMAT_VERSION);
    store_little_endian_u32(header + 12U, MODEL_HEADER_SIZE);
    store_little_endian_u32(header + 16U, TOKENIZER_BYTE_VOCABULARY_SIZE);
    store_little_endian_u32(header + 20U, merge_count);
    store_little_endian_u64(header + 24U, payload_size);
    payload_checksum(tokenizer, header + MODEL_CHECKSUM_OFFSET);

    tokenizer_status status = write_exact(file, header, sizeof(header));
    for (size_t index = 0U; index < tokenizer->merge_count && status == TOKENIZER_OK; ++index) {
        unsigned char encoded[MODEL_MERGE_SIZE] = {0};
        encode_merge(encoded, tokenizer->merges[index]);
        status = write_exact(file, encoded, sizeof(encoded));
    }
    return status;
}

tokenizer_status tokenizer_save(const tokenizer *tokenizer, const char *path) {
    if (tokenizer == NULL || path == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    const size_t path_length = strlen(path);
    static const char suffix[] = ".tmp";
    if (path_length > SIZE_MAX - sizeof(suffix)) {
        return TOKENIZER_OVERFLOW;
    }
    char *temporary_path = malloc(path_length + sizeof(suffix));
    if (temporary_path == NULL) {
        return TOKENIZER_ALLOCATION_FAILED;
    }
    memcpy(temporary_path, path, path_length);
    memcpy(temporary_path + path_length, suffix, sizeof(suffix));

    FILE *file = fopen(temporary_path, "wbx");
    if (file == NULL) {
        free(temporary_path);
        return TOKENIZER_IO_ERROR;
    }

    tokenizer_status status = write_model(tokenizer, file);
    if (fclose(file) != 0 && status == TOKENIZER_OK) {
        status = TOKENIZER_IO_ERROR;
    }
    if (status == TOKENIZER_OK && rename(temporary_path, path) != 0) {
        status = TOKENIZER_IO_ERROR;
    }
    if (status != TOKENIZER_OK) {
        (void)remove(temporary_path);
    }
    free(temporary_path);
    return status;
}

static tokenizer_status validate_header(const unsigned char header[MODEL_HEADER_SIZE],
                                        uint32_t *out_merge_count) {
    const uint32_t version = load_little_endian_u32(header + 8U);
    const uint32_t header_size = load_little_endian_u32(header + 12U);
    const uint32_t base_vocabulary_size = load_little_endian_u32(header + 16U);
    const uint32_t merge_count = load_little_endian_u32(header + 20U);
    const uint64_t payload_size = load_little_endian_u64(header + 24U);
    const uint64_t expected_payload_size = (uint64_t)merge_count * MODEL_MERGE_SIZE;

    if (memcmp(header, model_magic, sizeof(model_magic)) != 0 || version != MODEL_FORMAT_VERSION ||
        header_size != MODEL_HEADER_SIZE ||
        base_vocabulary_size != TOKENIZER_BYTE_VOCABULARY_SIZE ||
        merge_count > UINT32_MAX - TOKENIZER_BYTE_VOCABULARY_SIZE ||
        payload_size != expected_payload_size) {
        return TOKENIZER_INVALID_MODEL;
    }
    *out_merge_count = merge_count;
    return TOKENIZER_OK;
}

tokenizer_status tokenizer_load(const char *path, tokenizer **out_tokenizer) {
    if (path == NULL || out_tokenizer == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }
    *out_tokenizer = NULL;

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return TOKENIZER_IO_ERROR;
    }

    unsigned char header[MODEL_HEADER_SIZE] = {0};
    tokenizer_status status = read_exact(file, header, sizeof(header));
    uint32_t merge_count = 0U;
    if (status == TOKENIZER_OK) {
        status = validate_header(header, &merge_count);
    }

    tokenizer *tokenizer = NULL;
    if (status == TOKENIZER_OK) {
        status = tokenizer_create_byte_level(&tokenizer);
    }

    tokenizer_sha256_context sha256;
    tokenizer_sha256_init(&sha256);
    for (uint32_t index = 0U; index < merge_count && status == TOKENIZER_OK; ++index) {
        unsigned char encoded[MODEL_MERGE_SIZE] = {0};
        status = read_exact(file, encoded, sizeof(encoded));
        if (status == TOKENIZER_OK) {
            tokenizer_sha256_update(&sha256, encoded, sizeof(encoded));
            const token_id left_id = load_little_endian_u32(encoded);
            const token_id right_id = load_little_endian_u32(encoded + 4U);
            status = tokenizer_add_merge(tokenizer, left_id, right_id);
        }
    }

    unsigned char checksum[TOKENIZER_SHA256_DIGEST_SIZE] = {0};
    tokenizer_sha256_final(&sha256, checksum);
    if (status == TOKENIZER_OK &&
        memcmp(checksum, header + MODEL_CHECKSUM_OFFSET, sizeof(checksum)) != 0) {
        status = TOKENIZER_INVALID_MODEL;
    }
    if (status == TOKENIZER_OK) {
        const int trailing_byte = fgetc(file);
        if (trailing_byte != EOF) {
            status = TOKENIZER_INVALID_MODEL;
        } else if (ferror(file) != 0) {
            status = TOKENIZER_IO_ERROR;
        }
    }
    if (fclose(file) != 0 && status == TOKENIZER_OK) {
        status = TOKENIZER_IO_ERROR;
    }
    if (status != TOKENIZER_OK) {
        tokenizer_destroy(tokenizer);
        return status;
    }

    *out_tokenizer = tokenizer;
    return TOKENIZER_OK;
}
