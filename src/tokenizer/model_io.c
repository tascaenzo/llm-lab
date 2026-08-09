#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tokenizer_internal.h"

tokenizer_status tokenizer_save(const tokenizer *tokenizer, const char *path) {
    if (tokenizer == NULL || path == NULL) {
        return TOKENIZER_INVALID_ARGUMENT;
    }

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return TOKENIZER_IO_ERROR;
    }

    int write_failed =
        fprintf(file, "llm-tokenizer 1\nbase-vocab 256\nmerges %zu\n", tokenizer->merge_count) < 0;

    for (size_t index = 0U; index < tokenizer->merge_count && write_failed == 0; ++index) {
        const bpe_merge merge = tokenizer->merges[index];
        write_failed =
            fprintf(file, "%" PRIu32 " %" PRIu32 "\n", merge.left_id, merge.right_id) < 0;
    }

    if (fclose(file) != 0) {
        write_failed = 1;
    }

    return write_failed != 0 ? TOKENIZER_IO_ERROR : TOKENIZER_OK;
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

    char magic[32] = {0};
    unsigned version = 0U;
    char base_name[32] = {0};
    uint32_t base_size = 0U;
    char merge_name[32] = {0};
    size_t merge_count = 0U;

    const int header_ok =
        fscanf(file, "%31s %u", magic, &version) == 2 && strcmp(magic, "llm-tokenizer") == 0 &&
        version == 1U && fscanf(file, "%31s %" SCNu32, base_name, &base_size) == 2 &&
        strcmp(base_name, "base-vocab") == 0 && base_size == TOKENIZER_BYTE_VOCABULARY_SIZE &&
        fscanf(file, "%31s %zu", merge_name, &merge_count) == 2 &&
        strcmp(merge_name, "merges") == 0;
    if (header_ok == 0) {
        fclose(file);
        return TOKENIZER_INVALID_MODEL;
    }

    tokenizer *tokenizer = NULL;
    tokenizer_status status = tokenizer_create_byte_level(&tokenizer);
    if (status != TOKENIZER_OK) {
        fclose(file);
        return status;
    }

    for (size_t index = 0U; index < merge_count && status == TOKENIZER_OK; ++index) {
        token_id left_id = 0U;
        token_id right_id = 0U;
        if (fscanf(file, "%" SCNu32 " %" SCNu32, &left_id, &right_id) != 2) {
            status = TOKENIZER_INVALID_MODEL;
            break;
        }

        status = tokenizer_add_merge(tokenizer, left_id, right_id);
    }

    int next = 0;
    while (status == TOKENIZER_OK && (next = fgetc(file)) != EOF) {
        if (next != ' ' && next != '\t' && next != '\r' && next != '\n') {
            status = TOKENIZER_INVALID_MODEL;
        }
    }

    if (ferror(file) != 0 && status == TOKENIZER_OK) {
        status = TOKENIZER_IO_ERROR;
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
