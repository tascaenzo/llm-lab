#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset_internal.h"

#define TOKENIZER_CHECKSUM_OFFSET 56U
#define PAYLOAD_CHECKSUM_OFFSET 88U

static const unsigned char dataset_magic[8] = {'L', 'L', 'M', 'D', 'A', 'T', 'A', '\n'};
static const char *const split_suffixes[LM_DATASET_SPLIT_COUNT] = {
    ".train.llmdat",
    ".validation.llmdat",
    ".test.llmdat",
};

void lm_dataset_store_u32(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8U);
    bytes[2] = (unsigned char)(value >> 16U);
    bytes[3] = (unsigned char)(value >> 24U);
}

void lm_dataset_store_u64(unsigned char *bytes, uint64_t value) {
    for (size_t index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
}

uint32_t lm_dataset_load_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) | ((uint32_t)bytes[2] << 16U) |
           ((uint32_t)bytes[3] << 24U);
}

uint64_t lm_dataset_load_u64(const unsigned char *bytes) {
    uint64_t value = 0U;
    for (size_t index = 0U; index < 8U; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static int file_seek(FILE *file, uint64_t offset, int origin) {
    if (offset > (uint64_t)LONG_MAX) {
        return -1;
    }
    return fseek(file, (long)offset, origin);
}

static char *append_suffix(const char *prefix, const char *suffix) {
    const size_t prefix_length = strlen(prefix);
    const size_t suffix_length = strlen(suffix);
    if (prefix_length > SIZE_MAX - suffix_length - 1U) {
        return NULL;
    }
    char *path = malloc(prefix_length + suffix_length + 1U);
    if (path == NULL) {
        return NULL;
    }
    memcpy(path, prefix, prefix_length);
    memcpy(path + prefix_length, suffix, suffix_length + 1U);
    return path;
}

static int path_exists(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    (void)fclose(file);
    return 1;
}

lm_dataset_status lm_dataset_writer_open(lm_dataset_writer *writer, const char *output_prefix,
                                         lm_dataset_split split) {
    if (writer == NULL || output_prefix == NULL || (unsigned int)split >= LM_DATASET_SPLIT_COUNT) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *writer = (lm_dataset_writer){.split = split};
    writer->final_path = append_suffix(output_prefix, split_suffixes[split]);
    if (writer->final_path == NULL) {
        return LM_DATASET_ALLOCATION_FAILED;
    }
    writer->temporary_path = append_suffix(writer->final_path, ".part");
    if (writer->temporary_path == NULL) {
        free(writer->final_path);
        writer->final_path = NULL;
        return LM_DATASET_ALLOCATION_FAILED;
    }
    if (path_exists(writer->final_path) != 0 || path_exists(writer->temporary_path) != 0) {
        return LM_DATASET_OUTPUT_EXISTS;
    }

    writer->file = fopen(writer->temporary_path, "wbx");
    if (writer->file == NULL) {
        return LM_DATASET_IO_ERROR;
    }
    writer->temporary_created = 1;
    unsigned char empty_header[LM_DATASET_HEADER_SIZE] = {0};
    if (fwrite(empty_header, 1U, sizeof(empty_header), writer->file) != sizeof(empty_header)) {
        return LM_DATASET_IO_ERROR;
    }
    tokenizer_sha256_init(&writer->payload_sha256);
    return LM_DATASET_OK;
}

static lm_dataset_status writer_write_token(lm_dataset_writer *writer, token_id token) {
    unsigned char encoded[LM_DATASET_TOKEN_SIZE] = {0};
    lm_dataset_store_u32(encoded, token);
    if (fwrite(encoded, 1U, sizeof(encoded), writer->file) != sizeof(encoded)) {
        return LM_DATASET_IO_ERROR;
    }
    tokenizer_sha256_update(&writer->payload_sha256, encoded, sizeof(encoded));
    if (writer->token_count == UINT64_MAX) {
        return LM_DATASET_OVERFLOW;
    }
    ++writer->token_count;
    return LM_DATASET_OK;
}

lm_dataset_status lm_dataset_writer_append(lm_dataset_writer *writer, const token_id *tokens,
                                           size_t token_count, token_id end_of_document_token) {
    if (writer == NULL || writer->file == NULL || (tokens == NULL && token_count != 0U)) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    lm_dataset_status status = LM_DATASET_OK;
    for (size_t index = 0U; index < token_count && status == LM_DATASET_OK; ++index) {
        status = writer_write_token(writer, tokens[index]);
    }
    if (status == LM_DATASET_OK) {
        status = writer_write_token(writer, end_of_document_token);
    }
    if (status == LM_DATASET_OK) {
        if (writer->document_count == UINT64_MAX) {
            return LM_DATASET_OVERFLOW;
        }
        ++writer->document_count;
    }
    return status;
}

static lm_dataset_status writer_finish(lm_dataset_writer *writer, uint32_t model_vocabulary_size,
                                       uint32_t tokenizer_vocabulary_size,
                                       const unsigned char tokenizer_checksum[32]) {
    if (writer->token_count < 2U || writer->document_count == 0U ||
        writer->token_count > UINT64_MAX / LM_DATASET_TOKEN_SIZE) {
        return LM_DATASET_INSUFFICIENT_DATA;
    }

    unsigned char payload_checksum[32] = {0};
    tokenizer_sha256_final(&writer->payload_sha256, payload_checksum);
    unsigned char header[LM_DATASET_HEADER_SIZE] = {0};
    memcpy(header, dataset_magic, sizeof(dataset_magic));
    lm_dataset_store_u32(header + 8U, LM_DATASET_FORMAT_VERSION);
    lm_dataset_store_u32(header + 12U, LM_DATASET_HEADER_SIZE);
    lm_dataset_store_u32(header + 16U, tokenizer_vocabulary_size);
    lm_dataset_store_u32(header + 20U, model_vocabulary_size);
    lm_dataset_store_u32(header + 24U, tokenizer_vocabulary_size);
    lm_dataset_store_u32(header + 28U, (uint32_t)writer->split);
    lm_dataset_store_u64(header + 32U, writer->token_count);
    lm_dataset_store_u64(header + 40U, writer->document_count);
    lm_dataset_store_u64(header + 48U, writer->token_count * LM_DATASET_TOKEN_SIZE);
    memcpy(header + TOKENIZER_CHECKSUM_OFFSET, tokenizer_checksum, 32U);
    memcpy(header + PAYLOAD_CHECKSUM_OFFSET, payload_checksum, 32U);

    if (file_seek(writer->file, 0U, SEEK_SET) != 0 ||
        fwrite(header, 1U, sizeof(header), writer->file) != sizeof(header)) {
        return LM_DATASET_IO_ERROR;
    }
    if (fclose(writer->file) != 0) {
        writer->file = NULL;
        return LM_DATASET_IO_ERROR;
    }
    writer->file = NULL;
    return LM_DATASET_OK;
}

lm_dataset_status
lm_dataset_writers_publish(lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT],
                           uint32_t model_vocabulary_size, uint32_t tokenizer_vocabulary_size,
                           const unsigned char tokenizer_checksum[TOKENIZER_SHA256_DIGEST_SIZE]) {
    if (tokenizer_vocabulary_size == UINT32_MAX ||
        model_vocabulary_size < tokenizer_vocabulary_size + 1U) {
        return LM_DATASET_OVERFLOW;
    }
    lm_dataset_status status = LM_DATASET_OK;
    for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT && status == LM_DATASET_OK; ++index) {
        status = writer_finish(&writers[index], model_vocabulary_size, tokenizer_vocabulary_size,
                               tokenizer_checksum);
    }
    for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT && status == LM_DATASET_OK; ++index) {
        if (rename(writers[index].temporary_path, writers[index].final_path) != 0) {
            status = LM_DATASET_IO_ERROR;
        } else {
            writers[index].published = 1;
        }
    }
    if (status != LM_DATASET_OK) {
        lm_dataset_writers_abort(writers);
    }
    return status;
}

void lm_dataset_writers_abort(lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT]) {
    for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT; ++index) {
        if (writers[index].file != NULL) {
            (void)fclose(writers[index].file);
            writers[index].file = NULL;
        }
        if (writers[index].published != 0 && writers[index].final_path != NULL) {
            (void)remove(writers[index].final_path);
        }
        if (writers[index].temporary_created != 0 && writers[index].temporary_path != NULL) {
            (void)remove(writers[index].temporary_path);
        }
        free(writers[index].final_path);
        free(writers[index].temporary_path);
        writers[index].final_path = NULL;
        writers[index].temporary_path = NULL;
    }
}

static lm_dataset_status validate_header(const unsigned char header[LM_DATASET_HEADER_SIZE],
                                         lm_dataset *dataset) {
    const uint32_t version = lm_dataset_load_u32(header + 8U);
    const uint32_t header_size = lm_dataset_load_u32(header + 12U);
    const uint32_t tokenizer_vocabulary_size = lm_dataset_load_u32(header + 16U);
    const uint32_t model_vocabulary_size = lm_dataset_load_u32(header + 20U);
    const token_id end_of_document_token = lm_dataset_load_u32(header + 24U);
    const uint32_t split = lm_dataset_load_u32(header + 28U);
    const uint64_t token_count = lm_dataset_load_u64(header + 32U);
    const uint64_t document_count = lm_dataset_load_u64(header + 40U);
    const uint64_t payload_size = lm_dataset_load_u64(header + 48U);

    unsigned char reserved[8] = {0};
    if (memcmp(header, dataset_magic, sizeof(dataset_magic)) != 0 ||
        version != LM_DATASET_FORMAT_VERSION || header_size != LM_DATASET_HEADER_SIZE ||
        tokenizer_vocabulary_size == UINT32_MAX ||
        model_vocabulary_size < tokenizer_vocabulary_size + 1U ||
        end_of_document_token != tokenizer_vocabulary_size || split >= LM_DATASET_SPLIT_COUNT ||
        token_count < 2U || document_count == 0U ||
        token_count > UINT64_MAX / LM_DATASET_TOKEN_SIZE ||
        payload_size != token_count * LM_DATASET_TOKEN_SIZE ||
        memcmp(header + 120U, reserved, sizeof(reserved)) != 0) {
        return LM_DATASET_INVALID_FORMAT;
    }

    dataset->token_count = token_count;
    dataset->document_count = document_count;
    dataset->tokenizer_vocabulary_size = tokenizer_vocabulary_size;
    dataset->model_vocabulary_size = model_vocabulary_size;
    dataset->end_of_document_token = end_of_document_token;
    dataset->split = (lm_dataset_split)split;
    return LM_DATASET_OK;
}

lm_dataset_status lm_dataset_open_with_progress(const char *path,
                                                lm_dataset_open_progress_callback progress_callback,
                                                void *progress_context, lm_dataset **out_dataset) {
    if (path == NULL || out_dataset == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_dataset = NULL;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return LM_DATASET_IO_ERROR;
    }

    unsigned char header[LM_DATASET_HEADER_SIZE] = {0};
    lm_dataset_status status = fread(header, 1U, sizeof(header), file) == sizeof(header)
                                   ? LM_DATASET_OK
                                   : LM_DATASET_INVALID_FORMAT;
    lm_dataset *dataset = NULL;
    if (status == LM_DATASET_OK) {
        dataset = calloc(1U, sizeof(*dataset));
        status = dataset == NULL ? LM_DATASET_ALLOCATION_FAILED : validate_header(header, dataset);
    }

    tokenizer_sha256_context sha256;
    tokenizer_sha256_init(&sha256);
    uint64_t remaining = dataset == NULL ? 0U : dataset->token_count;
    const uint64_t total_payload_bytes = remaining * LM_DATASET_TOKEN_SIZE;
    uint64_t checked_payload_bytes = 0U;
    if (status == LM_DATASET_OK && progress_callback != NULL) {
        progress_callback(checked_payload_bytes, total_payload_bytes, progress_context);
    }
    unsigned char buffer[64U * 1024U];
    while (status == LM_DATASET_OK && remaining != 0U) {
        const uint64_t maximum_tokens = sizeof(buffer) / LM_DATASET_TOKEN_SIZE;
        const size_t tokens_to_read =
            (size_t)(remaining < maximum_tokens ? remaining : maximum_tokens);
        const size_t bytes_to_read = tokens_to_read * LM_DATASET_TOKEN_SIZE;
        if (fread(buffer, 1U, bytes_to_read, file) != bytes_to_read) {
            status = ferror(file) != 0 ? LM_DATASET_IO_ERROR : LM_DATASET_INVALID_FORMAT;
            break;
        }
        tokenizer_sha256_update(&sha256, buffer, bytes_to_read);
        for (size_t index = 0U; index < tokens_to_read; ++index) {
            if (lm_dataset_load_u32(buffer + index * LM_DATASET_TOKEN_SIZE) >=
                dataset->model_vocabulary_size) {
                status = LM_DATASET_INVALID_FORMAT;
                break;
            }
        }
        remaining -= tokens_to_read;
        checked_payload_bytes += bytes_to_read;
        if (progress_callback != NULL) {
            progress_callback(checked_payload_bytes, total_payload_bytes, progress_context);
        }
    }
    unsigned char checksum[32] = {0};
    tokenizer_sha256_final(&sha256, checksum);
    if (status == LM_DATASET_OK &&
        memcmp(checksum, header + PAYLOAD_CHECKSUM_OFFSET, sizeof(checksum)) != 0) {
        status = LM_DATASET_INVALID_FORMAT;
    }
    if (status == LM_DATASET_OK) {
        const int trailing = fgetc(file);
        if (trailing != EOF) {
            status = LM_DATASET_INVALID_FORMAT;
        } else if (ferror(file) != 0) {
            status = LM_DATASET_IO_ERROR;
        }
    }
    if (status == LM_DATASET_OK && file_seek(file, LM_DATASET_HEADER_SIZE, SEEK_SET) != 0) {
        status = LM_DATASET_IO_ERROR;
    }
    if (status != LM_DATASET_OK) {
        free(dataset);
        (void)fclose(file);
        return status;
    }
    dataset->file = file;
    *out_dataset = dataset;
    return LM_DATASET_OK;
}

lm_dataset_status lm_dataset_open(const char *path, lm_dataset **out_dataset) {
    return lm_dataset_open_with_progress(path, NULL, NULL, out_dataset);
}

void lm_dataset_close(lm_dataset *dataset) {
    if (dataset == NULL) {
        return;
    }
    if (dataset->file != NULL) {
        (void)fclose(dataset->file);
    }
    free(dataset);
}

lm_dataset_status lm_dataset_read_tokens(lm_dataset *dataset, uint64_t token_offset, size_t count,
                                         token_id *out_tokens) {
    if (dataset == NULL || (out_tokens == NULL && count != 0U) ||
        token_offset > dataset->token_count ||
        (uint64_t)count > dataset->token_count - token_offset ||
        token_offset > (UINT64_MAX - LM_DATASET_HEADER_SIZE) / LM_DATASET_TOKEN_SIZE) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    const uint64_t byte_offset = LM_DATASET_HEADER_SIZE + token_offset * LM_DATASET_TOKEN_SIZE;
    if (file_seek(dataset->file, byte_offset, SEEK_SET) != 0) {
        return LM_DATASET_IO_ERROR;
    }
    unsigned char encoded[LM_DATASET_TOKEN_SIZE] = {0};
    for (size_t index = 0U; index < count; ++index) {
        if (fread(encoded, 1U, sizeof(encoded), dataset->file) != sizeof(encoded)) {
            return LM_DATASET_IO_ERROR;
        }
        out_tokens[index] = lm_dataset_load_u32(encoded);
    }
    return LM_DATASET_OK;
}

uint64_t lm_dataset_token_count(const lm_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->token_count;
}

uint64_t lm_dataset_document_count(const lm_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->document_count;
}

uint32_t lm_dataset_tokenizer_vocabulary_size(const lm_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->tokenizer_vocabulary_size;
}

uint32_t lm_dataset_model_vocabulary_size(const lm_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->model_vocabulary_size;
}

token_id lm_dataset_end_of_document_token(const lm_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->end_of_document_token;
}

lm_dataset_split lm_dataset_get_split(const lm_dataset *dataset) {
    return dataset == NULL ? LM_DATASET_TRAIN : dataset->split;
}

const char *lm_dataset_status_string(lm_dataset_status status) {
    switch (status) {
    case LM_DATASET_OK:
        return "ok";
    case LM_DATASET_INVALID_ARGUMENT:
        return "invalid argument";
    case LM_DATASET_ALLOCATION_FAILED:
        return "allocation failed";
    case LM_DATASET_OVERFLOW:
        return "numeric overflow";
    case LM_DATASET_IO_ERROR:
        return "I/O error";
    case LM_DATASET_OUTPUT_EXISTS:
        return "output dataset or incomplete .part file already exists";
    case LM_DATASET_INVALID_JSONL:
        return "invalid documents JSONL";
    case LM_DATASET_DUPLICATE_DOCUMENT:
        return "duplicate document ID";
    case LM_DATASET_INVALID_FORMAT:
        return "invalid dataset format";
    case LM_DATASET_INSUFFICIENT_DATA:
        return "insufficient data in one or more splits";
    case LM_DATASET_TOKENIZER_ERROR:
        return "tokenizer error";
    }
    return "unknown dataset error";
}
