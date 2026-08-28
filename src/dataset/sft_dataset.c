#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset/sft_dataset.h"
#include "json_tokens.h"
#include "tokenizer/sha256.h"

#define LM_SFT_FORMAT_VERSION UINT32_C(1)
#define LM_SFT_HEADER_SIZE 160U
#define LM_SFT_TOKEN_SIZE 4U
#define LM_SFT_SPLIT_COUNT 3U

static const unsigned char sft_magic[8] = {'L', 'L', 'M', 'S', 'F', 'T', '\n', '\0'};
static const char *const sft_suffixes[LM_SFT_SPLIT_COUNT] = {".train.llmsft", ".validation.llmsft",
                                                             ".test.llmsft"};

typedef struct sft_writer {
    FILE *file;
    char *final_path;
    char *temporary_path;
    lm_dataset_split split;
    uint64_t example_count;
    uint64_t supervised_token_count;
    tokenizer_sha256_context payload_checksum;
    int published;
} sft_writer;

typedef struct sft_sequence {
    token_id *tokens;
    uint32_t *assistant_targets;
    size_t length;
    size_t capacity;
} sft_sequence;

typedef struct sft_id_set {
    uint64_t *hashes;
    unsigned char *occupied;
    size_t count;
    size_t capacity;
} sft_id_set;

struct lm_sft_dataset {
    FILE *file;
    uint64_t example_count;
    uint64_t supervised_token_count;
    uint32_t tokenizer_vocabulary_size;
    uint32_t model_vocabulary_size;
    size_t context_length;
    lm_dataset_split split;
    lm_chat_protocol protocol;
    uint64_t record_byte_count;
    unsigned char tokenizer_checksum[TOKENIZER_SHA256_DIGEST_SIZE];
};

struct lm_sft_batcher {
    lm_sft_dataset *dataset;
    size_t batch_size;
    uint64_t random_state;
    uint64_t epoch;
    uint64_t sample_index;
    uint64_t next_offset;
    uint64_t stride;
};

static void store_u32(unsigned char *bytes, uint32_t value) {
    for (size_t index = 0U; index < 4U; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
}

static void store_u64(unsigned char *bytes, uint64_t value) {
    for (size_t index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
}

static uint32_t load_u32(const unsigned char *bytes) {
    uint32_t value = 0U;
    for (size_t index = 0U; index < 4U; ++index) {
        value |= (uint32_t)bytes[index] << (index * 8U);
    }
    return value;
}

static uint64_t load_u64(const unsigned char *bytes) {
    uint64_t value = 0U;
    for (size_t index = 0U; index < 8U; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static int seek_file(FILE *file, uint64_t offset) {
    return offset <= (uint64_t)LONG_MAX ? fseek(file, (long)offset, SEEK_SET) : -1;
}

static lm_dataset_status checksum_file(const char *path, unsigned char digest[32]) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return LM_DATASET_IO_ERROR;
    }
    tokenizer_sha256_context checksum = {0};
    tokenizer_sha256_init(&checksum);
    unsigned char buffer[64U * 1024U];
    size_t count = 0U;
    while ((count = fread(buffer, 1U, sizeof(buffer), file)) != 0U) {
        tokenizer_sha256_update(&checksum, buffer, count);
    }
    lm_dataset_status status = ferror(file) == 0 ? LM_DATASET_OK : LM_DATASET_IO_ERROR;
    if (fclose(file) != 0) {
        status = LM_DATASET_IO_ERROR;
    }
    if (status == LM_DATASET_OK) {
        tokenizer_sha256_final(&checksum, digest);
    }
    return status;
}

lm_dataset_status lm_chat_protocol_v1(uint32_t tokenizer_vocabulary_size,
                                      uint32_t model_vocabulary_size,
                                      lm_chat_protocol *out_protocol) {
    if (out_protocol == NULL || tokenizer_vocabulary_size > UINT32_MAX - 8U ||
        model_vocabulary_size < tokenizer_vocabulary_size + 1U + LM_CHAT_RESERVED_TOKEN_COUNT) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_protocol = (lm_chat_protocol){
        .system_token = tokenizer_vocabulary_size + 1U,
        .user_token = tokenizer_vocabulary_size + 2U,
        .assistant_token = tokenizer_vocabulary_size + 3U,
        .end_token = tokenizer_vocabulary_size + 4U,
        .padding_token = tokenizer_vocabulary_size + 5U,
    };
    return LM_DATASET_OK;
}

static char *path_with_suffix(const char *prefix, const char *suffix) {
    const size_t prefix_length = strlen(prefix);
    const size_t suffix_length = strlen(suffix);
    if (prefix_length > SIZE_MAX - suffix_length - 1U) {
        return NULL;
    }
    char *path = malloc(prefix_length + suffix_length + 1U);
    if (path != NULL) {
        memcpy(path, prefix, prefix_length);
        memcpy(path + prefix_length, suffix, suffix_length + 1U);
    }
    return path;
}

static int file_exists(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    (void)fclose(file);
    return 1;
}

static lm_dataset_status writer_open(sft_writer *writer, const char *prefix,
                                     lm_dataset_split split) {
    *writer = (sft_writer){.split = split};
    writer->final_path = path_with_suffix(prefix, sft_suffixes[split]);
    writer->temporary_path =
        writer->final_path == NULL ? NULL : path_with_suffix(writer->final_path, ".part");
    if (writer->final_path == NULL || writer->temporary_path == NULL) {
        return LM_DATASET_ALLOCATION_FAILED;
    }
    if (file_exists(writer->final_path) != 0 || file_exists(writer->temporary_path) != 0) {
        return LM_DATASET_OUTPUT_EXISTS;
    }
    writer->file = fopen(writer->temporary_path, "wbx");
    if (writer->file == NULL) {
        return LM_DATASET_IO_ERROR;
    }
    unsigned char header[LM_SFT_HEADER_SIZE] = {0};
    if (fwrite(header, 1U, sizeof(header), writer->file) != sizeof(header)) {
        return LM_DATASET_IO_ERROR;
    }
    tokenizer_sha256_init(&writer->payload_checksum);
    return LM_DATASET_OK;
}

static void writers_abort(sft_writer writers[LM_SFT_SPLIT_COUNT]) {
    for (size_t index = 0U; index < LM_SFT_SPLIT_COUNT; ++index) {
        if (writers[index].file != NULL) {
            (void)fclose(writers[index].file);
        }
        if (writers[index].published != 0 && writers[index].final_path != NULL) {
            (void)remove(writers[index].final_path);
        }
        if (writers[index].temporary_path != NULL) {
            (void)remove(writers[index].temporary_path);
        }
        free(writers[index].final_path);
        free(writers[index].temporary_path);
    }
}

static lm_dataset_status writer_u32(sft_writer *writer, uint32_t value) {
    unsigned char bytes[4] = {0};
    store_u32(bytes, value);
    if (fwrite(bytes, 1U, sizeof(bytes), writer->file) != sizeof(bytes)) {
        return LM_DATASET_IO_ERROR;
    }
    tokenizer_sha256_update(&writer->payload_checksum, bytes, sizeof(bytes));
    return LM_DATASET_OK;
}

static lm_dataset_status writer_append(sft_writer *writer, const sft_sequence *sequence,
                                       size_t context_length, token_id padding_token) {
    if (sequence->length < 2U || sequence->length > context_length + 1U) {
        return LM_DATASET_INVALID_FORMAT;
    }
    size_t supervised_count = 0U;
    for (size_t part = 0U; part < 3U; ++part) {
        for (size_t position = 0U; position < context_length; ++position) {
            uint32_t value = 0U;
            if (part == 0U) {
                value =
                    position + 1U < sequence->length ? sequence->tokens[position] : padding_token;
            } else if (part == 1U) {
                value = position + 1U < sequence->length ? sequence->tokens[position + 1U]
                                                         : padding_token;
            } else {
                value = position + 1U < sequence->length
                            ? sequence->assistant_targets[position + 1U]
                            : 0U;
                supervised_count += value;
            }
            const lm_dataset_status status = writer_u32(writer, value);
            if (status != LM_DATASET_OK) {
                return status;
            }
        }
    }
    if (supervised_count == 0U || writer->example_count == UINT64_MAX ||
        supervised_count > UINT64_MAX - writer->supervised_token_count) {
        return LM_DATASET_INVALID_FORMAT;
    }
    ++writer->example_count;
    writer->supervised_token_count += supervised_count;
    return LM_DATASET_OK;
}

static lm_dataset_status writer_finish(sft_writer *writer, uint32_t tokenizer_vocabulary_size,
                                       uint32_t model_vocabulary_size, size_t context_length,
                                       const lm_chat_protocol *protocol,
                                       const unsigned char tokenizer_checksum[32]) {
    if (writer->example_count == 0U || context_length > UINT32_MAX ||
        context_length > UINT64_MAX / (UINT64_C(3) * LM_SFT_TOKEN_SIZE) ||
        writer->example_count >
            UINT64_MAX / ((uint64_t)context_length * UINT64_C(3) * LM_SFT_TOKEN_SIZE)) {
        return LM_DATASET_INSUFFICIENT_DATA;
    }
    unsigned char payload_checksum[32] = {0};
    tokenizer_sha256_final(&writer->payload_checksum, payload_checksum);
    const uint64_t payload_bytes =
        writer->example_count * (uint64_t)context_length * UINT64_C(3) * LM_SFT_TOKEN_SIZE;
    unsigned char header[LM_SFT_HEADER_SIZE] = {0};
    memcpy(header, sft_magic, sizeof(sft_magic));
    store_u32(header + 8U, LM_SFT_FORMAT_VERSION);
    store_u32(header + 12U, LM_SFT_HEADER_SIZE);
    store_u32(header + 16U, tokenizer_vocabulary_size);
    store_u32(header + 20U, model_vocabulary_size);
    store_u32(header + 24U, (uint32_t)context_length);
    store_u32(header + 28U, (uint32_t)writer->split);
    store_u64(header + 32U, writer->example_count);
    store_u64(header + 40U, writer->supervised_token_count);
    store_u64(header + 48U, payload_bytes);
    memcpy(header + 56U, tokenizer_checksum, 32U);
    memcpy(header + 88U, payload_checksum, 32U);
    store_u32(header + 120U, protocol->system_token);
    store_u32(header + 124U, protocol->user_token);
    store_u32(header + 128U, protocol->assistant_token);
    store_u32(header + 132U, protocol->end_token);
    store_u32(header + 136U, protocol->padding_token);
    if (seek_file(writer->file, 0U) != 0 ||
        fwrite(header, 1U, sizeof(header), writer->file) != sizeof(header) ||
        fclose(writer->file) != 0) {
        writer->file = NULL;
        return LM_DATASET_IO_ERROR;
    }
    writer->file = NULL;
    return LM_DATASET_OK;
}

static uint64_t fnv1a_64(const unsigned char *bytes, size_t length) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < length; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void id_set_destroy(sft_id_set *set) {
    free(set->hashes);
    free(set->occupied);
    *set = (sft_id_set){0};
}

static int id_set_resize(sft_id_set *set, size_t capacity) {
    uint64_t *hashes = calloc(capacity, sizeof(*hashes));
    unsigned char *occupied = calloc(capacity, sizeof(*occupied));
    if (hashes == NULL || occupied == NULL) {
        free(hashes);
        free(occupied);
        return 0;
    }
    for (size_t index = 0U; index < set->capacity; ++index) {
        if (set->occupied[index] == 0U)
            continue;
        size_t slot = (size_t)(set->hashes[index] & (capacity - 1U));
        while (occupied[slot] != 0U)
            slot = (slot + 1U) & (capacity - 1U);
        hashes[slot] = set->hashes[index];
        occupied[slot] = 1U;
    }
    free(set->hashes);
    free(set->occupied);
    set->hashes = hashes;
    set->occupied = occupied;
    set->capacity = capacity;
    return 1;
}

static int id_set_insert(sft_id_set *set, uint64_t hash) {
    if (set->capacity == 0U && id_set_resize(set, 16U) == 0)
        return -1;
    if (set->count >= set->capacity - set->capacity / 4U) {
        if (set->capacity > SIZE_MAX / 2U || id_set_resize(set, set->capacity * 2U) == 0)
            return -1;
    }
    size_t slot = (size_t)(hash & (set->capacity - 1U));
    while (set->occupied[slot] != 0U) {
        if (set->hashes[slot] == hash)
            return 0;
        slot = (slot + 1U) & (set->capacity - 1U);
    }
    set->hashes[slot] = hash;
    set->occupied[slot] = 1U;
    ++set->count;
    return 1;
}

static lm_dataset_split split_for_id(const unsigned char *id, size_t length) {
    const uint64_t bucket = fnv1a_64(id, length) % UINT64_C(10000);
    return bucket < UINT64_C(9000)
               ? LM_DATASET_TRAIN
               : (bucket < UINT64_C(9500) ? LM_DATASET_VALIDATION : LM_DATASET_TEST);
}

static int sequence_append(sft_sequence *sequence, token_id token, uint32_t assistant_target) {
    if (sequence->length == sequence->capacity) {
        size_t capacity = sequence->capacity == 0U ? 64U : sequence->capacity * 2U;
        if (capacity < sequence->capacity || capacity > SIZE_MAX / sizeof(*sequence->tokens)) {
            return 0;
        }
        token_id *tokens = realloc(sequence->tokens, capacity * sizeof(*tokens));
        if (tokens == NULL) {
            return 0;
        }
        sequence->tokens = tokens;
        uint32_t *targets =
            realloc(sequence->assistant_targets, capacity * sizeof(*sequence->assistant_targets));
        if (targets == NULL) {
            return 0;
        }
        sequence->assistant_targets = targets;
        sequence->capacity = capacity;
    }
    sequence->tokens[sequence->length] = token;
    sequence->assistant_targets[sequence->length] = assistant_target;
    ++sequence->length;
    return 1;
}

static void sequence_destroy(sft_sequence *sequence) {
    free(sequence->tokens);
    free(sequence->assistant_targets);
    *sequence = (sft_sequence){0};
}

static size_t object_value(const unsigned char *json, const lm_json_token *tokens,
                           size_t token_count, size_t object_index, const char *key) {
    for (size_t index = object_index + 1U; index + 1U < token_count; ++index) {
        if (tokens[index].parent == object_index && tokens[index].type == LM_JSON_STRING &&
            lm_json_token_equals(json, &tokens[index], key) != 0) {
            return index + 1U;
        }
    }
    return SIZE_MAX;
}

static int decoded_string_has_content(const unsigned char *json, const lm_json_token *token,
                                      lm_dataset_status *out_status) {
    unsigned char *value = NULL;
    size_t length = 0U;
    const int decode_status = lm_json_decode_string(json, token, &value, &length);
    if (decode_status != 1) {
        *out_status = decode_status < 0 ? LM_DATASET_ALLOCATION_FAILED : LM_DATASET_INVALID_JSONL;
        return 0;
    }
    int has_content = 0;
    for (size_t index = 0U; index < length; ++index) {
        if (value[index] == '\0') {
            has_content = 0;
            break;
        }
        if (isspace(value[index]) == 0) {
            has_content = 1;
        }
    }
    free(value);
    if (has_content == 0) {
        *out_status = LM_DATASET_INVALID_FORMAT;
    }
    return has_content;
}

static lm_dataset_status append_message(const unsigned char *json, const lm_json_token *tokens,
                                        size_t token_count, size_t message_index,
                                        const tokenizer *text_tokenizer,
                                        const lm_chat_protocol *protocol, int expected_role,
                                        sft_sequence *sequence) {
    const size_t role_index = object_value(json, tokens, token_count, message_index, "role");
    const size_t content_index = object_value(json, tokens, token_count, message_index, "content");
    if (role_index == SIZE_MAX || content_index == SIZE_MAX ||
        tokens[role_index].type != LM_JSON_STRING || tokens[content_index].type != LM_JSON_STRING) {
        return LM_DATASET_INVALID_JSONL;
    }
    token_id role_token = 0U;
    int is_assistant = 0;
    if (lm_json_token_equals(json, &tokens[role_index], "system") != 0) {
        role_token = protocol->system_token;
        if (expected_role != 0) {
            return LM_DATASET_INVALID_FORMAT;
        }
    } else if (lm_json_token_equals(json, &tokens[role_index], "user") != 0) {
        role_token = protocol->user_token;
        if (expected_role != 1) {
            return LM_DATASET_INVALID_FORMAT;
        }
    } else if (lm_json_token_equals(json, &tokens[role_index], "assistant") != 0) {
        role_token = protocol->assistant_token;
        is_assistant = 1;
        if (expected_role != 2) {
            return LM_DATASET_INVALID_FORMAT;
        }
    } else {
        return LM_DATASET_INVALID_FORMAT;
    }
    unsigned char *content = NULL;
    size_t content_length = 0U;
    int decode_status =
        lm_json_decode_string(json, &tokens[content_index], &content, &content_length);
    if (decode_status != 1) {
        return decode_status < 0 ? LM_DATASET_ALLOCATION_FAILED : LM_DATASET_INVALID_JSONL;
    }
    int has_content = 0;
    for (size_t index = 0U; index < content_length; ++index) {
        if (content[index] == '\0') {
            has_content = 0;
            break;
        }
        if (isspace(content[index]) == 0) {
            has_content = 1;
        }
    }
    if (has_content == 0) {
        free(content);
        return LM_DATASET_INVALID_FORMAT;
    }
    token_sequence content_tokens = {0};
    const tokenizer_status tokenize_status =
        tokenizer_encode(text_tokenizer, content, content_length, &content_tokens);
    free(content);
    if (tokenize_status != TOKENIZER_OK) {
        return tokenize_status == TOKENIZER_ALLOCATION_FAILED ? LM_DATASET_ALLOCATION_FAILED
                                                              : LM_DATASET_TOKENIZER_ERROR;
    }
    lm_dataset_status result = sequence_append(sequence, role_token, 0U) != 0
                                   ? LM_DATASET_OK
                                   : LM_DATASET_ALLOCATION_FAILED;
    for (size_t index = 0U; result == LM_DATASET_OK && index < content_tokens.length; ++index) {
        if (sequence_append(sequence, content_tokens.ids[index], is_assistant != 0 ? 1U : 0U) ==
            0) {
            result = LM_DATASET_ALLOCATION_FAILED;
        }
    }
    if (result == LM_DATASET_OK &&
        sequence_append(sequence, protocol->end_token, is_assistant != 0 ? 1U : 0U) == 0) {
        result = LM_DATASET_ALLOCATION_FAILED;
    }
    token_sequence_destroy(&content_tokens);
    return result;
}

static lm_dataset_status parse_conversation(const unsigned char *json, size_t json_length,
                                            const tokenizer *text_tokenizer,
                                            const lm_chat_protocol *protocol, size_t context_length,
                                            lm_dataset_split *out_split, uint64_t *out_id_hash,
                                            sft_sequence *out_sequence) {
    size_t capacity = 64U;
    lm_json_token *tokens = NULL;
    size_t token_count = 0U;
    int parse_status = -1;
    while (parse_status < 0) {
        lm_json_token *resized = realloc(tokens, capacity * sizeof(*tokens));
        if (resized == NULL) {
            free(tokens);
            return LM_DATASET_ALLOCATION_FAILED;
        }
        tokens = resized;
        parse_status = lm_json_tokenize(json, json_length, tokens, capacity, &token_count);
        if (parse_status < 0) {
            if (capacity > SIZE_MAX / 2U) {
                free(tokens);
                return LM_DATASET_OVERFLOW;
            }
            capacity *= 2U;
        }
    }
    if (parse_status == 0 || token_count == 0U || tokens[0].type != LM_JSON_OBJECT) {
        free(tokens);
        return LM_DATASET_INVALID_JSONL;
    }
    const size_t id_index = object_value(json, tokens, token_count, 0U, "id");
    const size_t source_index = object_value(json, tokens, token_count, 0U, "source");
    const size_t license_index = object_value(json, tokens, token_count, 0U, "license");
    const size_t messages_index = object_value(json, tokens, token_count, 0U, "messages");
    if (id_index == SIZE_MAX || source_index == SIZE_MAX || license_index == SIZE_MAX ||
        messages_index == SIZE_MAX || tokens[id_index].type != LM_JSON_STRING ||
        tokens[source_index].type != LM_JSON_STRING ||
        tokens[license_index].type != LM_JSON_STRING ||
        tokens[source_index].end == tokens[source_index].start ||
        tokens[license_index].end == tokens[license_index].start ||
        tokens[messages_index].type != LM_JSON_ARRAY) {
        free(tokens);
        return LM_DATASET_INVALID_JSONL;
    }
    lm_dataset_status metadata_status = LM_DATASET_OK;
    if (decoded_string_has_content(json, &tokens[source_index], &metadata_status) == 0 ||
        decoded_string_has_content(json, &tokens[license_index], &metadata_status) == 0) {
        free(tokens);
        return metadata_status;
    }
    unsigned char *id = NULL;
    size_t id_length = 0U;
    const int id_status = lm_json_decode_string(json, &tokens[id_index], &id, &id_length);
    if (id_status != 1 || id_length == 0U) {
        free(id);
        free(tokens);
        return id_status < 0 ? LM_DATASET_ALLOCATION_FAILED : LM_DATASET_INVALID_JSONL;
    }
    *out_id_hash = fnv1a_64(id, id_length);
    *out_split = split_for_id(id, id_length);
    free(id);

    int expected_role = 0;
    size_t message_count = 0U;
    int last_was_assistant = 0;
    lm_dataset_status result = LM_DATASET_OK;
    for (size_t index = messages_index + 1U; index < token_count; ++index) {
        if (tokens[index].parent != messages_index) {
            continue;
        }
        if (tokens[index].type != LM_JSON_OBJECT) {
            result = LM_DATASET_INVALID_FORMAT;
            break;
        }
        size_t role_index = object_value(json, tokens, token_count, index, "role");
        if (role_index == SIZE_MAX) {
            result = LM_DATASET_INVALID_JSONL;
            break;
        }
        if (message_count == 0U && lm_json_token_equals(json, &tokens[role_index], "system") == 0) {
            expected_role = 1;
        }
        result = append_message(json, tokens, token_count, index, text_tokenizer, protocol,
                                expected_role, out_sequence);
        if (result != LM_DATASET_OK) {
            break;
        }
        last_was_assistant = expected_role == 2;
        expected_role = expected_role == 0 ? 1 : (expected_role == 1 ? 2 : 1);
        ++message_count;
        if (out_sequence->length > context_length + 1U) {
            result = LM_DATASET_INVALID_FORMAT;
            break;
        }
    }
    free(tokens);
    if (result == LM_DATASET_OK && (message_count < 2U || last_was_assistant == 0)) {
        result = LM_DATASET_INVALID_FORMAT;
    }
    return result;
}

static int read_line(FILE *file, unsigned char **buffer, size_t *capacity, size_t *out_length,
                     int *out_has_line) {
    *out_length = 0U;
    *out_has_line = 0;
    for (;;) {
        const int character = fgetc(file);
        if (character == EOF) {
            if (ferror(file) != 0) {
                return 0;
            }
            if (*out_length == 0U) {
                return 1;
            }
            break;
        }
        if (character == '\n') {
            break;
        }
        if (*out_length == *capacity) {
            size_t new_capacity = *capacity == 0U ? 1024U : *capacity * 2U;
            if (new_capacity < *capacity) {
                return -1;
            }
            unsigned char *resized = realloc(*buffer, new_capacity);
            if (resized == NULL) {
                return -1;
            }
            *buffer = resized;
            *capacity = new_capacity;
        }
        (*buffer)[(*out_length)++] = (unsigned char)character;
    }
    if (*out_length != 0U && (*buffer)[*out_length - 1U] == '\r') {
        --*out_length;
    }
    *out_has_line = 1;
    return 1;
}

lm_dataset_status lm_sft_dataset_prepare_jsonl(const char *tokenizer_path,
                                               const char *conversations_jsonl_path,
                                               const char *output_prefix, size_t context_length,
                                               lm_sft_prepare_report *out_report) {
    if (tokenizer_path == NULL || conversations_jsonl_path == NULL || output_prefix == NULL ||
        out_report == NULL || context_length == 0U || context_length > UINT32_MAX) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_report = (lm_sft_prepare_report){0};
    tokenizer *text_tokenizer = NULL;
    tokenizer_status tokenizer_result = tokenizer_load(tokenizer_path, &text_tokenizer);
    if (tokenizer_result != TOKENIZER_OK) {
        return tokenizer_result == TOKENIZER_ALLOCATION_FAILED ? LM_DATASET_ALLOCATION_FAILED
                                                               : LM_DATASET_TOKENIZER_ERROR;
    }
    const uint32_t tokenizer_vocab_size = tokenizer_vocabulary_size(text_tokenizer);
    if (tokenizer_vocab_size > UINT32_MAX - 8U) {
        tokenizer_destroy(text_tokenizer);
        return LM_DATASET_OVERFLOW;
    }
    const uint32_t model_vocabulary_size = tokenizer_vocab_size + 8U;
    lm_chat_protocol protocol = {0};
    lm_dataset_status status =
        lm_chat_protocol_v1(tokenizer_vocab_size, model_vocabulary_size, &protocol);
    unsigned char tokenizer_checksum[32] = {0};
    if (status == LM_DATASET_OK) {
        status = checksum_file(tokenizer_path, tokenizer_checksum);
    }
    FILE *input = NULL;
    if (status == LM_DATASET_OK) {
        input = fopen(conversations_jsonl_path, "rb");
        status = input == NULL ? LM_DATASET_IO_ERROR : LM_DATASET_OK;
    }
    sft_writer writers[LM_SFT_SPLIT_COUNT] = {0};
    sft_id_set ids = {0};
    for (size_t index = 0U; index < LM_SFT_SPLIT_COUNT && status == LM_DATASET_OK; ++index) {
        status = writer_open(&writers[index], output_prefix, (lm_dataset_split)index);
    }
    unsigned char *line = NULL;
    size_t line_capacity = 0U;
    while (status == LM_DATASET_OK) {
        size_t line_length = 0U;
        int has_line = 0;
        const int line_status = read_line(input, &line, &line_capacity, &line_length, &has_line);
        if (line_status != 1) {
            status = line_status < 0 ? LM_DATASET_ALLOCATION_FAILED : LM_DATASET_IO_ERROR;
            break;
        }
        if (has_line == 0) {
            break;
        }
        if (line_length == 0U) {
            continue;
        }
        sft_sequence sequence = {0};
        lm_dataset_split split = LM_DATASET_TRAIN;
        uint64_t id_hash = 0U;
        status = parse_conversation(line, line_length, text_tokenizer, &protocol, context_length,
                                    &split, &id_hash, &sequence);
        if (status == LM_DATASET_OK) {
            const int insert_status = id_set_insert(&ids, id_hash);
            if (insert_status <= 0) {
                status = insert_status == 0 ? LM_DATASET_DUPLICATE_DOCUMENT
                                            : LM_DATASET_ALLOCATION_FAILED;
            }
        }
        if (status == LM_DATASET_OK) {
            status =
                writer_append(&writers[split], &sequence, context_length, protocol.padding_token);
        }
        sequence_destroy(&sequence);
    }
    free(line);
    id_set_destroy(&ids);
    if (input != NULL && fclose(input) != 0 && status == LM_DATASET_OK) {
        status = LM_DATASET_IO_ERROR;
    }
    for (size_t index = 0U; index < LM_SFT_SPLIT_COUNT && status == LM_DATASET_OK; ++index) {
        status = writer_finish(&writers[index], tokenizer_vocab_size, model_vocabulary_size,
                               context_length, &protocol, tokenizer_checksum);
    }
    for (size_t index = 0U; index < LM_SFT_SPLIT_COUNT && status == LM_DATASET_OK; ++index) {
        if (rename(writers[index].temporary_path, writers[index].final_path) != 0) {
            status = LM_DATASET_IO_ERROR;
        } else {
            writers[index].published = 1;
        }
    }
    if (status == LM_DATASET_OK) {
        out_report->tokenizer_vocabulary_size = tokenizer_vocab_size;
        out_report->model_vocabulary_size = model_vocabulary_size;
        out_report->context_length = context_length;
        out_report->protocol = protocol;
        for (size_t index = 0U; index < LM_SFT_SPLIT_COUNT; ++index) {
            out_report->example_counts[index] = writers[index].example_count;
            out_report->supervised_token_counts[index] = writers[index].supervised_token_count;
            writers[index].published = 0;
        }
    }
    writers_abort(writers);
    tokenizer_destroy(text_tokenizer);
    return status;
}

static lm_dataset_status validate_payload(FILE *file, uint64_t payload_bytes,
                                          const unsigned char expected[32]) {
    tokenizer_sha256_context checksum = {0};
    tokenizer_sha256_init(&checksum);
    unsigned char buffer[64U * 1024U];
    uint64_t remaining = payload_bytes;
    while (remaining != 0U) {
        const size_t wanted = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        const size_t count = fread(buffer, 1U, wanted, file);
        if (count != wanted) {
            return LM_DATASET_IO_ERROR;
        }
        tokenizer_sha256_update(&checksum, buffer, count);
        remaining -= count;
    }
    unsigned char digest[32] = {0};
    tokenizer_sha256_final(&checksum, digest);
    return memcmp(digest, expected, sizeof(digest)) == 0 && fgetc(file) == EOF
               ? LM_DATASET_OK
               : LM_DATASET_INVALID_FORMAT;
}

lm_dataset_status lm_sft_dataset_open(const char *path, lm_sft_dataset **out_dataset) {
    if (path == NULL || out_dataset == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_dataset = NULL;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return LM_DATASET_IO_ERROR;
    }
    unsigned char header[LM_SFT_HEADER_SIZE] = {0};
    lm_dataset_status status = fread(header, 1U, sizeof(header), file) == sizeof(header)
                                   ? LM_DATASET_OK
                                   : LM_DATASET_IO_ERROR;
    const uint32_t tokenizer_vocabulary_size = load_u32(header + 16U);
    const uint32_t model_vocabulary_size = load_u32(header + 20U);
    const uint32_t context_length = load_u32(header + 24U);
    const uint32_t split = load_u32(header + 28U);
    const uint64_t example_count = load_u64(header + 32U);
    const uint64_t supervised_token_count = load_u64(header + 40U);
    const uint64_t payload_bytes = load_u64(header + 48U);
    lm_chat_protocol protocol = {.system_token = load_u32(header + 120U),
                                 .user_token = load_u32(header + 124U),
                                 .assistant_token = load_u32(header + 128U),
                                 .end_token = load_u32(header + 132U),
                                 .padding_token = load_u32(header + 136U)};
    lm_chat_protocol expected_protocol = {0};
    if (status == LM_DATASET_OK &&
        (memcmp(header, sft_magic, sizeof(sft_magic)) != 0 ||
         load_u32(header + 8U) != LM_SFT_FORMAT_VERSION ||
         load_u32(header + 12U) != LM_SFT_HEADER_SIZE || context_length == 0U ||
         split >= LM_SFT_SPLIT_COUNT || example_count == 0U || supervised_token_count == 0U ||
         lm_chat_protocol_v1(tokenizer_vocabulary_size, model_vocabulary_size,
                             &expected_protocol) != LM_DATASET_OK ||
         protocol.system_token != expected_protocol.system_token ||
         protocol.user_token != expected_protocol.user_token ||
         protocol.assistant_token != expected_protocol.assistant_token ||
         protocol.end_token != expected_protocol.end_token ||
         protocol.padding_token != expected_protocol.padding_token ||
         example_count >
             UINT64_MAX / ((uint64_t)context_length * UINT64_C(3) * LM_SFT_TOKEN_SIZE) ||
         payload_bytes !=
             example_count * (uint64_t)context_length * UINT64_C(3) * LM_SFT_TOKEN_SIZE)) {
        status = LM_DATASET_INVALID_FORMAT;
    }
    if (status == LM_DATASET_OK) {
        status = validate_payload(file, payload_bytes, header + 88U);
    }
    lm_sft_dataset *dataset = NULL;
    if (status == LM_DATASET_OK) {
        dataset = calloc(1U, sizeof(*dataset));
        status = dataset == NULL ? LM_DATASET_ALLOCATION_FAILED : LM_DATASET_OK;
    }
    if (status == LM_DATASET_OK && seek_file(file, LM_SFT_HEADER_SIZE) != 0) {
        status = LM_DATASET_IO_ERROR;
    }
    if (status == LM_DATASET_OK) {
        dataset->file = file;
        dataset->example_count = example_count;
        dataset->supervised_token_count = supervised_token_count;
        dataset->tokenizer_vocabulary_size = tokenizer_vocabulary_size;
        dataset->model_vocabulary_size = model_vocabulary_size;
        dataset->context_length = context_length;
        dataset->split = (lm_dataset_split)split;
        dataset->protocol = protocol;
        dataset->record_byte_count = (uint64_t)context_length * UINT64_C(3) * LM_SFT_TOKEN_SIZE;
        memcpy(dataset->tokenizer_checksum, header + 56U, 32U);
        *out_dataset = dataset;
        return LM_DATASET_OK;
    }
    free(dataset);
    (void)fclose(file);
    return status;
}

void lm_sft_dataset_close(lm_sft_dataset *dataset) {
    if (dataset != NULL) {
        if (dataset->file != NULL) {
            (void)fclose(dataset->file);
        }
        free(dataset);
    }
}

lm_dataset_split lm_sft_dataset_get_split(const lm_sft_dataset *dataset) {
    return dataset == NULL ? LM_DATASET_TRAIN : dataset->split;
}

uint64_t lm_sft_dataset_example_count(const lm_sft_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->example_count;
}

uint64_t lm_sft_dataset_supervised_token_count(const lm_sft_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->supervised_token_count;
}

uint32_t lm_sft_dataset_model_vocabulary_size(const lm_sft_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->model_vocabulary_size;
}

size_t lm_sft_dataset_context_length(const lm_sft_dataset *dataset) {
    return dataset == NULL ? 0U : dataset->context_length;
}

lm_chat_protocol lm_sft_dataset_protocol(const lm_sft_dataset *dataset) {
    return dataset == NULL ? (lm_chat_protocol){0} : dataset->protocol;
}

lm_dataset_status lm_sft_dataset_tokenizer_matches(const lm_sft_dataset *dataset,
                                                   const char *tokenizer_path, int *out_matches) {
    if (dataset == NULL || tokenizer_path == NULL || out_matches == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_matches = 0;
    unsigned char checksum[32] = {0};
    const lm_dataset_status status = checksum_file(tokenizer_path, checksum);
    if (status == LM_DATASET_OK) {
        *out_matches = memcmp(checksum, dataset->tokenizer_checksum, sizeof(checksum)) == 0;
    }
    return status;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12U;
    value ^= value << 25U;
    value ^= value >> 27U;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static uint64_t gcd(uint64_t left, uint64_t right) {
    while (right != 0U) {
        const uint64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

static void start_epoch(lm_sft_batcher *batcher) {
    const uint64_t count = batcher->dataset->example_count;
    batcher->sample_index = 0U;
    if (count == 1U) {
        batcher->next_offset = 0U;
        batcher->stride = 0U;
        return;
    }
    batcher->next_offset = next_random(&batcher->random_state) % count;
    do {
        batcher->stride = 1U + next_random(&batcher->random_state) % (count - 1U);
    } while (gcd(batcher->stride, count) != 1U);
}

lm_dataset_status lm_sft_batcher_create(lm_sft_dataset *dataset, size_t batch_size, uint64_t seed,
                                        lm_sft_batcher **out_batcher) {
    if (dataset == NULL || batch_size == 0U || out_batcher == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    lm_sft_batcher *batcher = calloc(1U, sizeof(*batcher));
    if (batcher == NULL) {
        return LM_DATASET_ALLOCATION_FAILED;
    }
    batcher->dataset = dataset;
    batcher->batch_size = batch_size;
    batcher->random_state = seed == 0U ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    start_epoch(batcher);
    *out_batcher = batcher;
    return LM_DATASET_OK;
}

void lm_sft_batcher_destroy(lm_sft_batcher *batcher) { free(batcher); }

static lm_dataset_status read_u32_array(FILE *file, uint32_t *output, size_t count) {
    unsigned char bytes[4] = {0};
    for (size_t index = 0U; index < count; ++index) {
        if (fread(bytes, 1U, sizeof(bytes), file) != sizeof(bytes)) {
            return LM_DATASET_IO_ERROR;
        }
        output[index] = load_u32(bytes);
    }
    return LM_DATASET_OK;
}

lm_dataset_status lm_sft_batcher_next(lm_sft_batcher *batcher, token_id *out_inputs,
                                      token_id *out_targets, uint32_t *out_loss_mask,
                                      size_t *out_active_target_count) {
    if (batcher == NULL || out_inputs == NULL || out_targets == NULL || out_loss_mask == NULL ||
        out_active_target_count == NULL ||
        batcher->batch_size > SIZE_MAX / batcher->dataset->context_length) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    const size_t context = batcher->dataset->context_length;
    size_t active_count = 0U;
    for (size_t row = 0U; row < batcher->batch_size; ++row) {
        if (batcher->sample_index == batcher->dataset->example_count) {
            ++batcher->epoch;
            start_epoch(batcher);
        }
        const uint64_t example = batcher->next_offset;
        if (example > (UINT64_MAX - LM_SFT_HEADER_SIZE) / batcher->dataset->record_byte_count) {
            return LM_DATASET_OVERFLOW;
        }
        const uint64_t offset = LM_SFT_HEADER_SIZE + example * batcher->dataset->record_byte_count;
        if (seek_file(batcher->dataset->file, offset) != 0) {
            return LM_DATASET_IO_ERROR;
        }
        lm_dataset_status status =
            read_u32_array(batcher->dataset->file, out_inputs + row * context, context);
        if (status == LM_DATASET_OK) {
            status = read_u32_array(batcher->dataset->file, out_targets + row * context, context);
        }
        if (status == LM_DATASET_OK) {
            status = read_u32_array(batcher->dataset->file, out_loss_mask + row * context, context);
        }
        if (status != LM_DATASET_OK) {
            return status;
        }
        for (size_t position = 0U; position < context; ++position) {
            const uint32_t mask = out_loss_mask[row * context + position];
            if (mask > 1U || (mask != 0U && active_count == SIZE_MAX)) {
                return LM_DATASET_INVALID_FORMAT;
            }
            active_count += mask;
        }
        ++batcher->sample_index;
        batcher->next_offset =
            (batcher->next_offset + batcher->stride) % batcher->dataset->example_count;
    }
    if (active_count == 0U) {
        return LM_DATASET_INVALID_FORMAT;
    }
    *out_active_target_count = active_count;
    return LM_DATASET_OK;
}

lm_dataset_status lm_sft_batcher_get_state(const lm_sft_batcher *batcher,
                                           lm_batcher_state *out_state) {
    if (batcher == NULL || out_state == NULL) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    *out_state = (lm_batcher_state){.random_state = batcher->random_state,
                                    .epoch = batcher->epoch,
                                    .sample_index = batcher->sample_index,
                                    .next_offset = batcher->next_offset,
                                    .stride = batcher->stride};
    return LM_DATASET_OK;
}

lm_dataset_status lm_sft_batcher_set_state(lm_sft_batcher *batcher, const lm_batcher_state *state) {
    if (batcher == NULL || state == NULL || state->random_state == 0U ||
        state->sample_index > batcher->dataset->example_count ||
        state->next_offset >= batcher->dataset->example_count ||
        (batcher->dataset->example_count > 1U &&
         (state->stride == 0U || state->stride >= batcher->dataset->example_count ||
          gcd(state->stride, batcher->dataset->example_count) != 1U))) {
        return LM_DATASET_INVALID_ARGUMENT;
    }
    batcher->random_state = state->random_state;
    batcher->epoch = state->epoch;
    batcher->sample_index = state->sample_index;
    batcher->next_offset = state->next_offset;
    batcher->stride = state->stride;
    return LM_DATASET_OK;
}
