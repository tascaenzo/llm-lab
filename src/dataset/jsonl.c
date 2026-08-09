#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset_internal.h"

#define JSON_MAX_DEPTH 64U

typedef struct byte_string {
    unsigned char *data;
    size_t length;
    size_t capacity;
} byte_string;

typedef struct document_record {
    byte_string id;
    byte_string text;
    int has_id;
    int has_text;
} document_record;

typedef struct id_set {
    uint64_t *hashes;
    unsigned char *occupied;
    size_t count;
    size_t capacity;
} id_set;

static void byte_string_destroy(byte_string *string) {
    free(string->data);
    *string = (byte_string){0};
}

static int byte_string_reserve(byte_string *string, size_t required) {
    if (required <= string->capacity) {
        return 1;
    }
    size_t capacity = string->capacity == 0U ? 64U : string->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2U) {
            return 0;
        }
        capacity *= 2U;
    }
    unsigned char *data = realloc(string->data, capacity);
    if (data == NULL) {
        return 0;
    }
    string->data = data;
    string->capacity = capacity;
    return 1;
}

static int byte_string_push(byte_string *string, unsigned char byte) {
    if (string->length == SIZE_MAX || byte_string_reserve(string, string->length + 1U) == 0) {
        return 0;
    }
    string->data[string->length++] = byte;
    return 1;
}

static int append_utf8(byte_string *string, uint32_t codepoint) {
    if (codepoint <= UINT32_C(0x7f)) {
        return byte_string_push(string, (unsigned char)codepoint);
    }
    if (codepoint <= UINT32_C(0x7ff)) {
        return byte_string_push(string, (unsigned char)(0xc0U | (codepoint >> 6U))) != 0 &&
               byte_string_push(string, (unsigned char)(0x80U | (codepoint & 0x3fU))) != 0;
    }
    if (codepoint <= UINT32_C(0xffff)) {
        return byte_string_push(string, (unsigned char)(0xe0U | (codepoint >> 12U))) != 0 &&
               byte_string_push(string, (unsigned char)(0x80U | ((codepoint >> 6U) & 0x3fU))) !=
                   0 &&
               byte_string_push(string, (unsigned char)(0x80U | (codepoint & 0x3fU))) != 0;
    }
    if (codepoint <= UINT32_C(0x10ffff)) {
        return byte_string_push(string, (unsigned char)(0xf0U | (codepoint >> 18U))) != 0 &&
               byte_string_push(string, (unsigned char)(0x80U | ((codepoint >> 12U) & 0x3fU))) !=
                   0 &&
               byte_string_push(string, (unsigned char)(0x80U | ((codepoint >> 6U) & 0x3fU))) !=
                   0 &&
               byte_string_push(string, (unsigned char)(0x80U | (codepoint & 0x3fU))) != 0;
    }
    return 0;
}

static int hex_digit(unsigned char byte, uint32_t *out_value) {
    if (byte >= '0' && byte <= '9') {
        *out_value = byte - '0';
        return 1;
    }
    if (byte >= 'a' && byte <= 'f') {
        *out_value = UINT32_C(10) + byte - 'a';
        return 1;
    }
    if (byte >= 'A' && byte <= 'F') {
        *out_value = UINT32_C(10) + byte - 'A';
        return 1;
    }
    return 0;
}

static int parse_hex_quad(const unsigned char **cursor, uint32_t *out_value) {
    uint32_t value = 0U;
    for (size_t index = 0U; index < 4U; ++index) {
        uint32_t digit = 0U;
        if (**cursor == '\0' || hex_digit(**cursor, &digit) == 0) {
            return 0;
        }
        value = value * UINT32_C(16) + digit;
        ++*cursor;
    }
    *out_value = value;
    return 1;
}

static int parse_json_string(const unsigned char **cursor, byte_string *out_string) {
    if (**cursor != '"') {
        return 0;
    }
    ++*cursor;
    while (**cursor != '"') {
        const unsigned char byte = **cursor;
        if (byte == '\0' || byte < 0x20U) {
            return 0;
        }
        ++*cursor;
        if (byte != '\\') {
            if (byte_string_push(out_string, byte) == 0) {
                return -1;
            }
            continue;
        }

        const unsigned char escaped = **cursor;
        if (escaped == '\0') {
            return 0;
        }
        ++*cursor;
        unsigned char decoded = 0U;
        switch (escaped) {
        case '"':
        case '\\':
        case '/':
            decoded = escaped;
            break;
        case 'b':
            decoded = '\b';
            break;
        case 'f':
            decoded = '\f';
            break;
        case 'n':
            decoded = '\n';
            break;
        case 'r':
            decoded = '\r';
            break;
        case 't':
            decoded = '\t';
            break;
        case 'u': {
            uint32_t codepoint = 0U;
            if (parse_hex_quad(cursor, &codepoint) == 0) {
                return 0;
            }
            if (codepoint >= UINT32_C(0xd800) && codepoint <= UINT32_C(0xdbff)) {
                if ((*cursor)[0] != '\\' || (*cursor)[1] != 'u') {
                    return 0;
                }
                *cursor += 2;
                uint32_t low = 0U;
                if (parse_hex_quad(cursor, &low) == 0 || low < UINT32_C(0xdc00) ||
                    low > UINT32_C(0xdfff)) {
                    return 0;
                }
                codepoint = UINT32_C(0x10000) + ((codepoint - UINT32_C(0xd800)) << 10U) +
                            (low - UINT32_C(0xdc00));
            } else if (codepoint >= UINT32_C(0xdc00) && codepoint <= UINT32_C(0xdfff)) {
                return 0;
            }
            if (append_utf8(out_string, codepoint) == 0) {
                return -1;
            }
            continue;
        }
        default:
            return 0;
        }
        if (byte_string_push(out_string, decoded) == 0) {
            return -1;
        }
    }
    ++*cursor;
    return 1;
}

static void skip_whitespace(const unsigned char **cursor) {
    while (**cursor == ' ' || **cursor == '\t' || **cursor == '\r' || **cursor == '\n') {
        ++*cursor;
    }
}

static int skip_json_value(const unsigned char **cursor, unsigned int depth);

static int starts_with(const unsigned char *cursor, const char *literal) {
    for (size_t index = 0U; literal[index] != '\0'; ++index) {
        if (cursor[index] == '\0' || cursor[index] != (unsigned char)literal[index]) {
            return 0;
        }
    }
    return 1;
}

static int skip_json_array(const unsigned char **cursor, unsigned int depth) {
    ++*cursor;
    skip_whitespace(cursor);
    if (**cursor == ']') {
        ++*cursor;
        return 1;
    }
    for (;;) {
        const int status = skip_json_value(cursor, depth + 1U);
        if (status != 1) {
            return status;
        }
        skip_whitespace(cursor);
        if (**cursor == ']') {
            ++*cursor;
            return 1;
        }
        if (**cursor != ',') {
            return 0;
        }
        ++*cursor;
        skip_whitespace(cursor);
    }
}

static int skip_json_object(const unsigned char **cursor, unsigned int depth) {
    ++*cursor;
    skip_whitespace(cursor);
    if (**cursor == '}') {
        ++*cursor;
        return 1;
    }
    for (;;) {
        byte_string key = {0};
        const int key_status = parse_json_string(cursor, &key);
        byte_string_destroy(&key);
        if (key_status != 1) {
            return key_status;
        }
        skip_whitespace(cursor);
        if (**cursor != ':') {
            return 0;
        }
        ++*cursor;
        skip_whitespace(cursor);
        const int value_status = skip_json_value(cursor, depth + 1U);
        if (value_status != 1) {
            return value_status;
        }
        skip_whitespace(cursor);
        if (**cursor == '}') {
            ++*cursor;
            return 1;
        }
        if (**cursor != ',') {
            return 0;
        }
        ++*cursor;
        skip_whitespace(cursor);
    }
}

static int skip_json_value(const unsigned char **cursor, unsigned int depth) {
    if (depth > JSON_MAX_DEPTH) {
        return 0;
    }
    if (**cursor == '"') {
        byte_string value = {0};
        const int status = parse_json_string(cursor, &value);
        byte_string_destroy(&value);
        return status;
    }
    if (**cursor == '{') {
        return skip_json_object(cursor, depth);
    }
    if (**cursor == '[') {
        return skip_json_array(cursor, depth);
    }

    static const char *const literals[] = {"true", "false", "null"};
    for (size_t index = 0U; index < sizeof(literals) / sizeof(literals[0]); ++index) {
        const size_t length = strlen(literals[index]);
        if (starts_with(*cursor, literals[index]) != 0) {
            *cursor += length;
            return 1;
        }
    }

    const unsigned char *start = *cursor;
    if (**cursor == '-') {
        ++*cursor;
    }
    if (**cursor == '0') {
        ++*cursor;
    } else if (**cursor >= '1' && **cursor <= '9') {
        while (isdigit(**cursor) != 0) {
            ++*cursor;
        }
    } else {
        return 0;
    }
    if (**cursor == '.') {
        ++*cursor;
        if (isdigit(**cursor) == 0) {
            return 0;
        }
        while (isdigit(**cursor) != 0) {
            ++*cursor;
        }
    }
    if (**cursor == 'e' || **cursor == 'E') {
        ++*cursor;
        if (**cursor == '+' || **cursor == '-') {
            ++*cursor;
        }
        if (isdigit(**cursor) == 0) {
            return 0;
        }
        while (isdigit(**cursor) != 0) {
            ++*cursor;
        }
    }
    return *cursor != start;
}

static int key_equals(const byte_string *key, const char *expected) {
    const size_t length = strlen(expected);
    return key->length == length && memcmp(key->data, expected, length) == 0;
}

static int parse_document(const unsigned char *line, document_record *record) {
    const unsigned char *cursor = line;
    skip_whitespace(&cursor);
    if (*cursor != '{') {
        return 0;
    }
    ++cursor;
    skip_whitespace(&cursor);
    if (*cursor == '}') {
        return 0;
    }

    for (;;) {
        byte_string key = {0};
        int status = parse_json_string(&cursor, &key);
        if (status != 1) {
            byte_string_destroy(&key);
            return status;
        }
        skip_whitespace(&cursor);
        if (*cursor != ':') {
            byte_string_destroy(&key);
            return 0;
        }
        ++cursor;
        skip_whitespace(&cursor);

        if (key_equals(&key, "id") != 0 || key_equals(&key, "text") != 0) {
            const int is_id = key_equals(&key, "id");
            if ((is_id != 0 && record->has_id != 0) || (is_id == 0 && record->has_text != 0)) {
                byte_string_destroy(&key);
                return 0;
            }
            byte_string *destination = is_id != 0 ? &record->id : &record->text;
            status = parse_json_string(&cursor, destination);
            if (status != 1) {
                byte_string_destroy(&key);
                return status;
            }
            if (is_id != 0) {
                record->has_id = 1;
            } else {
                record->has_text = 1;
            }
        } else {
            status = skip_json_value(&cursor, 0U);
            if (status != 1) {
                byte_string_destroy(&key);
                return status;
            }
        }
        byte_string_destroy(&key);
        skip_whitespace(&cursor);
        if (*cursor == '}') {
            ++cursor;
            break;
        }
        if (*cursor != ',') {
            return 0;
        }
        ++cursor;
        skip_whitespace(&cursor);
    }
    skip_whitespace(&cursor);
    return *cursor == '\0' && record->has_id != 0 && record->has_text != 0 &&
           record->id.length != 0U;
}

static int read_jsonl_line(FILE *file, byte_string *line, int *out_has_line,
                           uint64_t *out_bytes_read) {
    line->length = 0U;
    *out_has_line = 0;
    *out_bytes_read = 0U;
    for (;;) {
        const int character = fgetc(file);
        if (character == EOF) {
            if (ferror(file) != 0) {
                return 0;
            }
            if (line->length == 0U) {
                return 1;
            }
            break;
        }
        if (*out_bytes_read == UINT64_MAX) {
            return -2;
        }
        ++*out_bytes_read;
        if (character == '\n') {
            break;
        }
        if (byte_string_push(line, (unsigned char)character) == 0) {
            return -1;
        }
    }
    if (line->length != 0U && line->data[line->length - 1U] == '\r') {
        --line->length;
    }
    if (byte_string_push(line, '\0') == 0) {
        return -1;
    }
    --line->length;
    *out_has_line = 1;
    return 1;
}

static void notify_progress(const lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT],
                            uint64_t bytes_read, uint64_t total_bytes,
                            lm_dataset_progress_callback progress_callback,
                            void *progress_context) {
    if (progress_callback == NULL) {
        return;
    }
    uint64_t documents_processed = 0U;
    uint64_t token_counts[LM_DATASET_SPLIT_COUNT] = {0};
    for (size_t index = 0U; index < LM_DATASET_SPLIT_COUNT; ++index) {
        documents_processed += writers[index].document_count;
        token_counts[index] = writers[index].token_count;
    }
    progress_callback(bytes_read, total_bytes, documents_processed, token_counts, progress_context);
}

static uint64_t fnv1a_64(const unsigned char *data, size_t length) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < length; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static lm_dataset_split split_from_hash(uint64_t hash) {
    const uint64_t bucket = hash % UINT64_C(10000);
    if (bucket < UINT64_C(9000)) {
        return LM_DATASET_TRAIN;
    }
    return bucket < UINT64_C(9500) ? LM_DATASET_VALIDATION : LM_DATASET_TEST;
}

static void id_set_destroy(id_set *set) {
    free(set->hashes);
    free(set->occupied);
    *set = (id_set){0};
}

static int id_set_resize(id_set *set, size_t capacity) {
    uint64_t *hashes = calloc(capacity, sizeof(*hashes));
    unsigned char *occupied = calloc(capacity, sizeof(*occupied));
    if (hashes == NULL || occupied == NULL) {
        free(hashes);
        free(occupied);
        return 0;
    }
    for (size_t index = 0U; index < set->capacity; ++index) {
        if (set->occupied[index] == 0U) {
            continue;
        }
        size_t slot = (size_t)(set->hashes[index] & (capacity - 1U));
        while (occupied[slot] != 0U) {
            slot = (slot + 1U) & (capacity - 1U);
        }
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

static int id_set_insert(id_set *set, uint64_t hash) {
    if (set->capacity == 0U && id_set_resize(set, 16U) == 0) {
        return -1;
    }
    if (set->count >= set->capacity - set->capacity / 4U) {
        if (set->capacity > SIZE_MAX / 2U || id_set_resize(set, set->capacity * 2U) == 0) {
            return -1;
        }
    }
    size_t slot = (size_t)(hash & (set->capacity - 1U));
    while (set->occupied[slot] != 0U) {
        if (set->hashes[slot] == hash) {
            return 0;
        }
        slot = (slot + 1U) & (set->capacity - 1U);
    }
    set->hashes[slot] = hash;
    set->occupied[slot] = 1U;
    ++set->count;
    return 1;
}

lm_dataset_status lm_dataset_parse_documents(FILE *input, const tokenizer *tokenizer,
                                             lm_dataset_writer writers[LM_DATASET_SPLIT_COUNT],
                                             uint64_t total_bytes,
                                             lm_dataset_progress_callback progress_callback,
                                             void *progress_context) {
    byte_string line = {0};
    id_set document_ids = {0};
    lm_dataset_status result = LM_DATASET_OK;
    uint64_t bytes_read = 0U;
    uint64_t last_notified_documents = 0U;
    notify_progress(writers, bytes_read, total_bytes, progress_callback, progress_context);

    for (;;) {
        int has_line = 0;
        uint64_t line_bytes = 0U;
        const int read_status = read_jsonl_line(input, &line, &has_line, &line_bytes);
        if (read_status == 0) {
            result = LM_DATASET_IO_ERROR;
            break;
        }
        if (read_status < 0) {
            result = read_status == -2 ? LM_DATASET_OVERFLOW : LM_DATASET_ALLOCATION_FAILED;
            break;
        }
        if (line_bytes > UINT64_MAX - bytes_read) {
            result = LM_DATASET_OVERFLOW;
            break;
        }
        bytes_read += line_bytes;
        if (has_line == 0) {
            break;
        }
        const unsigned char *cursor = line.data;
        skip_whitespace(&cursor);
        if (*cursor == '\0') {
            continue;
        }

        document_record record = {0};
        const int parse_status = parse_document(line.data, &record);
        if (parse_status != 1) {
            byte_string_destroy(&record.id);
            byte_string_destroy(&record.text);
            result = parse_status < 0 ? LM_DATASET_ALLOCATION_FAILED : LM_DATASET_INVALID_JSONL;
            break;
        }
        const uint64_t document_hash = fnv1a_64(record.id.data, record.id.length);
        const int insert_status = id_set_insert(&document_ids, document_hash);
        if (insert_status <= 0) {
            byte_string_destroy(&record.id);
            byte_string_destroy(&record.text);
            result =
                insert_status == 0 ? LM_DATASET_DUPLICATE_DOCUMENT : LM_DATASET_ALLOCATION_FAILED;
            break;
        }

        token_sequence tokens = {0};
        const tokenizer_status tokenizer_result =
            tokenizer_encode(tokenizer, record.text.data, record.text.length, &tokens);
        if (tokenizer_result != TOKENIZER_OK) {
            byte_string_destroy(&record.id);
            byte_string_destroy(&record.text);
            result = tokenizer_result == TOKENIZER_ALLOCATION_FAILED
                         ? LM_DATASET_ALLOCATION_FAILED
                         : (tokenizer_result == TOKENIZER_OVERFLOW ? LM_DATASET_OVERFLOW
                                                                   : LM_DATASET_TOKENIZER_ERROR);
            break;
        }
        const lm_dataset_split split = split_from_hash(document_hash);
        result = lm_dataset_writer_append(&writers[split], tokens.ids, tokens.length,
                                          tokenizer_vocabulary_size(tokenizer));
        token_sequence_destroy(&tokens);
        byte_string_destroy(&record.id);
        byte_string_destroy(&record.text);
        if (result != LM_DATASET_OK) {
            break;
        }
        const uint64_t documents_processed = document_ids.count;
        if (documents_processed - last_notified_documents >= UINT64_C(1024)) {
            notify_progress(writers, bytes_read, total_bytes, progress_callback, progress_context);
            last_notified_documents = documents_processed;
        }
    }

    notify_progress(writers, bytes_read, total_bytes, progress_callback, progress_context);
    byte_string_destroy(&line);
    id_set_destroy(&document_ids);
    return result;
}
