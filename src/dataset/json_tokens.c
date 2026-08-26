#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "json_tokens.h"

#define LM_JSON_NO_PARENT SIZE_MAX
#define LM_JSON_MAX_DEPTH 64U

typedef struct lm_json_parser {
    size_t position;
    size_t next_token;
    size_t parent;
    size_t depth;
} lm_json_parser;

static int allocate_token(lm_json_parser *parser, lm_json_token *tokens, size_t capacity,
                          lm_json_token_type type, size_t start, size_t *out_index) {
    if (parser->next_token == capacity) {
        return -1;
    }
    const size_t index = parser->next_token++;
    tokens[index] = (lm_json_token){
        .type = type, .start = start, .end = 0U, .child_count = 0U, .parent = parser->parent};
    if (parser->parent != LM_JSON_NO_PARENT) {
        ++tokens[parser->parent].child_count;
    }
    *out_index = index;
    return 1;
}

static int parse_string(const unsigned char *json, size_t length, lm_json_parser *parser,
                        lm_json_token *tokens, size_t capacity) {
    const size_t start = ++parser->position;
    while (parser->position < length) {
        const unsigned char byte = json[parser->position];
        if (byte == '"') {
            size_t index = 0U;
            const int status = allocate_token(parser, tokens, capacity, LM_JSON_STRING, start,
                                              &index);
            if (status != 1) {
                return status;
            }
            tokens[index].end = parser->position++;
            return 1;
        }
        if (byte < 0x20U) {
            return 0;
        }
        if (byte == '\\') {
            ++parser->position;
            if (parser->position >= length) {
                return 0;
            }
            const unsigned char escaped = json[parser->position];
            if (escaped == 'u') {
                if (length - parser->position <= 4U) {
                    return 0;
                }
                for (size_t index = 1U; index <= 4U; ++index) {
                    const unsigned char digit = json[parser->position + index];
                    if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f') ||
                          (digit >= 'A' && digit <= 'F'))) {
                        return 0;
                    }
                }
                parser->position += 4U;
            } else if (strchr("\"\\/bfnrt", (int)escaped) == NULL) {
                return 0;
            }
        }
        ++parser->position;
    }
    return 0;
}

static int primitive_delimiter(unsigned char byte) {
    return byte == ',' || byte == ']' || byte == '}' || byte == ' ' || byte == '\t' ||
           byte == '\r' || byte == '\n';
}

static int parse_primitive(const unsigned char *json, size_t length, lm_json_parser *parser,
                           lm_json_token *tokens, size_t capacity) {
    const size_t start = parser->position;
    while (parser->position < length && primitive_delimiter(json[parser->position]) == 0) {
        const unsigned char byte = json[parser->position];
        if (byte < 0x20U || byte == ':' || byte == '"' || byte == '[' || byte == '{') {
            return 0;
        }
        ++parser->position;
    }
    if (parser->position == start) {
        return 0;
    }
    size_t index = 0U;
    const int status =
        allocate_token(parser, tokens, capacity, LM_JSON_PRIMITIVE, start, &index);
    if (status != 1) {
        return status;
    }
    tokens[index].end = parser->position;
    return 1;
}

int lm_json_tokenize(const unsigned char *json, size_t length, lm_json_token *tokens,
                     size_t capacity, size_t *out_count) {
    if (json == NULL || tokens == NULL || capacity == 0U || out_count == NULL) {
        return 0;
    }
    lm_json_parser parser = {.position = 0U, .next_token = 0U, .parent = LM_JSON_NO_PARENT};
    for (;;) {
        while (parser.position < length &&
               (json[parser.position] == ' ' || json[parser.position] == '\t' ||
                json[parser.position] == '\r' || json[parser.position] == '\n')) {
            ++parser.position;
        }
        if (parser.position == length) {
            break;
        }
        const unsigned char byte = json[parser.position];
        if (byte == '{' || byte == '[') {
            if (parser.depth == LM_JSON_MAX_DEPTH) {
                return 0;
            }
            size_t index = 0U;
            const int status = allocate_token(&parser, tokens, capacity,
                                              byte == '{' ? LM_JSON_OBJECT : LM_JSON_ARRAY,
                                              parser.position, &index);
            if (status != 1) {
                return status;
            }
            parser.parent = index;
            ++parser.depth;
            ++parser.position;
            continue;
        }
        if (byte == '}' || byte == ']') {
            if (parser.parent == LM_JSON_NO_PARENT ||
                (byte == '}' && tokens[parser.parent].type != LM_JSON_OBJECT) ||
                (byte == ']' && tokens[parser.parent].type != LM_JSON_ARRAY)) {
                return 0;
            }
            tokens[parser.parent].end = ++parser.position;
            parser.parent = tokens[parser.parent].parent;
            --parser.depth;
            continue;
        }
        if (byte == '"') {
            const int status = parse_string(json, length, &parser, tokens, capacity);
            if (status != 1) {
                return status;
            }
            continue;
        }
        if (byte == ':' || byte == ',') {
            ++parser.position;
            continue;
        }
        const int status = parse_primitive(json, length, &parser, tokens, capacity);
        if (status != 1) {
            return status;
        }
    }
    if (parser.parent != LM_JSON_NO_PARENT || parser.next_token == 0U ||
        tokens[0].parent != LM_JSON_NO_PARENT || tokens[0].end == 0U) {
        return 0;
    }
    *out_count = parser.next_token;
    return 1;
}

int lm_json_token_equals(const unsigned char *json, const lm_json_token *token,
                         const char *literal) {
    if (json == NULL || token == NULL || literal == NULL || token->type != LM_JSON_STRING) {
        return 0;
    }
    const size_t length = strlen(literal);
    return token->end - token->start == length &&
           memcmp(json + token->start, literal, length) == 0;
}

static int hex_value(unsigned char byte, uint32_t *out_value) {
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

static int append_utf8(unsigned char *output, size_t capacity, size_t *length, uint32_t codepoint) {
    unsigned char bytes[4] = {0};
    size_t count = 0U;
    if (codepoint <= UINT32_C(0x7f)) {
        bytes[count++] = (unsigned char)codepoint;
    } else if (codepoint <= UINT32_C(0x7ff)) {
        bytes[count++] = (unsigned char)(0xc0U | (codepoint >> 6U));
        bytes[count++] = (unsigned char)(0x80U | (codepoint & 0x3fU));
    } else if (codepoint <= UINT32_C(0xffff)) {
        bytes[count++] = (unsigned char)(0xe0U | (codepoint >> 12U));
        bytes[count++] = (unsigned char)(0x80U | ((codepoint >> 6U) & 0x3fU));
        bytes[count++] = (unsigned char)(0x80U | (codepoint & 0x3fU));
    } else if (codepoint <= UINT32_C(0x10ffff)) {
        bytes[count++] = (unsigned char)(0xf0U | (codepoint >> 18U));
        bytes[count++] = (unsigned char)(0x80U | ((codepoint >> 12U) & 0x3fU));
        bytes[count++] = (unsigned char)(0x80U | ((codepoint >> 6U) & 0x3fU));
        bytes[count++] = (unsigned char)(0x80U | (codepoint & 0x3fU));
    } else {
        return 0;
    }
    if (count > capacity - *length) {
        return -1;
    }
    memcpy(output + *length, bytes, count);
    *length += count;
    return 1;
}

int lm_json_decode_string(const unsigned char *json, const lm_json_token *token,
                          unsigned char **out_bytes, size_t *out_length) {
    if (json == NULL || token == NULL || token->type != LM_JSON_STRING || out_bytes == NULL ||
        out_length == NULL || token->end < token->start) {
        return 0;
    }
    const size_t capacity = token->end - token->start;
    unsigned char *output = malloc(capacity == 0U ? 1U : capacity);
    if (output == NULL) {
        return -1;
    }
    size_t length = 0U;
    for (size_t index = token->start; index < token->end; ++index) {
        unsigned char byte = json[index];
        if (byte != '\\') {
            output[length++] = byte;
            continue;
        }
        byte = json[++index];
        if (byte == 'u') {
            uint32_t codepoint = 0U;
            for (size_t digit_index = 0U; digit_index < 4U; ++digit_index) {
                uint32_t digit = 0U;
                if (hex_value(json[++index], &digit) == 0) {
                    free(output);
                    return 0;
                }
                codepoint = codepoint * UINT32_C(16) + digit;
            }
            if (codepoint >= UINT32_C(0xd800) && codepoint <= UINT32_C(0xdbff)) {
                if (token->end - index <= 6U || json[index + 1U] != '\\' ||
                    json[index + 2U] != 'u') {
                    free(output);
                    return 0;
                }
                index += 2U;
                uint32_t low = 0U;
                for (size_t digit_index = 0U; digit_index < 4U; ++digit_index) {
                    uint32_t digit = 0U;
                    if (hex_value(json[++index], &digit) == 0) {
                        free(output);
                        return 0;
                    }
                    low = low * UINT32_C(16) + digit;
                }
                if (low < UINT32_C(0xdc00) || low > UINT32_C(0xdfff)) {
                    free(output);
                    return 0;
                }
                codepoint = UINT32_C(0x10000) + ((codepoint - UINT32_C(0xd800)) << 10U) +
                            low - UINT32_C(0xdc00);
            } else if (codepoint >= UINT32_C(0xdc00) && codepoint <= UINT32_C(0xdfff)) {
                free(output);
                return 0;
            }
            const int status = append_utf8(output, capacity, &length, codepoint);
            if (status != 1) {
                free(output);
                return status;
            }
            continue;
        }
        switch (byte) {
        case '"':
        case '\\':
        case '/':
            output[length++] = byte;
            break;
        case 'b': output[length++] = '\b'; break;
        case 'f': output[length++] = '\f'; break;
        case 'n': output[length++] = '\n'; break;
        case 'r': output[length++] = '\r'; break;
        case 't': output[length++] = '\t'; break;
        default:
            free(output);
            return 0;
        }
    }
    *out_bytes = output;
    *out_length = length;
    return 1;
}
