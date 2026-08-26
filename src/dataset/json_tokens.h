#ifndef LLM_LAB_DATASET_JSON_TOKENS_H
#define LLM_LAB_DATASET_JSON_TOKENS_H

#include <stddef.h>

typedef enum lm_json_token_type {
    LM_JSON_OBJECT = 1,
    LM_JSON_ARRAY,
    LM_JSON_STRING,
    LM_JSON_PRIMITIVE
} lm_json_token_type;

typedef struct lm_json_token {
    lm_json_token_type type;
    size_t start;
    size_t end;
    size_t child_count;
    size_t parent;
} lm_json_token;

/** Tokenizes one complete UTF-8 JSON value. Returns 1, 0 for syntax errors, -1 for capacity. */
int lm_json_tokenize(const unsigned char *json, size_t length, lm_json_token *tokens,
                     size_t capacity, size_t *out_count);

int lm_json_token_equals(const unsigned char *json, const lm_json_token *token,
                         const char *literal);

/** Decodes a JSON string token to newly allocated UTF-8 bytes. */
int lm_json_decode_string(const unsigned char *json, const lm_json_token *token,
                          unsigned char **out_bytes, size_t *out_length);

#endif
