#ifndef LLM_LAB_TOKENIZER_SHA256_H
#define LLM_LAB_TOKENIZER_SHA256_H

#include <stddef.h>
#include <stdint.h>

#define TOKENIZER_SHA256_DIGEST_SIZE 32U

/** Incremental SHA-256 used to validate binary artifacts. */
typedef struct tokenizer_sha256_context {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_length;
} tokenizer_sha256_context;

void tokenizer_sha256_init(tokenizer_sha256_context *context);
void tokenizer_sha256_update(tokenizer_sha256_context *context, const unsigned char *data,
                             size_t length);
void tokenizer_sha256_final(tokenizer_sha256_context *context,
                            unsigned char digest[TOKENIZER_SHA256_DIGEST_SIZE]);

#endif
