#ifndef TOKENIZER_TOKENIZER_H
#define TOKENIZER_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define TOKENIZER_BYTE_VOCABULARY_SIZE UINT32_C(256)

typedef uint32_t token_id;

typedef enum tokenizer_status {
    TOKENIZER_OK = 0,
    TOKENIZER_INVALID_ARGUMENT,
    TOKENIZER_ALLOCATION_FAILED,
    TOKENIZER_OVERFLOW,
    TOKENIZER_INVALID_TOKEN,
    TOKENIZER_IO_ERROR,
    TOKENIZER_INVALID_MODEL
} tokenizer_status;

typedef struct tokenizer tokenizer;

typedef struct token_sequence {
    token_id *ids;
    size_t length;
} token_sequence;

typedef enum tokenizer_train_phase {
    TOKENIZER_TRAIN_READING_INPUT,
    TOKENIZER_TRAIN_COLLECTING_PAIRS,
    TOKENIZER_TRAIN_BUILDING_HEAP,
    TOKENIZER_TRAIN_MERGING
} tokenizer_train_phase;

typedef void (*tokenizer_train_progress_callback)(tokenizer_train_phase phase, uint64_t completed,
                                                  uint64_t total, void *context);

/** Creates a tokenizer containing exactly the 256 byte tokens. */
tokenizer_status tokenizer_create_byte_level(tokenizer **out_tokenizer);

/** Destroys a tokenizer created by tokenizer_create_byte_level. */
void tokenizer_destroy(tokenizer *tokenizer);

/** Trains a Byte-level BPE tokenizer from one or more corpus files. */
tokenizer_status tokenizer_train(const char *const *input_paths, size_t input_count,
                                 uint32_t target_vocabulary_size, tokenizer **out_tokenizer);

/** Like tokenizer_train, with optional progress notifications for a user interface. */
tokenizer_status tokenizer_train_with_progress(const char *const *input_paths, size_t input_count,
                                               uint32_t target_vocabulary_size,
                                               tokenizer_train_progress_callback progress_callback,
                                               void *progress_context, tokenizer **out_tokenizer);

/** Saves a tokenizer in the portable binary .llmtok format. */
tokenizer_status tokenizer_save(const tokenizer *tokenizer, const char *path);

/** Loads a tokenizer saved by tokenizer_save. */
tokenizer_status tokenizer_load(const char *path, tokenizer **out_tokenizer);

/** Returns the total number of tokens in the model vocabulary. */
uint32_t tokenizer_vocabulary_size(const tokenizer *tokenizer);

/** Returns the number of BPE merge rules in the model. */
size_t tokenizer_merge_count(const tokenizer *tokenizer);

/** Encodes input bytes as token IDs. */
tokenizer_status tokenizer_encode(const tokenizer *tokenizer, const unsigned char *input,
                                  size_t input_length, token_sequence *out_tokens);

/** Decodes token IDs to raw bytes. The result is not NUL-terminated. */
tokenizer_status tokenizer_decode(const tokenizer *tokenizer, const token_sequence *tokens,
                                  unsigned char **out_bytes, size_t *out_length);

/** Releases the ID buffer stored in a token sequence. */
void token_sequence_destroy(token_sequence *tokens);

/** Releases bytes returned by tokenizer_decode. */
void tokenizer_bytes_destroy(unsigned char *bytes);

/** Returns a stable human-readable description of a tokenizer status. */
const char *tokenizer_status_string(tokenizer_status status);

#endif
