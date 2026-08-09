#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_support.h"
#include "tokenizer_internal.h"

typedef struct expected_pretokens {
    const unsigned char *const *values;
    const size_t *lengths;
    size_t count;
    size_t index;
} expected_pretokens;

typedef struct file_observation {
    size_t calls;
    size_t length;
    uint64_t bytes_read;
} file_observation;

static tokenizer_status expect_pretoken(const unsigned char *bytes, size_t length, void *context) {
    expected_pretokens *expected = context;
    if (expected->index >= expected->count || length != expected->lengths[expected->index] ||
        memcmp(bytes, expected->values[expected->index], length) != 0) {
        return TOKENIZER_INVALID_ARGUMENT;
    }
    ++expected->index;
    return TOKENIZER_OK;
}

static tokenizer_status observe_file_pretoken(const unsigned char *bytes, size_t length,
                                              void *context) {
    file_observation *observation = context;
    for (size_t index = 0U; index < length; ++index) {
        if (bytes[index] != 'a') {
            return TOKENIZER_INVALID_ARGUMENT;
        }
    }
    ++observation->calls;
    observation->length += length;
    return TOKENIZER_OK;
}

static void observe_progress(uint64_t bytes_read, void *context) {
    file_observation *observation = context;
    observation->bytes_read = bytes_read;
}

int main(void) {
    static const unsigned char input[] = "ciao, l'Italia!\t42";
    static const unsigned char word_ciao[] = "ciao";
    static const unsigned char comma[] = ",";
    static const unsigned char space[] = " ";
    static const unsigned char word_italia[] = "l'Italia";
    static const unsigned char exclamation[] = "!";
    static const unsigned char tab[] = "\t";
    static const unsigned char number[] = "42";
    static const unsigned char *const expected_values[] = {
        word_ciao, comma, space, word_italia, exclamation, tab, number,
    };
    static const size_t expected_lengths[] = {4U, 1U, 1U, 8U, 1U, 1U, 2U};
    expected_pretokens expected = {
        .values = expected_values,
        .lengths = expected_lengths,
        .count = sizeof(expected_values) / sizeof(expected_values[0]),
    };
    TEST_ASSERT(tokenizer_pretokenize_bytes(input, sizeof(input) - 1U, expect_pretoken,
                                            &expected) == TOKENIZER_OK);
    TEST_ASSERT(expected.index == expected.count);

    expected.index = 0U;
    TEST_ASSERT(tokenizer_pretokenize_bytes(NULL, 0U, expect_pretoken, &expected) == TOKENIZER_OK);
    TEST_ASSERT(expected.index == 0U);
    TEST_ASSERT(tokenizer_pretokenize_bytes(NULL, 1U, expect_pretoken, &expected) ==
                TOKENIZER_INVALID_ARGUMENT);

    char path[1024] = {0};
    TEST_ASSERT(snprintf(path, sizeof(path), "%s/large-pretoken.txt", LLM_LAB_TEST_BINARY_DIR) > 0);
    FILE *file = fopen(path, "wb");
    TEST_ASSERT(file != NULL);
    static const size_t file_size = 64U * 1024U + 17U;
    for (size_t index = 0U; index < file_size; ++index) {
        TEST_ASSERT(fputc('a', file) != EOF);
    }
    TEST_ASSERT(fclose(file) == 0);

    file_observation observation = {0};
    TEST_ASSERT(tokenizer_pretokenize_file(path, observe_file_pretoken, observe_progress,
                                           &observation, &observation) == TOKENIZER_OK);
    TEST_ASSERT(observation.calls == 1U);
    TEST_ASSERT(observation.length == file_size);
    TEST_ASSERT(observation.bytes_read == file_size);
    TEST_ASSERT(remove(path) == 0);
    return EXIT_SUCCESS;
}
