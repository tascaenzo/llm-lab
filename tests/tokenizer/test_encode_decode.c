#include <stdlib.h>
#include <string.h>

#include "test_support.h"
#include "tokenizer_internal.h"

static int round_trip(const tokenizer *model, const unsigned char *input, size_t input_length) {
    token_sequence tokens = {0};
    if (tokenizer_encode(model, input, input_length, &tokens) != TOKENIZER_OK) {
        return 0;
    }

    unsigned char *decoded = NULL;
    size_t decoded_length = 0U;
    const tokenizer_status status = tokenizer_decode(model, &tokens, &decoded, &decoded_length);
    const int matches = status == TOKENIZER_OK && decoded_length == input_length &&
                        (input_length == 0U || memcmp(input, decoded, input_length) == 0);
    tokenizer_bytes_destroy(decoded);
    token_sequence_destroy(&tokens);
    return matches;
}

int main(void) {
    tokenizer *byte_model = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&byte_model) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_vocabulary_size(byte_model) == TOKENIZER_BYTE_VOCABULARY_SIZE);
    TEST_ASSERT(tokenizer_merge_count(byte_model) == 0U);

    static const unsigned char arbitrary_bytes[] = {
        'I', 't', 'a', 'l', 'i', 'a', 'n', 'o', ':', ' ', 0xC3U, 0xA8U, 0x00U, 0xFFU,
    };
    TEST_ASSERT(round_trip(byte_model, arbitrary_bytes, sizeof(arbitrary_bytes)) != 0);
    TEST_ASSERT(round_trip(byte_model, NULL, 0U) != 0);

    token_sequence bytes = {0};
    TEST_ASSERT(tokenizer_encode(byte_model, arbitrary_bytes, sizeof(arbitrary_bytes), &bytes) ==
                TOKENIZER_OK);
    TEST_ASSERT(bytes.length == sizeof(arbitrary_bytes));
    for (size_t index = 0U; index < bytes.length; ++index) {
        TEST_ASSERT(bytes.ids[index] == arbitrary_bytes[index]);
    }
    token_sequence_destroy(&bytes);
    tokenizer_destroy(byte_model);

    tokenizer *merged_model = NULL;
    TEST_ASSERT(tokenizer_create_byte_level(&merged_model) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_add_merge(merged_model, 97U, 98U) == TOKENIZER_OK);
    TEST_ASSERT(tokenizer_add_merge(merged_model, 256U, 99U) == TOKENIZER_OK);
    static const unsigned char merged_input[] = "abc!";
    token_sequence merged = {0};
    TEST_ASSERT(tokenizer_encode(merged_model, merged_input, sizeof(merged_input) - 1U, &merged) ==
                TOKENIZER_OK);
    TEST_ASSERT(merged.length == 2U);
    TEST_ASSERT(merged.ids[0] == 257U);
    TEST_ASSERT(merged.ids[1] == (token_id)'!');
    TEST_ASSERT(round_trip(merged_model, merged_input, sizeof(merged_input) - 1U) != 0);
    token_sequence_destroy(&merged);

    token_id invalid_id = 999U;
    const token_sequence invalid = {.ids = &invalid_id, .length = 1U};
    unsigned char *decoded = NULL;
    size_t decoded_length = 0U;
    TEST_ASSERT(tokenizer_decode(merged_model, &invalid, &decoded, &decoded_length) ==
                TOKENIZER_INVALID_TOKEN);
    TEST_ASSERT(decoded == NULL);
    TEST_ASSERT(decoded_length == 0U);
    tokenizer_destroy(merged_model);
    return EXIT_SUCCESS;
}
