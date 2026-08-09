#include <string.h>

#include "sha256.h"

#define SHA256_BLOCK_SIZE 64U

static const uint32_t round_constants[64] = {
    UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
    UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
    UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
    UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
    UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
    UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
    UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
    UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
    UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
    UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
    UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2),
};

static uint32_t rotate_right(uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32U - amount));
}

static uint32_t load_big_endian_u32(const unsigned char *bytes) {
    return ((uint32_t)bytes[0] << 24U) | ((uint32_t)bytes[1] << 16U) | ((uint32_t)bytes[2] << 8U) |
           (uint32_t)bytes[3];
}

static void store_big_endian_u32(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char)(value >> 24U);
    bytes[1] = (unsigned char)(value >> 16U);
    bytes[2] = (unsigned char)(value >> 8U);
    bytes[3] = (unsigned char)value;
}

static void store_big_endian_u64(unsigned char *bytes, uint64_t value) {
    for (size_t index = 0U; index < 8U; ++index) {
        bytes[7U - index] = (unsigned char)(value >> (index * 8U));
    }
}

static void transform_block(tokenizer_sha256_context *context, const unsigned char *block) {
    uint32_t words[64] = {0};
    for (size_t index = 0U; index < 16U; ++index) {
        words[index] = load_big_endian_u32(block + index * 4U);
    }
    for (size_t index = 16U; index < 64U; ++index) {
        const uint32_t first = words[index - 15U];
        const uint32_t second = words[index - 2U];
        const uint32_t sigma_zero =
            rotate_right(first, 7U) ^ rotate_right(first, 18U) ^ (first >> 3U);
        const uint32_t sigma_one =
            rotate_right(second, 17U) ^ rotate_right(second, 19U) ^ (second >> 10U);
        words[index] = words[index - 16U] + sigma_zero + words[index - 7U] + sigma_one;
    }

    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];

    for (size_t index = 0U; index < 64U; ++index) {
        const uint32_t choice = (e & f) ^ (~e & g);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t sum_zero = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const uint32_t sum_one = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const uint32_t temporary_one = h + sum_one + choice + round_constants[index] + words[index];
        const uint32_t temporary_two = sum_zero + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary_one;
        d = c;
        c = b;
        b = a;
        a = temporary_one + temporary_two;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void tokenizer_sha256_init(tokenizer_sha256_context *context) {
    *context = (tokenizer_sha256_context){
        .state =
            {
                UINT32_C(0x6a09e667),
                UINT32_C(0xbb67ae85),
                UINT32_C(0x3c6ef372),
                UINT32_C(0xa54ff53a),
                UINT32_C(0x510e527f),
                UINT32_C(0x9b05688c),
                UINT32_C(0x1f83d9ab),
                UINT32_C(0x5be0cd19),
            },
    };
}

void tokenizer_sha256_update(tokenizer_sha256_context *context, const unsigned char *data,
                             size_t length) {
    context->total_bytes += (uint64_t)length;
    while (length != 0U) {
        const size_t available = SHA256_BLOCK_SIZE - context->block_length;
        const size_t copied = length < available ? length : available;
        memcpy(context->block + context->block_length, data, copied);
        context->block_length += copied;
        data += copied;
        length -= copied;
        if (context->block_length == SHA256_BLOCK_SIZE) {
            transform_block(context, context->block);
            context->block_length = 0U;
        }
    }
}

void tokenizer_sha256_final(tokenizer_sha256_context *context,
                            unsigned char digest[TOKENIZER_SHA256_DIGEST_SIZE]) {
    const uint64_t total_bits = context->total_bytes * UINT64_C(8);
    context->block[context->block_length++] = 0x80U;
    if (context->block_length > 56U) {
        memset(context->block + context->block_length, 0,
               SHA256_BLOCK_SIZE - context->block_length);
        transform_block(context, context->block);
        context->block_length = 0U;
    }
    memset(context->block + context->block_length, 0, 56U - context->block_length);
    store_big_endian_u64(context->block + 56U, total_bits);
    transform_block(context, context->block);

    for (size_t index = 0U; index < 8U; ++index) {
        store_big_endian_u32(digest + index * 4U, context->state[index]);
    }
}
