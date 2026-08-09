#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer/tokenizer.h"

#define INPUT_CAPACITY 4096U

static int read_line(const char *prompt, char *buffer, size_t capacity) {
    printf("%s", prompt);
    fflush(stdout);

    if (fgets(buffer, (int)capacity, stdin) == NULL) {
        return 0;
    }

    const size_t length = strlen(buffer);
    if (length != 0U && buffer[length - 1U] == '\n') {
        buffer[length - 1U] = '\0';
        return 1;
    }
    if (feof(stdin) != 0) {
        return 1;
    }

    int character = 0;
    while ((character = getchar()) != '\n' && character != EOF) {
    }
    fprintf(stderr, "Input troppo lungo (massimo %zu caratteri).\n", capacity - 1U);
    return -1;
}

static void print_status_error(const char *operation, tokenizer_status status) {
    fprintf(stderr, "%s: %s\n", operation, tokenizer_status_string(status));
}

static void print_token_ids(const token_sequence *tokens) {
    printf("Token IDs (%zu):", tokens->length);
    for (size_t index = 0U; index < tokens->length; ++index) {
        printf(" %" PRIu32, tokens->ids[index]);
    }
    printf("\n");
}

static void encode_text(const tokenizer *model) {
    char text[INPUT_CAPACITY] = {0};
    if (read_line("Testo da codificare: ", text, sizeof(text)) <= 0) {
        return;
    }

    token_sequence tokens = {0};
    const tokenizer_status status =
        tokenizer_encode(model, (const unsigned char *)text, strlen(text), &tokens);
    if (status == TOKENIZER_OK) {
        print_token_ids(&tokens);
    } else {
        print_status_error("Codifica non riuscita", status);
    }
    token_sequence_destroy(&tokens);
}

static int append_token(token_sequence *tokens, token_id id) {
    if (tokens->length >= SIZE_MAX / sizeof(*tokens->ids)) {
        return 0;
    }
    token_id *new_ids = realloc(tokens->ids, (tokens->length + 1U) * sizeof(*new_ids));
    if (new_ids == NULL) {
        return 0;
    }
    tokens->ids = new_ids;
    tokens->ids[tokens->length++] = id;
    return 1;
}

static int parse_token_ids(char *text, token_sequence *out_tokens) {
    char *cursor = text;
    while (*cursor != '\0') {
        while (isspace((unsigned char)*cursor) != 0) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        if (isdigit((unsigned char)*cursor) == 0) {
            return 0;
        }

        errno = 0;
        char *end = NULL;
        const unsigned long value = strtoul(cursor, &end, 10);
        if (errno != 0 || end == cursor || value > UINT32_MAX) {
            return 0;
        }
        if (append_token(out_tokens, (token_id)value) == 0) {
            return 0;
        }
        cursor = end;
    }
    return out_tokens->length != 0U;
}

static void decode_token_ids(const tokenizer *model) {
    char line[INPUT_CAPACITY] = {0};
    if (read_line("Token ID separati da spazi: ", line, sizeof(line)) <= 0) {
        return;
    }

    token_sequence tokens = {0};
    if (parse_token_ids(line, &tokens) == 0) {
        fprintf(stderr, "La lista di ID non e' valida.\n");
        token_sequence_destroy(&tokens);
        return;
    }

    unsigned char *decoded = NULL;
    size_t decoded_length = 0U;
    const tokenizer_status status = tokenizer_decode(model, &tokens, &decoded, &decoded_length);
    if (status == TOKENIZER_OK) {
        printf("Testo decodificato: ");
        (void)fwrite(decoded, 1U, decoded_length, stdout);
        printf("\n");
    } else {
        print_status_error("Decodifica non riuscita", status);
    }
    tokenizer_bytes_destroy(decoded);
    token_sequence_destroy(&tokens);
}

static void print_menu(void) {
    printf("\n=== Tokenizer Lab ===\n"
           "1) Testo -> token ID\n"
           "2) Token ID -> testo\n"
           "0) Esci\n");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s MODEL.llmtok\n", argv[0]);
        return EXIT_FAILURE;
    }

    tokenizer *model = NULL;
    const tokenizer_status load_status = tokenizer_load(argv[1], &model);
    if (load_status != TOKENIZER_OK) {
        print_status_error("Caricamento non riuscito", load_status);
        return EXIT_FAILURE;
    }
    printf("Caricato %s: %" PRIu32 " token, %zu merge BPE.\n", argv[1],
           tokenizer_vocabulary_size(model), tokenizer_merge_count(model));

    for (;;) {
        print_menu();
        char choice[INPUT_CAPACITY] = {0};
        const int read_status = read_line("Scelta: ", choice, sizeof(choice));
        if (read_status == 0 || strcmp(choice, "0") == 0) {
            printf("\nFine.\n");
            break;
        }
        if (read_status < 0) {
            continue;
        }
        if (strcmp(choice, "1") == 0) {
            encode_text(model);
        } else if (strcmp(choice, "2") == 0) {
            decode_token_ids(model);
        } else {
            fprintf(stderr, "Scelta non disponibile.\n");
        }
    }

    tokenizer_destroy(model);
    return EXIT_SUCCESS;
}
