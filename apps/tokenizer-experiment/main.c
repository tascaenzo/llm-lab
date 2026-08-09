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

    int character = 0;
    while ((character = getchar()) != '\n' && character != EOF) {
    }

    fprintf(stderr, "Input troppo lungo (massimo %zu caratteri).\n", capacity - 1U);
    return -1;
}

static int parse_uint32(const char *text, uint32_t minimum, uint32_t *out_value) {
    errno = 0;
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX || value < minimum) {
        return 0;
    }

    *out_value = (uint32_t)value;
    return 1;
}

static int prompt_uint32(const char *prompt, uint32_t minimum, uint32_t *out_value) {
    char line[INPUT_CAPACITY] = {0};
    const int result = read_line(prompt, line, sizeof(line));
    if (result <= 0) {
        return result;
    }

    if (parse_uint32(line, minimum, out_value) == 0) {
        fprintf(stderr, "Inserisci un intero di almeno %" PRIu32 ".\n", minimum);
        return -1;
    }

    return 1;
}

static void print_token_ids(const token_sequence *tokens) {
    printf("Token IDs (%zu):", tokens->length);
    for (size_t index = 0U; index < tokens->length; ++index) {
        printf(" %" PRIu32, tokens->ids[index]);
    }
    printf("\n");
}

static void print_status_error(const char *operation, tokenizer_status status) {
    fprintf(stderr, "%s: %s\n", operation, tokenizer_status_string(status));
}

static tokenizer *load_tokenizer_from_prompt(void) {
    char model_path[INPUT_CAPACITY] = {0};
    const int result = read_line("Percorso del file .llmtok: ", model_path, sizeof(model_path));
    if (result <= 0) {
        return NULL;
    }

    tokenizer *tokenizer = NULL;
    const tokenizer_status status = tokenizer_load(model_path, &tokenizer);
    if (status != TOKENIZER_OK) {
        print_status_error("Caricamento non riuscito", status);
        return NULL;
    }

    printf("Caricato: %" PRIu32 " token, %zu merge BPE.\n", tokenizer_vocabulary_size(tokenizer),
           tokenizer_merge_count(tokenizer));
    return tokenizer;
}

static void train_vocabulary(void) {
    char corpus_path[INPUT_CAPACITY] = {0};
    char output_path[INPUT_CAPACITY] = {0};
    uint32_t vocabulary_size = 0U;

    if (read_line("Percorso del corpus: ", corpus_path, sizeof(corpus_path)) <= 0 ||
        read_line("File .llmtok da creare: ", output_path, sizeof(output_path)) <= 0 ||
        prompt_uint32("Vocabolario massimo (es. 32000): ", TOKENIZER_BYTE_VOCABULARY_SIZE,
                      &vocabulary_size) <= 0) {
        return;
    }

    const char *input_paths[] = {corpus_path};
    tokenizer *tokenizer = NULL;
    tokenizer_status status = tokenizer_train(input_paths, 1U, vocabulary_size, &tokenizer);
    if (status != TOKENIZER_OK) {
        print_status_error("Training non riuscito", status);
        return;
    }

    status = tokenizer_save(tokenizer, output_path);
    if (status != TOKENIZER_OK) {
        print_status_error("Salvataggio non riuscito", status);
        tokenizer_destroy(tokenizer);
        return;
    }

    printf("Creato %s con %" PRIu32 " token e %zu merge BPE.\n", output_path,
           tokenizer_vocabulary_size(tokenizer), tokenizer_merge_count(tokenizer));
    tokenizer_destroy(tokenizer);
}

static void encode_text(void) {
    tokenizer *tokenizer = load_tokenizer_from_prompt();
    if (tokenizer == NULL) {
        return;
    }

    char text[INPUT_CAPACITY] = {0};
    if (read_line("Testo da codificare: ", text, sizeof(text)) <= 0) {
        tokenizer_destroy(tokenizer);
        return;
    }

    token_sequence tokens = {0};
    const tokenizer_status status =
        tokenizer_encode(tokenizer, (const unsigned char *)text, strlen(text), &tokens);
    if (status == TOKENIZER_OK) {
        print_token_ids(&tokens);
    } else {
        print_status_error("Codifica non riuscita", status);
    }

    token_sequence_destroy(&tokens);
    tokenizer_destroy(tokenizer);
}

static int append_token(token_sequence *tokens, token_id id) {
    if (tokens->length == SIZE_MAX / sizeof(*tokens->ids)) {
        return 0;
    }

    token_id *new_ids = realloc(tokens->ids, (tokens->length + 1U) * sizeof(*new_ids));
    if (new_ids == NULL) {
        return 0;
    }

    tokens->ids = new_ids;
    tokens->ids[tokens->length] = id;
    ++tokens->length;
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

static void decode_token_ids(void) {
    tokenizer *tokenizer = load_tokenizer_from_prompt();
    if (tokenizer == NULL) {
        return;
    }

    char line[INPUT_CAPACITY] = {0};
    if (read_line("Token ID separati da spazi: ", line, sizeof(line)) <= 0) {
        tokenizer_destroy(tokenizer);
        return;
    }

    token_sequence tokens = {0};
    if (parse_token_ids(line, &tokens) == 0) {
        fprintf(stderr, "La lista di ID non e' valida.\n");
        token_sequence_destroy(&tokens);
        tokenizer_destroy(tokenizer);
        return;
    }

    unsigned char *decoded = NULL;
    size_t decoded_length = 0U;
    const tokenizer_status status = tokenizer_decode(tokenizer, &tokens, &decoded, &decoded_length);
    if (status == TOKENIZER_OK) {
        printf("Testo decodificato: ");
        fwrite(decoded, 1U, decoded_length, stdout);
        printf("\n");
    } else {
        print_status_error("Decodifica non riuscita", status);
    }

    tokenizer_bytes_destroy(decoded);
    token_sequence_destroy(&tokens);
    tokenizer_destroy(tokenizer);
}

static void print_menu(void) {
    printf("\n=== Tokenizer Lab ===\n"
           "1) Genera e salva un nuovo vocabolario BPE\n"
           "2) Carica un vocabolario e codifica testo in ID\n"
           "3) Carica un vocabolario e decodifica ID in testo\n"
           "0) Esci\n");
}

int main(void) {
    for (;;) {
        print_menu();

        uint32_t choice = 0U;
        const int result = prompt_uint32("Scelta: ", 0U, &choice);
        if (result == 0) {
            printf("\nFine.\n");
            return 0;
        }
        if (result < 0) {
            continue;
        }

        switch (choice) {
        case 0U:
            printf("Fine.\n");
            return 0;
        case 1U:
            train_vocabulary();
            break;
        case 2U:
            encode_text();
            break;
        case 3U:
            decode_token_ids();
            break;
        default:
            fprintf(stderr, "Scelta non disponibile.\n");
            break;
        }
    }
}
