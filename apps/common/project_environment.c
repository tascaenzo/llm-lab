#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "project_environment.h"

#define LLM_PROJECT_ENVIRONMENT_LINE_CAPACITY 8192U

static char *skip_space(char *text) {
    while (*text != '\0' && isspace((unsigned char)*text) != 0) {
        ++text;
    }
    return text;
}

static void trim_right(char *text) {
    size_t length = strlen(text);
    while (length > 0U && isspace((unsigned char)text[length - 1U]) != 0) {
        text[--length] = '\0';
    }
}

static int key_is_valid(const char *key) {
    if (key[0] == '\0' || (isalpha((unsigned char)key[0]) == 0 && key[0] != '_')) {
        return 0;
    }
    for (size_t index = 1U; key[index] != '\0'; ++index) {
        if (isalnum((unsigned char)key[index]) == 0 && key[index] != '_') {
            return 0;
        }
    }
    return 1;
}

static int decode_quoted_value(char *value, char **out_value) {
    const char quote = *value++;
    char *read = value;
    char *write = value;
    while (*read != '\0' && *read != quote) {
        if (quote == '"' && *read == '\\') {
            ++read;
            if (*read == '\0') {
                return 0;
            }
            if (*read == 'n') {
                *write++ = '\n';
            } else if (*read == 'r') {
                *write++ = '\r';
            } else if (*read == 't') {
                *write++ = '\t';
            } else if (*read == '"' || *read == '\\') {
                *write++ = *read;
            } else {
                return 0;
            }
            ++read;
            continue;
        }
        *write++ = *read++;
    }
    if (*read != quote) {
        return 0;
    }
    *write = '\0';
    read = skip_space(read + 1);
    if (*read != '\0' && *read != '#') {
        return 0;
    }
    *out_value = value;
    return 1;
}

static char *decode_unquoted_value(char *value) {
    for (char *cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor == '#' && (cursor == value || isspace((unsigned char)cursor[-1]) != 0)) {
            *cursor = '\0';
            break;
        }
    }
    trim_right(value);
    return value;
}

static int load_line(char *line, const char *path, size_t line_number) {
    char *entry = skip_space(line);
    if (*entry == '\0' || *entry == '#') {
        return 1;
    }
    if (strncmp(entry, "export", 6U) == 0 && isspace((unsigned char)entry[6]) != 0) {
        entry = skip_space(entry + 6);
    }
    char *separator = strchr(entry, '=');
    if (separator == NULL) {
        fprintf(stderr, "%s:%zu: expected KEY=VALUE\n", path, line_number);
        return 0;
    }
    *separator = '\0';
    trim_right(entry);
    if (key_is_valid(entry) == 0) {
        fprintf(stderr, "%s:%zu: invalid environment variable name\n", path, line_number);
        return 0;
    }

    char *value = skip_space(separator + 1);
    if (*value == '\'' || *value == '"') {
        if (decode_quoted_value(value, &value) == 0) {
            fprintf(stderr, "%s:%zu: invalid quoted environment value\n", path, line_number);
            return 0;
        }
    } else {
        value = decode_unquoted_value(value);
    }
    if (setenv(entry, value, 0) != 0) {
        fprintf(stderr, "%s:%zu: setting environment variable failed\n", path, line_number);
        return 0;
    }
    return 1;
}

int llm_project_environment_load(void) {
    const char *configured_path = getenv("LLM_LAB_ENV_FILE");
    const int path_is_explicit = configured_path != NULL && configured_path[0] != '\0';
    const char *path = path_is_explicit != 0 ? configured_path : ".env";
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        if (errno == ENOENT && path_is_explicit == 0) {
            return 1;
        }
        fprintf(stderr, "Loading environment file failed: %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[LLM_PROJECT_ENVIRONMENT_LINE_CAPACITY];
    size_t line_number = 0U;
    int status = 1;
    while (status != 0 && fgets(line, sizeof(line), file) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        if (length == sizeof(line) - 1U && line[length - 1U] != '\n' && feof(file) == 0) {
            fprintf(stderr, "%s:%zu: environment line is too long\n", path, line_number);
            status = 0;
            break;
        }
        line[strcspn(line, "\r\n")] = '\0';
        status = load_line(line, path, line_number);
    }
    if (ferror(file) != 0) {
        fprintf(stderr, "Reading environment file failed: %s\n", path);
        status = 0;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Closing environment file failed: %s\n", path);
        status = 0;
    }
    return status;
}
