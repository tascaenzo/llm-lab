#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "tokenizer/tokenizer.h"

typedef struct cli_training_progress {
    tokenizer_train_phase phase;
    double phase_started_at;
    double last_update_at;
    int has_phase;
    int interactive;
} cli_training_progress;

static double current_time_seconds(void) {
    struct timespec value = {0};
    (void)timespec_get(&value, TIME_UTC);
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int standard_error_is_terminal(void) {
#ifdef _WIN32
    return _isatty(_fileno(stderr));
#else
    return isatty(STDERR_FILENO);
#endif
}

static const char *training_phase_name(tokenizer_train_phase phase) {
    switch (phase) {
    case TOKENIZER_TRAIN_READING_INPUT:
        return "Lettura corpus";
    case TOKENIZER_TRAIN_COLLECTING_PAIRS:
        return "Indicizzazione coppie";
    case TOKENIZER_TRAIN_BUILDING_HEAP:
        return "Preparazione heap";
    case TOKENIZER_TRAIN_MERGING:
        return "Merge BPE";
    }

    return "Training";
}

static void format_bytes(uint64_t value, char *buffer, size_t buffer_size) {
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double size = (double)value;
    size_t unit = 0U;
    while (size >= 1024.0 && unit + 1U < sizeof(units) / sizeof(units[0])) {
        size /= 1024.0;
        ++unit;
    }
    (void)snprintf(buffer, buffer_size, "%.1f %s", size, units[unit]);
}

static void format_duration(double seconds, char *buffer, size_t buffer_size) {
    const uint64_t total_seconds = seconds > 0.0 ? (uint64_t)seconds : 0U;
    const uint64_t minutes = total_seconds / 60U;
    const uint64_t hours = minutes / 60U;
    if (hours != 0U) {
        (void)snprintf(buffer, buffer_size, "%" PRIu64 ":%02" PRIu64 ":%02" PRIu64, hours,
                       minutes % 60U, total_seconds % 60U);
    } else {
        (void)snprintf(buffer, buffer_size, "%" PRIu64 ":%02" PRIu64, minutes, total_seconds % 60U);
    }
}

static void show_training_progress(tokenizer_train_phase phase, uint64_t completed, uint64_t total,
                                   void *context) {
    cli_training_progress *progress = context;
    const double now = current_time_seconds();
    const int phase_changed = progress->has_phase == 0 || progress->phase != phase;
    if (phase_changed != 0) {
        progress->phase = phase;
        progress->phase_started_at = now;
        progress->last_update_at = 0.0;
        progress->has_phase = 1;
        if (progress->interactive == 0) {
            fprintf(stderr, "Training: %s...\n", training_phase_name(phase));
        }
    }

    if (progress->interactive == 0) {
        return;
    }
    if (phase_changed == 0 && completed != total && now - progress->last_update_at < 0.5) {
        return;
    }

    const double fraction = total == 0U ? 1.0 : (double)completed / (double)total;
    const size_t bar_width = 26U;
    const size_t filled = (size_t)(fraction * (double)bar_width);
    const double elapsed = now - progress->phase_started_at;
    const double rate = elapsed > 0.0 ? (double)completed / elapsed : 0.0;
    const double remaining = rate > 0.0 ? (double)(total - completed) / rate : 0.0;
    char eta[32] = "--:--";
    char units[64] = {0};
    if (rate > 0.0) {
        format_duration(remaining, eta, sizeof(eta));
    }

    if (phase == TOKENIZER_TRAIN_READING_INPUT) {
        char completed_bytes[32] = {0};
        char total_bytes[32] = {0};
        format_bytes(completed, completed_bytes, sizeof(completed_bytes));
        format_bytes(total, total_bytes, sizeof(total_bytes));
        (void)snprintf(units, sizeof(units), "%s/%s", completed_bytes, total_bytes);
    } else if (phase == TOKENIZER_TRAIN_MERGING) {
        (void)snprintf(units, sizeof(units), "merge %" PRIu64 "/%" PRIu64, completed, total);
    } else {
        (void)snprintf(units, sizeof(units), "%" PRIu64 "/%" PRIu64, completed, total);
    }

    fprintf(stderr, "\r%s [", training_phase_name(phase));
    for (size_t index = 0U; index < bar_width; ++index) {
        fputc(index < filled ? '#' : '-', stderr);
    }
    fprintf(stderr, "] %5.1f%% %s ETA %s", fraction * 100.0, units, eta);
    fflush(stderr);
    progress->last_update_at = now;
}

static void finish_training_progress(const cli_training_progress *progress) {
    if (progress->interactive != 0 && progress->has_phase != 0) {
        fputc('\n', stderr);
    }
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s tokenizer train OUTPUT.llmtok VOCAB_SIZE INPUT...\n",
            program);
}

static int parse_vocabulary_size(const char *text, uint32_t *out_size) {
    errno = 0;
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return 0;
    }

    *out_size = (uint32_t)value;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "tokenizer") != 0 || argc < 6 || strcmp(argv[2], "train") != 0) {
        print_usage(argv[0]);
        return 1;
    }

    uint32_t vocabulary_size = 0U;
    if (parse_vocabulary_size(argv[4], &vocabulary_size) == 0 ||
        vocabulary_size < TOKENIZER_BYTE_VOCABULARY_SIZE) {
        fprintf(stderr, "VOCAB_SIZE must be an integer of at least 256.\n");
        return 1;
    }

    cli_training_progress progress = {.interactive = standard_error_is_terminal()};
    tokenizer *tokenizer = NULL;
    tokenizer_status status = tokenizer_train_with_progress(
        (const char *const *)&argv[5], (size_t)(argc - 5), vocabulary_size, show_training_progress,
        &progress, &tokenizer);
    finish_training_progress(&progress);
    if (status != TOKENIZER_OK) {
        fprintf(stderr, "Training failed: %s\n", tokenizer_status_string(status));
        return 1;
    }

    status = tokenizer_save(tokenizer, argv[3]);
    if (status != TOKENIZER_OK) {
        fprintf(stderr, "Saving failed: %s\n", tokenizer_status_string(status));
        tokenizer_destroy(tokenizer);
        return 1;
    }

    printf("Saved %s with %" PRIu32 " tokens and %zu merges.\n", argv[3],
           tokenizer_vocabulary_size(tokenizer), tokenizer_merge_count(tokenizer));
    tokenizer_destroy(tokenizer);
    return 0;
}
