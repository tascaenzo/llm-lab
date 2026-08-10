#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "dataset/dataset.h"
#include "tokenizer/tokenizer.h"

typedef struct cli_training_progress {
    tokenizer_train_phase phase;
    double phase_started_at;
    double last_update_at;
    int has_phase;
    int interactive;
} cli_training_progress;

typedef struct cli_dataset_progress {
    double started_at;
    double last_update_at;
    uint64_t next_log_bytes;
    int interactive;
    int has_output;
} cli_dataset_progress;

static double current_time_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER counter = {0};
    LARGE_INTEGER frequency = {0};
    if (QueryPerformanceCounter(&counter) != 0 && QueryPerformanceFrequency(&frequency) != 0) {
        return (double)counter.QuadPart / (double)frequency.QuadPart;
    }
#elif defined(CLOCK_MONOTONIC)
    struct timespec monotonic = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &monotonic) == 0) {
        return (double)monotonic.tv_sec + (double)monotonic.tv_nsec / 1000000000.0;
    }
#endif
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

static uint64_t sum_dataset_tokens(const uint64_t token_counts[3]) {
    uint64_t total = 0U;
    for (size_t index = 0U; index < 3U; ++index) {
        if (token_counts[index] > UINT64_MAX - total) {
            return UINT64_MAX;
        }
        total += token_counts[index];
    }
    return total;
}

static void show_dataset_progress(uint64_t bytes_read, uint64_t total_bytes,
                                  uint64_t documents_processed, const uint64_t token_counts[3],
                                  void *context) {
    cli_dataset_progress *progress = context;
    const double now = current_time_seconds();
    const int completed = bytes_read >= total_bytes;
    const uint64_t total_tokens = sum_dataset_tokens(token_counts);
    const double fraction = total_bytes == 0U ? 1.0 : (double)bytes_read / (double)total_bytes;
    const double elapsed = now - progress->started_at;
    const double bytes_per_second = elapsed > 0.0 ? (double)bytes_read / elapsed : 0.0;
    const double remaining_seconds = bytes_per_second > 0.0 && total_bytes > bytes_read
                                         ? (double)(total_bytes - bytes_read) / bytes_per_second
                                         : 0.0;

    if (progress->interactive == 0) {
        const uint64_t interval = total_bytes < UINT64_C(20) ? UINT64_C(1) : total_bytes / 20U;
        if (bytes_read != 0U && completed == 0 && bytes_read < progress->next_log_bytes) {
            return;
        }
        progress->next_log_bytes =
            bytes_read > UINT64_MAX - interval ? UINT64_MAX : bytes_read + interval;
    } else if (completed == 0 && bytes_read != 0U && now - progress->last_update_at < 0.5) {
        return;
    }

    char completed_bytes[32] = {0};
    char total_size[32] = {0};
    char eta[32] = "--:--";
    format_bytes(bytes_read, completed_bytes, sizeof(completed_bytes));
    format_bytes(total_bytes, total_size, sizeof(total_size));
    if (bytes_per_second > 0.0 && completed == 0) {
        format_duration(remaining_seconds, eta, sizeof(eta));
    }
    const double mebibytes_per_second = bytes_per_second / (1024.0 * 1024.0);

    if (progress->interactive != 0) {
        const size_t bar_width = 26U;
        const size_t filled = (size_t)(fraction * (double)bar_width);
        fprintf(stderr, "\rDataset [");
        for (size_t index = 0U; index < bar_width; ++index) {
            fputc(index < filled ? '#' : '-', stderr);
        }
        fprintf(stderr,
                "] %5.1f%% %s/%s | documenti %" PRIu64 " | token %" PRIu64 " | %.1f MiB/s | ETA %s",
                fraction * 100.0, completed_bytes, total_size, documents_processed, total_tokens,
                mebibytes_per_second, eta);
        fflush(stderr);
    } else {
        fprintf(stderr,
                "Dataset: %5.1f%% %s/%s, documenti %" PRIu64 ", token %" PRIu64
                ", %.1f MiB/s, ETA %s\n",
                fraction * 100.0, completed_bytes, total_size, documents_processed, total_tokens,
                mebibytes_per_second, eta);
    }
    progress->last_update_at = now;
    progress->has_output = 1;
}

static void finish_dataset_progress(const cli_dataset_progress *progress) {
    if (progress->interactive != 0 && progress->has_output != 0) {
        fputc('\n', stderr);
    }
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s tokenizer train OUTPUT.llmtok VOCAB_SIZE INPUT...\n"
            "  %s tokenizer evaluate MODEL.llmtok MAX_BYTES INPUT...\n"
            "  %s dataset prepare MODEL.llmtok DOCUMENTS.jsonl OUTPUT_PREFIX\n",
            program, program, program);
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

static int run_tokenizer_train(int argc, char **argv) {
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

static int parse_maximum_bytes(const char *text, size_t *out_size) {
    errno = 0;
    char *end = NULL;
    const uintmax_t value = strtoumax(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0U || value > SIZE_MAX) {
        return 0;
    }
    *out_size = (size_t)value;
    return 1;
}

static int evaluate_file(const tokenizer *tokenizer, const char *path, size_t maximum_bytes,
                         uint64_t *total_bytes, uint64_t *total_tokens) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open evaluation input: %s\n", path);
        return 0;
    }

    unsigned char *input = malloc(maximum_bytes);
    if (input == NULL) {
        fclose(file);
        fprintf(stderr, "Evaluation allocation failed.\n");
        return 0;
    }

    const size_t input_length = fread(input, 1U, maximum_bytes, file);
    const int stream_failed = ferror(file) != 0;
    const int close_failed = fclose(file) != 0;
    const int read_failed = stream_failed != 0 || close_failed != 0;
    if (read_failed != 0) {
        free(input);
        fprintf(stderr, "Cannot read evaluation input: %s\n", path);
        return 0;
    }

    token_sequence tokens = {0};
    tokenizer_status status = tokenizer_encode(tokenizer, input, input_length, &tokens);
    unsigned char *decoded = NULL;
    size_t decoded_length = 0U;
    if (status == TOKENIZER_OK) {
        status = tokenizer_decode(tokenizer, &tokens, &decoded, &decoded_length);
    }
    if (status != TOKENIZER_OK || decoded_length != input_length ||
        (input_length != 0U && memcmp(input, decoded, input_length) != 0)) {
        fprintf(stderr, "Evaluation round-trip failed for %s: %s\n", path,
                tokenizer_status_string(status));
        tokenizer_bytes_destroy(decoded);
        token_sequence_destroy(&tokens);
        free(input);
        return 0;
    }

    if (input_length > UINT64_MAX - *total_bytes || tokens.length > UINT64_MAX - *total_tokens) {
        fprintf(stderr, "Evaluation counters overflowed for %s.\n", path);
        tokenizer_bytes_destroy(decoded);
        token_sequence_destroy(&tokens);
        free(input);
        return 0;
    }

    *total_bytes += input_length;
    *total_tokens += tokens.length;
    tokenizer_bytes_destroy(decoded);
    token_sequence_destroy(&tokens);
    free(input);
    return 1;
}

static int run_tokenizer_evaluate(int argc, char **argv) {
    size_t maximum_bytes = 0U;
    if (parse_maximum_bytes(argv[4], &maximum_bytes) == 0) {
        fprintf(stderr, "MAX_BYTES must be a positive integer supported by this system.\n");
        return 1;
    }

    tokenizer *tokenizer = NULL;
    tokenizer_status status = tokenizer_load(argv[3], &tokenizer);
    if (status != TOKENIZER_OK) {
        fprintf(stderr, "Loading failed: %s\n", tokenizer_status_string(status));
        return 1;
    }

    const double started_at = current_time_seconds();
    uint64_t total_bytes = 0U;
    uint64_t total_tokens = 0U;
    for (int index = 5; index < argc && total_bytes < maximum_bytes; ++index) {
        const size_t remaining = maximum_bytes - (size_t)total_bytes;
        if (evaluate_file(tokenizer, argv[index], remaining, &total_bytes, &total_tokens) == 0) {
            tokenizer_destroy(tokenizer);
            return 1;
        }
    }
    const double elapsed = current_time_seconds() - started_at;
    tokenizer_destroy(tokenizer);

    if (total_bytes == 0U || total_tokens == 0U) {
        fprintf(stderr, "Evaluation inputs contain no data.\n");
        return 1;
    }

    const double bytes_per_token = (double)total_bytes / (double)total_tokens;
    const double mebibytes_per_second =
        elapsed > 0.0 ? ((double)total_bytes / (1024.0 * 1024.0)) / elapsed : 0.0;
    printf("{\"schema\":\"llm-lab-tokenizer-evaluation-v1\","
           "\"bytes\":%" PRIu64 ",\"tokens\":%" PRIu64 ","
           "\"bytes_per_token\":%.6f,\"seconds\":%.6f,"
           "\"mebibytes_per_second\":%.6f,\"round_trip\":true}\n",
           total_bytes, total_tokens, bytes_per_token, elapsed, mebibytes_per_second);
    return 0;
}

static int run_dataset_prepare(char **argv) {
    cli_dataset_progress progress = {
        .started_at = current_time_seconds(),
        .interactive = standard_error_is_terminal(),
    };
    lm_dataset_prepare_report report = {0};
    const lm_dataset_status status = lm_dataset_prepare_jsonl_with_progress(
        argv[3], argv[4], argv[5], show_dataset_progress, &progress, &report);
    finish_dataset_progress(&progress);
    if (status != LM_DATASET_OK) {
        fprintf(stderr, "Dataset preparation failed: %s\n", lm_dataset_status_string(status));
        return 1;
    }

    printf("{\"schema\":\"llm-lab-dataset-report-v1\","
           "\"tokenizer_vocabulary_size\":%" PRIu32 ","
           "\"model_vocabulary_size\":%" PRIu32 ","
           "\"end_of_document_token\":%" PRIu32 ","
           "\"splits\":{"
           "\"train\":{\"documents\":%" PRIu64 ",\"tokens\":%" PRIu64 "},"
           "\"validation\":{\"documents\":%" PRIu64 ",\"tokens\":%" PRIu64 "},"
           "\"test\":{\"documents\":%" PRIu64 ",\"tokens\":%" PRIu64 "}}}\n",
           report.tokenizer_vocabulary_size, report.model_vocabulary_size,
           report.end_of_document_token, report.document_counts[LM_DATASET_TRAIN],
           report.token_counts[LM_DATASET_TRAIN], report.document_counts[LM_DATASET_VALIDATION],
           report.token_counts[LM_DATASET_VALIDATION], report.document_counts[LM_DATASET_TEST],
           report.token_counts[LM_DATASET_TEST]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "tokenizer") == 0 && strcmp(argv[2], "train") == 0) {
        return run_tokenizer_train(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "tokenizer") == 0 && strcmp(argv[2], "evaluate") == 0) {
        return run_tokenizer_evaluate(argc, argv);
    }
    if (argc == 6 && strcmp(argv[1], "dataset") == 0 && strcmp(argv[2], "prepare") == 0) {
        return run_dataset_prepare(argv);
    }

    print_usage(argv[0]);
    return 1;
}
