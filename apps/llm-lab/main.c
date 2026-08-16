#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <unistd.h>

#include "dataset/dataset.h"
#include "model/model.h"
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

typedef struct cli_model_dataset_open_progress {
    double started_at;
    double last_update_at;
    uint64_t next_log_bytes;
    int interactive;
    int has_output;
} cli_model_dataset_open_progress;

typedef struct cli_model_training_progress {
    double started_at;
    double last_update_at;
    size_t next_log_step;
    int interactive;
    int has_output;
} cli_model_training_progress;

typedef struct cli_generation_candidate {
    token_id token;
    float logit;
} cli_generation_candidate;

typedef struct cli_generation_options {
    float temperature;
    float repetition_penalty;
    size_t top_k;
    uint64_t random_state;
} cli_generation_options;

static double current_time_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec monotonic = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &monotonic) == 0) {
        return (double)monotonic.tv_sec + (double)monotonic.tv_nsec / 1000000000.0;
    }
#endif
    struct timespec value = {0};
    (void)timespec_get(&value, TIME_UTC);
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int standard_error_is_terminal(void) { return isatty(STDERR_FILENO); }

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

static void show_model_dataset_open_progress(uint64_t bytes_read, uint64_t total_bytes,
                                             void *context) {
    cli_model_dataset_open_progress *progress = context;
    const double now = current_time_seconds();
    const int completed = bytes_read >= total_bytes;
    const double fraction = total_bytes == 0U ? 1.0 : (double)bytes_read / (double)total_bytes;
    const double elapsed = now - progress->started_at;
    const double bytes_per_second = elapsed > 0.0 ? (double)bytes_read / elapsed : 0.0;
    const double remaining_seconds = bytes_per_second > 0.0 && completed == 0
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
    if (remaining_seconds > 0.0) {
        format_duration(remaining_seconds, eta, sizeof(eta));
    }
    const double mebibytes_per_second = bytes_per_second / (1024.0 * 1024.0);
    if (progress->interactive != 0) {
        const size_t bar_width = 26U;
        const size_t filled = (size_t)(fraction * (double)bar_width);
        fprintf(stderr, "\rVerifica dataset [");
        for (size_t index = 0U; index < bar_width; ++index) {
            fputc(index < filled ? '#' : '-', stderr);
        }
        fprintf(stderr, "] %5.1f%% %s/%s | %.1f MiB/s | ETA %s", fraction * 100.0, completed_bytes,
                total_size, mebibytes_per_second, eta);
        fflush(stderr);
    } else {
        fprintf(stderr, "Verifica dataset: %5.1f%% %s/%s, %.1f MiB/s, ETA %s\n", fraction * 100.0,
                completed_bytes, total_size, mebibytes_per_second, eta);
    }
    progress->last_update_at = now;
    progress->has_output = 1;
}

static void finish_model_dataset_open_progress(const cli_model_dataset_open_progress *progress) {
    if (progress->interactive != 0 && progress->has_output != 0) {
        fputc('\n', stderr);
    }
}

static void show_model_training_progress(size_t completed, size_t total, float loss,
                                         void *context) {
    cli_model_training_progress *progress = context;
    const double now = current_time_seconds();
    const int is_complete = completed >= total;
    const double fraction = total == 0U ? 1.0 : (double)completed / (double)total;
    const double elapsed = now - progress->started_at;
    const double steps_per_second = elapsed > 0.0 ? (double)completed / elapsed : 0.0;
    const double remaining_seconds = steps_per_second > 0.0 && is_complete == 0
                                         ? (double)(total - completed) / steps_per_second
                                         : 0.0;
    if (progress->interactive == 0) {
        const size_t interval = total < 20U ? 1U : total / 20U;
        if (completed != 0U && is_complete == 0 && completed < progress->next_log_step) {
            return;
        }
        progress->next_log_step = completed > SIZE_MAX - interval ? SIZE_MAX : completed + interval;
    } else if (is_complete == 0 && completed != 0U && now - progress->last_update_at < 0.5) {
        return;
    }

    char eta[32] = "--:--";
    if (remaining_seconds > 0.0) {
        format_duration(remaining_seconds, eta, sizeof(eta));
    }
    if (progress->interactive != 0) {
        const size_t bar_width = 26U;
        const size_t filled = (size_t)(fraction * (double)bar_width);
        fprintf(stderr, "\rTraining modello [");
        for (size_t index = 0U; index < bar_width; ++index) {
            fputc(index < filled ? '#' : '-', stderr);
        }
        if (completed == 0U) {
            fprintf(stderr, "] %5.1f%% step 0/%zu | preparazione...", fraction * 100.0, total);
        } else {
            fprintf(stderr, "] %5.1f%% step %zu/%zu | loss %.6f | %.2f step/s | ETA %s",
                    fraction * 100.0, completed, total, loss, steps_per_second, eta);
        }
        fflush(stderr);
    } else if (completed == 0U) {
        fprintf(stderr, "Training modello: preparazione di %zu step...\n", total);
    } else {
        fprintf(stderr, "Training modello: %5.1f%% step %zu/%zu, loss %.6f, %.2f step/s, ETA %s\n",
                fraction * 100.0, completed, total, loss, steps_per_second, eta);
    }
    progress->last_update_at = now;
    progress->has_output = 1;
}

static void finish_model_training_progress(const cli_model_training_progress *progress) {
    if (progress->interactive != 0 && progress->has_output != 0) {
        fputc('\n', stderr);
    }
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s tokenizer train OUTPUT.llmtok VOCAB_SIZE INPUT...\n"
            "  %s tokenizer evaluate MODEL.llmtok MAX_BYTES INPUT...\n"
            "  %s dataset prepare MODEL.llmtok DOCUMENTS.jsonl OUTPUT_PREFIX"
            " [--reserved-tokens N]\n"
            "  %s model train TRAIN.llmdat STEPS [--batch-size B] [--context T]"
            " [--hidden C] [--layers L] [--heads H] [--ffn F]"
            " [--learning-rate LR] [--gradient-accumulation N]"
            " [--warmup-steps N] [--total-steps N] [--min-learning-rate LR] [--seed N]"
            " [--beta1 B] [--beta2 B] [--epsilon E] [--weight-decay W]"
            " [--sampling shuffled|random] [--gradient-clip N]"
            " [--backend cpu|metal]"
            " [--checkpoint FILE] [--checkpoint-every N] [--resume FILE]"
            " [--validation FILE] [--validation-every N] [--validation-batches N]"
            " [--best-checkpoint FILE] [--log FILE]\n"
            "  %s model generate CHECKPOINT.llmckpt TOKENIZER.llmtok TOKENS PROMPT"
            " [--temperature T] [--top-k K] [--repetition-penalty P] [--seed N]\n"
            "  %s model evaluate VALIDATION.llmdat CHECKPOINT.llmckpt STEPS"
            " [--batch-size B] [--seed N]\n",
            program, program, program, program, program, program);
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

static int parse_positive_size(const char *text, size_t *out_size) {
    return parse_maximum_bytes(text, out_size);
}

static int parse_size(const char *text, size_t *out_size) {
    errno = 0;
    char *end = NULL;
    const uintmax_t value = strtoumax(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) {
        return 0;
    }
    *out_size = (size_t)value;
    return 1;
}

static int parse_seed(const char *text, uint64_t *out_seed) {
    errno = 0;
    char *end = NULL;
    const uintmax_t value = strtoumax(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT64_MAX) {
        return 0;
    }
    *out_seed = (uint64_t)value;
    return 1;
}

static int parse_positive_float(const char *text, float *out_value) {
    errno = 0;
    char *end = NULL;
    const float value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || isfinite(value) == 0 || value <= 0.0F) {
        return 0;
    }
    *out_value = value;
    return 1;
}

static int generation_candidate_compare(const void *left, const void *right) {
    const cli_generation_candidate *left_candidate = left;
    const cli_generation_candidate *right_candidate = right;
    if (left_candidate->logit > right_candidate->logit) {
        return -1;
    }
    if (left_candidate->logit < right_candidate->logit) {
        return 1;
    }
    return left_candidate->token < right_candidate->token   ? -1
           : left_candidate->token > right_candidate->token ? 1
                                                            : 0;
}

static uint64_t generation_next_random(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12U;
    value ^= value << 25U;
    value ^= value >> 27U;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static double generation_uniform(uint64_t *state) {
    return (double)(generation_next_random(state) >> 11U) * (1.0 / 9007199254740992.0);
}

static token_id sample_generation_token(const float *logits, uint32_t tokenizer_vocabulary_size,
                                        const token_id *tokens, size_t token_count,
                                        size_t context_length, cli_generation_candidate *candidates,
                                        unsigned char *recent_tokens,
                                        cli_generation_options *options) {
    /*
     * The dataset vocabulary appends <EOD> and, optionally, reserved IDs after
     * the tokenizer vocabulary.  The latter have no textual representation
     * during base-model generation, so sampling them would make decode fail.
     */
    const size_t candidate_count = (size_t)tokenizer_vocabulary_size - 1U;
    (void)memset(recent_tokens, 0, tokenizer_vocabulary_size * sizeof(*recent_tokens));
    const size_t recent_start = token_count > context_length ? token_count - context_length : 0U;
    for (size_t index = recent_start; index < token_count; ++index) {
        if (tokens[index] < tokenizer_vocabulary_size) {
            recent_tokens[tokens[index]] = 1U;
        }
    }
    for (uint32_t token = 1U; token < tokenizer_vocabulary_size; ++token) {
        float adjusted = logits[token];
        if (recent_tokens[token] != 0U) {
            adjusted = adjusted >= 0.0F ? adjusted / options->repetition_penalty
                                        : adjusted * options->repetition_penalty;
        }
        candidates[token - 1U] = (cli_generation_candidate){.token = token, .logit = adjusted};
    }
    qsort(candidates, candidate_count, sizeof(*candidates), generation_candidate_compare);
    const size_t selected_count =
        options->top_k < candidate_count ? options->top_k : candidate_count;
    const float maximum = candidates[0].logit;
    double weight_sum = 0.0;
    for (size_t index = 0U; index < selected_count; ++index) {
        weight_sum +=
            exp(((double)candidates[index].logit - (double)maximum) / (double)options->temperature);
    }
    double sample = generation_uniform(&options->random_state) * weight_sum;
    for (size_t index = 0U; index < selected_count; ++index) {
        sample -=
            exp(((double)candidates[index].logit - (double)maximum) / (double)options->temperature);
        if (sample <= 0.0) {
            return candidates[index].token;
        }
    }
    return candidates[selected_count - 1U].token;
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

static int run_dataset_prepare(int argc, char **argv) {
    size_t reserved_tokens = 0U;
    for (int index = 6; index < argc; index += 2) {
        if (index + 1 >= argc || strcmp(argv[index], "--reserved-tokens") != 0 ||
            parse_size(argv[index + 1], &reserved_tokens) == 0 || reserved_tokens > UINT32_MAX) {
            fprintf(stderr, "Invalid dataset option: use --reserved-tokens N.\n");
            return 1;
        }
    }
    cli_dataset_progress progress = {
        .started_at = current_time_seconds(),
        .interactive = standard_error_is_terminal(),
    };
    lm_dataset_prepare_report report = {0};
    const lm_dataset_status status =
        lm_dataset_prepare_jsonl_reserved(argv[3], argv[4], argv[5], (uint32_t)reserved_tokens,
                                          show_dataset_progress, &progress, &report);
    finish_dataset_progress(&progress);
    if (status != LM_DATASET_OK) {
        fprintf(stderr, "Dataset preparation failed: %s\n", lm_dataset_status_string(status));
        return 1;
    }

    printf("{\"schema\":\"llm-lab-dataset-report-v1\","
           "\"tokenizer_vocabulary_size\":%" PRIu32 ","
           "\"model_vocabulary_size\":%" PRIu32 ","
           "\"reserved_token_count\":%" PRIu32 ","
           "\"end_of_document_token\":%" PRIu32 ","
           "\"splits\":{"
           "\"train\":{\"documents\":%" PRIu64 ",\"tokens\":%" PRIu64 "},"
           "\"validation\":{\"documents\":%" PRIu64 ",\"tokens\":%" PRIu64 "},"
           "\"test\":{\"documents\":%" PRIu64 ",\"tokens\":%" PRIu64 "}}}\n",
           report.tokenizer_vocabulary_size, report.model_vocabulary_size,
           report.reserved_token_count, report.end_of_document_token,
           report.document_counts[LM_DATASET_TRAIN], report.token_counts[LM_DATASET_TRAIN],
           report.document_counts[LM_DATASET_VALIDATION],
           report.token_counts[LM_DATASET_VALIDATION], report.document_counts[LM_DATASET_TEST],
           report.token_counts[LM_DATASET_TEST]);
    return 0;
}

static const char *model_sampling_name(lm_batcher_sampling sampling) {
    if (sampling == LM_BATCHER_SHUFFLED_BLOCKS)
        return "shuffled";
    if (sampling == LM_BATCHER_SHUFFLED_WINDOWS)
        return "shuffled-windows";
    return "random";
}

static char *path_with_suffix(const char *path, const char *suffix) {
    if (path == NULL || suffix == NULL)
        return NULL;
    const size_t path_length = strlen(path);
    const size_t suffix_length = strlen(suffix);
    if (path_length > SIZE_MAX - suffix_length - 1U)
        return NULL;
    char *result = malloc(path_length + suffix_length + 1U);
    if (result != NULL) {
        memcpy(result, path, path_length);
        memcpy(result + path_length, suffix, suffix_length + 1U);
    }
    return result;
}

static int load_best_validation_loss(const char *checkpoint_path, double *out_loss) {
    char *metadata_path = path_with_suffix(checkpoint_path, ".metrics.json");
    if (metadata_path == NULL)
        return 0;
    FILE *file = fopen(metadata_path, "rb");
    free(metadata_path);
    if (file == NULL)
        return 0;
    char buffer[512] = {0};
    const size_t length = fread(buffer, 1U, sizeof(buffer) - 1U, file);
    const int read_failed = ferror(file) != 0 || fclose(file) != 0;
    if (read_failed != 0)
        return 0;
    buffer[length] = '\0';
    char *field = strstr(buffer, "\"loss\":");
    if (field == NULL)
        return 0;
    char *end = NULL;
    const double value = strtod(field + strlen("\"loss\":"), &end);
    if (end == field + strlen("\"loss\":") || isfinite(value) == 0)
        return 0;
    *out_loss = value;
    return 1;
}

static int save_best_validation_metadata(const char *checkpoint_path, unsigned long long step,
                                         double loss) {
    char *metadata_path = path_with_suffix(checkpoint_path, ".metrics.json");
    char *temporary_path = path_with_suffix(checkpoint_path, ".metrics.json.part");
    if (metadata_path == NULL || temporary_path == NULL) {
        free(temporary_path);
        free(metadata_path);
        return 0;
    }
    FILE *file = fopen(temporary_path, "wb");
    int success = file != NULL;
    if (success != 0) {
        success = fprintf(file,
                          "{\"schema\":\"llm-lab-best-checkpoint-v1\",\"step\":%llu,"
                          "\"loss\":%.9g,\"perplexity\":%.9g}\n",
                          step, loss, exp(loss)) > 0;
        if (fclose(file) != 0)
            success = 0;
    }
    if (success != 0 && rename(temporary_path, metadata_path) != 0)
        success = 0;
    if (success == 0)
        (void)remove(temporary_path);
    free(temporary_path);
    free(metadata_path);
    return success;
}

/**
 * Set when the user asks a long training run to stop. The loop finishes the
 * step in flight, writes the checkpoint and reports normally, so days of work
 * are not lost to a keyboard interrupt.
 */
static volatile sig_atomic_t model_training_interrupted = 0;

static void request_model_training_stop(int signal_number) {
    (void)signal_number;
    model_training_interrupted = 1;
}

static void install_model_training_signals(void) {
    (void)signal(SIGINT, request_model_training_stop);
    (void)signal(SIGTERM, request_model_training_stop);
}

static int run_model_train(int argc, char **argv) {
    size_t steps = 0U;
    lm_trainer_config trainer_config = {.batch_size = 2U,
                                        .context_length = 32U,
                                        .seed = UINT64_C(1),
                                        .learning_rate = 1.0e-3F,
                                        .beta1 = 0.9F,
                                        .beta2 = 0.999F,
                                        .epsilon = 1.0e-8F,
                                        .weight_decay = 0.01F,
                                        .gradient_accumulation_steps = 1U,
                                        .warmup_steps = 0U,
                                        .total_steps = 0U,
                                        .minimum_learning_rate = 1.0e-3F,
                                        .sampling = LM_BATCHER_SHUFFLED_BLOCKS,
                                        .gradient_clip_norm = 0.0F};
    size_t hidden_size = 64U;
    size_t layer_count = 1U;
    size_t head_count = 1U;
    size_t feed_forward_size = 0U;
    const char *checkpoint_path = NULL;
    const char *resume_path = NULL;
    const char *validation_path = NULL;
    const char *best_checkpoint_path = NULL;
    const char *log_path = NULL;
    size_t checkpoint_every = 0U;
    size_t validation_every = 0U;
    size_t validation_batches = 100U;
    int use_metal = 0;
    int has_model_options = 0;
    if (parse_positive_size(argv[4], &steps) == 0) {
        fprintf(stderr, "STEPS must be a positive integer supported by this system.\n");
        return 1;
    }
    for (int index = 5; index < argc; index += 2) {
        if (index + 1 >= argc) {
            fprintf(stderr, "Model training options require a value.\n");
            return 1;
        }
        const char *option = argv[index];
        const char *value = argv[index + 1];
        int parsed = 0;
        if (strcmp(option, "--batch-size") == 0) {
            parsed = parse_positive_size(value, &trainer_config.batch_size);
            has_model_options = 1;
        } else if (strcmp(option, "--context") == 0) {
            parsed = parse_positive_size(value, &trainer_config.context_length);
            has_model_options = 1;
        } else if (strcmp(option, "--hidden") == 0) {
            parsed = parse_positive_size(value, &hidden_size);
            has_model_options = 1;
        } else if (strcmp(option, "--layers") == 0) {
            parsed = parse_size(value, &layer_count);
            has_model_options = 1;
        } else if (strcmp(option, "--heads") == 0) {
            parsed = parse_positive_size(value, &head_count);
            has_model_options = 1;
        } else if (strcmp(option, "--ffn") == 0) {
            parsed = parse_size(value, &feed_forward_size);
            has_model_options = 1;
        } else if (strcmp(option, "--learning-rate") == 0) {
            parsed = parse_positive_float(value, &trainer_config.learning_rate);
            has_model_options = 1;
        } else if (strcmp(option, "--gradient-accumulation") == 0) {
            parsed = parse_positive_size(value, &trainer_config.gradient_accumulation_steps);
            has_model_options = 1;
        } else if (strcmp(option, "--warmup-steps") == 0) {
            parsed = parse_seed(value, &trainer_config.warmup_steps);
            has_model_options = 1;
        } else if (strcmp(option, "--total-steps") == 0) {
            parsed = parse_seed(value, &trainer_config.total_steps);
            has_model_options = 1;
        } else if (strcmp(option, "--min-learning-rate") == 0) {
            parsed = parse_positive_float(value, &trainer_config.minimum_learning_rate);
            has_model_options = 1;
        } else if (strcmp(option, "--beta1") == 0) {
            parsed = parse_positive_float(value, &trainer_config.beta1);
            has_model_options = 1;
        } else if (strcmp(option, "--beta2") == 0) {
            parsed = parse_positive_float(value, &trainer_config.beta2);
            has_model_options = 1;
        } else if (strcmp(option, "--epsilon") == 0) {
            parsed = parse_positive_float(value, &trainer_config.epsilon);
            has_model_options = 1;
        } else if (strcmp(option, "--weight-decay") == 0) {
            parsed = parse_positive_float(value, &trainer_config.weight_decay);
            has_model_options = 1;
        } else if (strcmp(option, "--sampling") == 0) {
            if (strcmp(value, "shuffled") == 0) {
                trainer_config.sampling = LM_BATCHER_SHUFFLED_BLOCKS;
                parsed = 1;
            } else if (strcmp(value, "random") == 0) {
                trainer_config.sampling = LM_BATCHER_RANDOM_WINDOWS;
                parsed = 1;
            }
            has_model_options = 1;
        } else if (strcmp(option, "--gradient-clip") == 0) {
            parsed = parse_positive_float(value, &trainer_config.gradient_clip_norm);
            has_model_options = 1;
        } else if (strcmp(option, "--seed") == 0) {
            parsed = parse_seed(value, &trainer_config.seed);
            has_model_options = 1;
        } else if (strcmp(option, "--backend") == 0) {
            if (strcmp(value, "cpu") == 0) {
                parsed = 1;
            } else if (strcmp(value, "metal") == 0) {
                use_metal = 1;
                parsed = 1;
            }
        } else if (strcmp(option, "--checkpoint") == 0 && checkpoint_path == NULL) {
            checkpoint_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--checkpoint-every") == 0) {
            parsed = parse_positive_size(value, &checkpoint_every);
        } else if (strcmp(option, "--resume") == 0 && resume_path == NULL) {
            resume_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--validation") == 0 && validation_path == NULL) {
            validation_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--validation-every") == 0) {
            parsed = parse_positive_size(value, &validation_every);
        } else if (strcmp(option, "--validation-batches") == 0) {
            parsed = parse_positive_size(value, &validation_batches);
        } else if (strcmp(option, "--best-checkpoint") == 0 && best_checkpoint_path == NULL) {
            best_checkpoint_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--log") == 0 && log_path == NULL) {
            log_path = value;
            parsed = value[0] != '\0';
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid model training option: %s %s\n", option, value);
            return 1;
        }
    }
    if (resume_path != NULL && has_model_options != 0) {
        fprintf(stderr,
                "--resume restores model and trainer settings; do not override their options.\n");
        return 1;
    }
    if (checkpoint_every != 0U && checkpoint_path == NULL) {
        fprintf(stderr, "--checkpoint-every requires --checkpoint FILE.\n");
        return 1;
    }
    if (validation_path == NULL && (validation_every != 0U || best_checkpoint_path != NULL)) {
        fprintf(stderr, "Validation options require --validation FILE.\n");
        return 1;
    }
    if (validation_path != NULL && validation_every == 0U)
        validation_every = 2000U;
    if (checkpoint_path != NULL && best_checkpoint_path != NULL &&
        strcmp(checkpoint_path, best_checkpoint_path) == 0) {
        fprintf(stderr, "Latest and best checkpoint paths must be different.\n");
        return 1;
    }
    if (trainer_config.minimum_learning_rate > trainer_config.learning_rate) {
        fprintf(stderr, "--min-learning-rate must not exceed --learning-rate.\n");
        return 1;
    }

    cli_model_dataset_open_progress dataset_progress = {
        .started_at = current_time_seconds(),
        .interactive = standard_error_is_terminal(),
    };
    lm_dataset *dataset = NULL;
    lm_dataset_status dataset_status = lm_dataset_open_with_progress(
        argv[3], show_model_dataset_open_progress, &dataset_progress, &dataset);
    finish_model_dataset_open_progress(&dataset_progress);
    if (dataset_status != LM_DATASET_OK) {
        fprintf(stderr, "Opening training dataset failed: %s\n",
                lm_dataset_status_string(dataset_status));
        return 1;
    }
    if (lm_dataset_get_split(dataset) != LM_DATASET_TRAIN) {
        fprintf(stderr, "Model training requires a training split artifact.\n");
        lm_dataset_close(dataset);
        return 1;
    }

    lm_dataset *validation_dataset = NULL;
    if (validation_path != NULL) {
        dataset_progress = (cli_model_dataset_open_progress){
            .started_at = current_time_seconds(),
            .interactive = standard_error_is_terminal(),
        };
        dataset_status =
            lm_dataset_open_with_progress(validation_path, show_model_dataset_open_progress,
                                          &dataset_progress, &validation_dataset);
        finish_model_dataset_open_progress(&dataset_progress);
        if (dataset_status != LM_DATASET_OK ||
            lm_dataset_get_split(validation_dataset) != LM_DATASET_VALIDATION) {
            fprintf(stderr, "Opening validation dataset failed: %s\n",
                    lm_dataset_status_string(dataset_status));
            lm_dataset_close(validation_dataset);
            lm_dataset_close(dataset);
            return 1;
        }
    }

    llm_backend *backend = NULL;
    lm_model *model = NULL;
    lm_trainer *trainer = NULL;
    llm_status status =
        use_metal != 0 ? llm_backend_metal_create(&backend) : llm_backend_cpu_create(&backend);
    lm_model_config model_config = {.vocabulary_size = lm_dataset_model_vocabulary_size(dataset),
                                    .context_length = trainer_config.context_length,
                                    .hidden_size = hidden_size,
                                    .layer_count = layer_count,
                                    .head_count = layer_count == 0U ? 0U : head_count,
                                    .feed_forward_size = layer_count == 0U ? 0U : feed_forward_size,
                                    .seed = trainer_config.seed};
    if (status == LLM_OK && resume_path != NULL) {
        status = lm_trainer_load_checkpoint(backend, dataset, resume_path, &model, &trainer);
    } else if (status == LLM_OK) {
        status = lm_model_create(backend, &model_config, &model);
    }
    if (status == LLM_OK && resume_path == NULL) {
        status = lm_trainer_create(model, dataset, &trainer_config, &trainer);
    }
    if (status == LLM_OK) {
        status = lm_model_get_config(model, &model_config);
    }
    if (status == LLM_OK) {
        status = lm_trainer_get_config(trainer, &trainer_config);
    }
    if (status != LLM_OK) {
        fprintf(stderr, "Creating model training failed: %s\n", llm_status_string(status));
        lm_trainer_destroy(trainer);
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_dataset_close(validation_dataset);
        lm_dataset_close(dataset);
        return 1;
    }
    if (validation_dataset != NULL &&
        lm_dataset_model_vocabulary_size(validation_dataset) != model_config.vocabulary_size) {
        fprintf(stderr, "Training and validation vocabularies do not match.\n");
        lm_trainer_destroy(trainer);
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_dataset_close(validation_dataset);
        lm_dataset_close(dataset);
        return 1;
    }
    uint64_t parameter_value_count = 0U;
    for (size_t index = 0U; index < lm_model_parameter_count(model); ++index) {
        const llm_tensor *parameter = lm_model_parameter_value(model, index);
        if (parameter == NULL || parameter->element_count > UINT64_MAX - parameter_value_count) {
            fprintf(stderr, "Counting model parameters failed.\n");
            lm_trainer_destroy(trainer);
            lm_model_destroy(model);
            llm_backend_destroy(backend);
            lm_dataset_close(validation_dataset);
            lm_dataset_close(dataset);
            return 1;
        }
        parameter_value_count += parameter->element_count;
    }
    fprintf(stderr,
            "Modello: %zu layer, %zu head, hidden %zu, FFN %zu, contesto %zu, "
            "%.2fM parametri.\n",
            model_config.layer_count, model_config.head_count, model_config.hidden_size,
            model_config.feed_forward_size, model_config.context_length,
            (double)parameter_value_count / 1000000.0);

    FILE *log_file = NULL;
    if (log_path != NULL) {
        log_file = fopen(log_path, "a");
        if (log_file == NULL) {
            fprintf(stderr, "Opening training log failed.\n");
            lm_trainer_destroy(trainer);
            lm_model_destroy(model);
            llm_backend_destroy(backend);
            lm_dataset_close(validation_dataset);
            lm_dataset_close(dataset);
            return 1;
        }
        (void)setvbuf(log_file, NULL, _IOLBF, 0U);
        fprintf(log_file,
                "{\"schema\":\"llm-lab-training-event-v1\",\"event\":\"run\","
                "\"start_step\":%llu,\"requested_updates\":%zu,\"parameter_count\":%" PRIu64
                ",\"backend\":\"%s\",\"sampling\":\"%s\",\"batch_size\":%zu,"
                "\"context_length\":%zu,\"gradient_accumulation\":%zu,"
                "\"validation_every\":%zu,\"validation_batches\":%zu}\n",
                lm_trainer_step_count(trainer), steps, parameter_value_count,
                use_metal != 0 ? "metal" : "cpu", model_sampling_name(trainer_config.sampling),
                trainer_config.batch_size, trainer_config.context_length,
                trainer_config.gradient_accumulation_steps, validation_every, validation_batches);
    }

    if (trainer_config.batch_size > SIZE_MAX / trainer_config.context_length ||
        trainer_config.batch_size * trainer_config.context_length >
            SIZE_MAX / trainer_config.gradient_accumulation_steps) {
        fprintf(stderr, "Counting tokens per update failed.\n");
        if (log_file != NULL)
            (void)fclose(log_file);
        lm_trainer_destroy(trainer);
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_dataset_close(validation_dataset);
        lm_dataset_close(dataset);
        return 1;
    }
    const size_t tokens_per_update = trainer_config.batch_size * trainer_config.context_length *
                                     trainer_config.gradient_accumulation_steps;
    double best_validation_loss = INFINITY;
    if (best_checkpoint_path != NULL)
        (void)load_best_validation_loss(best_checkpoint_path, &best_validation_loss);
    double latest_validation_loss = INFINITY;
    int has_validation_result = 0;
    int saved_best_checkpoint = 0;
    unsigned long long last_checkpoint_step = ULLONG_MAX;
    install_model_training_signals();

    cli_model_training_progress training_progress = {
        .started_at = current_time_seconds(),
        .interactive = standard_error_is_terminal(),
    };
    float loss = 0.0F;
    show_model_training_progress(0U, steps, loss, &training_progress);
    for (size_t index = 0U; index < steps; ++index) {
        const double step_started_at = current_time_seconds();
        status = lm_trainer_step(trainer, &loss);
        if (status != LLM_OK) {
            finish_model_training_progress(&training_progress);
            fprintf(stderr, "Model training failed at step %zu: %s\n", index + 1U,
                    llm_status_string(status));
            lm_trainer_destroy(trainer);
            lm_model_destroy(model);
            llm_backend_destroy(backend);
            if (log_file != NULL)
                (void)fclose(log_file);
            lm_dataset_close(validation_dataset);
            lm_dataset_close(dataset);
            return 1;
        }
        const double step_seconds = current_time_seconds() - step_started_at;
        const unsigned long long global_step = lm_trainer_step_count(trainer);
        const double tokens_per_second =
            step_seconds > 0.0 ? (double)tokens_per_update / step_seconds : 0.0;
        llm_metal_backend_metrics metal_metrics = {0};
        if (use_metal != 0)
            (void)llm_backend_metal_get_metrics(backend, &metal_metrics);
        const uint64_t tokens_seen = global_step <= UINT64_MAX / (uint64_t)tokens_per_update
                                         ? (uint64_t)global_step * (uint64_t)tokens_per_update
                                         : UINT64_MAX;
        if (log_file != NULL) {
            fprintf(log_file,
                    "{\"schema\":\"llm-lab-training-event-v1\",\"event\":\"train\","
                    "\"step\":%llu,\"tokens\":%" PRIu64 ",\"loss\":%.9g,"
                    "\"learning_rate\":%.9g,\"gradient_norm\":%.9g,"
                    "\"step_seconds\":%.9g,\"steps_per_second\":%.9g,"
                    "\"tokens_per_second\":%.9g,"
                    "\"metal_active_bytes\":%zu,\"metal_peak_active_bytes\":%zu,"
                    "\"metal_total_gpu_seconds\":%.9g}\n",
                    global_step, tokens_seen, loss, lm_trainer_learning_rate(trainer),
                    lm_trainer_gradient_norm(trainer), step_seconds,
                    step_seconds > 0.0 ? 1.0 / step_seconds : 0.0, tokens_per_second,
                    metal_metrics.active_buffer_bytes, metal_metrics.peak_active_buffer_bytes,
                    metal_metrics.total_gpu_seconds);
        }
        const int final_requested_step = index + 1U == steps || model_training_interrupted != 0;
        const int should_validate =
            validation_dataset != NULL &&
            (global_step % (unsigned long long)validation_every == 0U || final_requested_step != 0);
        if (should_validate != 0) {
            finish_model_training_progress(&training_progress);
            fprintf(stderr, "Validation step %llu: %zu batch...\n", global_step,
                    validation_batches);
            float validation_loss = 0.0F;
            status = lm_model_evaluate_validation(model, validation_dataset,
                                                  trainer_config.batch_size, validation_batches,
                                                  trainer_config.seed, &validation_loss);
            if (status != LLM_OK) {
                fprintf(stderr, "Validation failed: %s\n", llm_status_string(status));
                if (log_file != NULL)
                    (void)fclose(log_file);
                lm_trainer_destroy(trainer);
                lm_model_destroy(model);
                llm_backend_destroy(backend);
                lm_dataset_close(validation_dataset);
                lm_dataset_close(dataset);
                return 1;
            }
            latest_validation_loss = validation_loss;
            has_validation_result = 1;
            const int improved = latest_validation_loss < best_validation_loss;
            if (improved != 0 && best_checkpoint_path != NULL) {
                status = lm_trainer_save_checkpoint(trainer, dataset, best_checkpoint_path);
                if (status == LLM_OK &&
                    save_best_validation_metadata(best_checkpoint_path, global_step,
                                                  latest_validation_loss) == 0) {
                    status = LLM_BACKEND_ERROR;
                }
                if (status != LLM_OK) {
                    fprintf(stderr, "Saving best checkpoint failed: %s\n",
                            llm_status_string(status));
                    if (log_file != NULL)
                        (void)fclose(log_file);
                    lm_trainer_destroy(trainer);
                    lm_model_destroy(model);
                    llm_backend_destroy(backend);
                    lm_dataset_close(validation_dataset);
                    lm_dataset_close(dataset);
                    return 1;
                }
            }
            if (improved != 0 && best_checkpoint_path != NULL)
                saved_best_checkpoint = 1;
            if (improved != 0)
                best_validation_loss = latest_validation_loss;
            fprintf(stderr, "Validation: loss %.6f, perplexity %.3f%s\n", latest_validation_loss,
                    exp(latest_validation_loss), improved != 0 ? ", nuovo best" : "");
            if (log_file != NULL) {
                fprintf(log_file,
                        "{\"schema\":\"llm-lab-training-event-v1\","
                        "\"event\":\"validation\",\"step\":%llu,\"batches\":%zu,"
                        "\"loss\":%.9g,\"perplexity\":%.9g,\"best_loss\":%.9g,"
                        "\"improved\":%s}\n",
                        global_step, validation_batches, latest_validation_loss,
                        exp(latest_validation_loss), best_validation_loss,
                        improved != 0 ? "true" : "false");
            }
            training_progress.last_update_at = 0.0;
        }
        if (checkpoint_every != 0U && global_step % (unsigned long long)checkpoint_every == 0U) {
            status = lm_trainer_save_checkpoint(trainer, dataset, checkpoint_path);
            if (status != LLM_OK) {
                finish_model_training_progress(&training_progress);
                fprintf(stderr, "Saving periodic checkpoint failed: %s\n",
                        llm_status_string(status));
                lm_trainer_destroy(trainer);
                lm_model_destroy(model);
                llm_backend_destroy(backend);
                if (log_file != NULL)
                    (void)fclose(log_file);
                lm_dataset_close(validation_dataset);
                lm_dataset_close(dataset);
                return 1;
            }
            last_checkpoint_step = global_step;
        }
        show_model_training_progress(index + 1U, steps, loss, &training_progress);
        if (model_training_interrupted != 0) {
            finish_model_training_progress(&training_progress);
            fprintf(stderr, "Interruzione richiesta: chiusura dopo lo step %llu.\n", global_step);
            if (log_file != NULL) {
                fprintf(log_file,
                        "{\"schema\":\"llm-lab-training-event-v1\","
                        "\"event\":\"interrupted\",\"step\":%llu}\n",
                        global_step);
            }
            break;
        }
    }
    finish_model_training_progress(&training_progress);
    if (checkpoint_path != NULL && last_checkpoint_step != lm_trainer_step_count(trainer)) {
        status = lm_trainer_save_checkpoint(trainer, dataset, checkpoint_path);
        if (status != LLM_OK) {
            fprintf(stderr, "Saving checkpoint failed: %s\n", llm_status_string(status));
            lm_trainer_destroy(trainer);
            lm_model_destroy(model);
            llm_backend_destroy(backend);
            if (log_file != NULL)
                (void)fclose(log_file);
            lm_dataset_close(validation_dataset);
            lm_dataset_close(dataset);
            return 1;
        }
    }
    printf("{\"schema\":\"llm-lab-model-training-v1\",\"steps\":%llu,"
           "\"loss\":%.8f,\"vocabulary_size\":%" PRIu32 ","
           "\"context_length\":%zu,\"hidden_size\":%zu,\"layer_count\":%zu,"
           "\"head_count\":%zu,\"feed_forward_size\":%zu,\"parameter_count\":%" PRIu64 ","
           "\"gradient_accumulation_steps\":%zu,\"learning_rate\":%.9g,"
           "\"gradient_norm\":%.9g,\"sampling\":\"%s\","
           "\"backend\":\"%s\",\"validation_loss\":",
           lm_trainer_step_count(trainer), loss, model_config.vocabulary_size,
           model_config.context_length, model_config.hidden_size, model_config.layer_count,
           model_config.head_count, model_config.feed_forward_size, parameter_value_count,
           trainer_config.gradient_accumulation_steps, lm_trainer_learning_rate(trainer),
           lm_trainer_gradient_norm(trainer), model_sampling_name(trainer_config.sampling),
           use_metal != 0 ? "metal" : "cpu");
    if (has_validation_result != 0)
        printf("%.8f,\"validation_perplexity\":%.8f,", latest_validation_loss,
               exp(latest_validation_loss));
    else
        printf("null,\"validation_perplexity\":null,");
    printf("\"checkpoint_saved\":%s,\"best_checkpoint_saved\":%s,\"interrupted\":%s}\n",
           checkpoint_path == NULL ? "false" : "true",
           saved_best_checkpoint != 0 ? "true" : "false",
           model_training_interrupted != 0 ? "true" : "false");
    if (log_file != NULL && fclose(log_file) != 0)
        fprintf(stderr, "Closing training log failed.\n");
    lm_trainer_destroy(trainer);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    lm_dataset_close(validation_dataset);
    lm_dataset_close(dataset);
    return 0;
}

static size_t valid_utf8_sequence_length(const unsigned char *bytes, size_t length) {
    if (length == 0U || bytes[0] < 0x80U) {
        return length == 0U ? 0U : 1U;
    }
    if (bytes[0] >= 0xc2U && bytes[0] <= 0xdfU && length >= 2U && bytes[1] >= 0x80U &&
        bytes[1] <= 0xbfU) {
        return 2U;
    }
    if (length >= 3U &&
        ((bytes[0] == 0xe0U && bytes[1] >= 0xa0U && bytes[1] <= 0xbfU) ||
         ((bytes[0] >= 0xe1U && bytes[0] <= 0xecU) && bytes[1] >= 0x80U && bytes[1] <= 0xbfU) ||
         (bytes[0] == 0xedU && bytes[1] >= 0x80U && bytes[1] <= 0x9fU) ||
         ((bytes[0] >= 0xeeU && bytes[0] <= 0xefU) && bytes[1] >= 0x80U && bytes[1] <= 0xbfU)) &&
        bytes[2] >= 0x80U && bytes[2] <= 0xbfU) {
        return 3U;
    }
    if (length >= 4U &&
        ((bytes[0] == 0xf0U && bytes[1] >= 0x90U && bytes[1] <= 0xbfU) ||
         ((bytes[0] >= 0xf1U && bytes[0] <= 0xf3U) && bytes[1] >= 0x80U && bytes[1] <= 0xbfU) ||
         (bytes[0] == 0xf4U && bytes[1] >= 0x80U && bytes[1] <= 0x8fU)) &&
        bytes[2] >= 0x80U && bytes[2] <= 0xbfU && bytes[3] >= 0x80U && bytes[3] <= 0xbfU) {
        return 4U;
    }
    return 0U;
}

static void write_generation_bytes(const unsigned char *bytes, size_t length) {
    for (size_t index = 0U; index < length; ++index) {
        const unsigned char value = bytes[index];
        if (value >= 32U && value <= 126U) {
            fputc(value, stdout);
        } else if (value == '\n') {
            fputs("\\n", stdout);
        } else if (value == '\r') {
            fputs("\\r", stdout);
        } else if (value == '\t') {
            fputs("\\t", stdout);
        } else if (value >= 0x80U) {
            const size_t sequence_length =
                valid_utf8_sequence_length(bytes + index, length - index);
            if (sequence_length != 0U) {
                (void)fwrite(bytes + index, 1U, sequence_length, stdout);
                index += sequence_length - 1U;
            } else {
                fprintf(stdout, "\\x%02x", value);
            }
        } else {
            fprintf(stdout, "\\x%02x", value);
        }
    }
}

static int run_model_generate(int argc, char **argv) {
    size_t generated_count = 0U;
    if (parse_positive_size(argv[5], &generated_count) == 0) {
        fprintf(stderr, "TOKENS must be a positive integer supported by this system.\n");
        return 1;
    }
    cli_generation_options options = {
        .temperature = 0.8F, .repetition_penalty = 1.1F, .top_k = 40U, .random_state = UINT64_C(1)};
    for (int index = 7; index < argc; index += 2) {
        if (index + 1 >= argc) {
            fprintf(stderr, "Model generation options require a value.\n");
            return 1;
        }
        const char *option = argv[index];
        const char *value = argv[index + 1];
        int parsed = 0;
        if (strcmp(option, "--temperature") == 0) {
            parsed = parse_positive_float(value, &options.temperature);
        } else if (strcmp(option, "--top-k") == 0) {
            parsed = parse_positive_size(value, &options.top_k);
        } else if (strcmp(option, "--repetition-penalty") == 0) {
            parsed = parse_positive_float(value, &options.repetition_penalty) &&
                     options.repetition_penalty >= 1.0F;
        } else if (strcmp(option, "--seed") == 0) {
            parsed = parse_seed(value, &options.random_state);
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid model generation option: %s %s\n", option, value);
            return 1;
        }
    }
    if (options.random_state == 0U) {
        options.random_state = UINT64_C(0x9e3779b97f4a7c15);
    }
    tokenizer *tokenizer = NULL;
    tokenizer_status tokenizer_result = tokenizer_load(argv[4], &tokenizer);
    if (tokenizer_result != TOKENIZER_OK) {
        fprintf(stderr, "Loading tokenizer failed: %s\n",
                tokenizer_status_string(tokenizer_result));
        return 1;
    }
    token_sequence sequence = {0};
    tokenizer_result =
        tokenizer_encode(tokenizer, (const unsigned char *)argv[6], strlen(argv[6]), &sequence);
    if (tokenizer_result != TOKENIZER_OK || sequence.length == 0U ||
        sequence.length > SIZE_MAX - generated_count) {
        fprintf(stderr, "Encoding prompt failed: %s\n", tokenizer_status_string(tokenizer_result));
        token_sequence_destroy(&sequence);
        tokenizer_destroy(tokenizer);
        return 1;
    }
    token_id *all_tokens =
        realloc(sequence.ids, (sequence.length + generated_count) * sizeof(*all_tokens));
    if (all_tokens == NULL) {
        fprintf(stderr, "Generating text failed: allocation failed\n");
        token_sequence_destroy(&sequence);
        tokenizer_destroy(tokenizer);
        return 1;
    }
    sequence.ids = all_tokens;

    llm_backend *backend = NULL;
    lm_model *model = NULL;
    llm_status status = llm_backend_cpu_create(&backend);
    if (status == LLM_OK) {
        status = lm_trainer_load_checkpoint(backend, NULL, argv[3], &model, NULL);
    }
    lm_model_config config = {0};
    if (status == LLM_OK) {
        status = lm_model_get_config(model, &config);
    }
    const uint32_t tokenizer_size = tokenizer_vocabulary_size(tokenizer);
    if (status != LLM_OK || tokenizer_size == UINT32_MAX ||
        config.vocabulary_size < tokenizer_size + 1U) {
        fprintf(stderr, "Loading checkpoint failed: %s\n",
                status == LLM_OK ? "tokenizer vocabulary does not match checkpoint"
                                 : llm_status_string(status));
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        token_sequence_destroy(&sequence);
        tokenizer_destroy(tokenizer);
        return 1;
    }

    const size_t input_shape[] = {1U, config.context_length};
    const size_t logits_shape[] = {config.context_length, config.vocabulary_size};
    llm_tensor input_ids = {0};
    llm_tensor logits = {0};
    status = llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &input_ids);
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits);
    }
    float *host_logits = NULL;
    if (status == LLM_OK && logits.element_count <= SIZE_MAX / sizeof(*host_logits)) {
        host_logits = malloc(logits.element_count * sizeof(*host_logits));
        status = host_logits == NULL ? LLM_ALLOCATION_FAILED : LLM_OK;
    } else if (status == LLM_OK) {
        status = LLM_OVERFLOW;
    }
    cli_generation_candidate *candidates = NULL;
    unsigned char *recent_tokens = NULL;
    if (status == LLM_OK && tokenizer_size > 1U) {
        candidates = malloc(((size_t)tokenizer_size - 1U) * sizeof(*candidates));
        recent_tokens = calloc(tokenizer_size, sizeof(*recent_tokens));
        status = candidates == NULL || recent_tokens == NULL ? LLM_ALLOCATION_FAILED : LLM_OK;
    } else if (status == LLM_OK) {
        status = LLM_INVALID_SHAPE;
    }
    for (size_t generated = 0U; status == LLM_OK && generated < generated_count; ++generated) {
        token_id context[LLM_TENSOR_MAX_RANK == 4U ? config.context_length : 1U];
        const size_t available = sequence.length + generated;
        const size_t used = available < config.context_length ? available : config.context_length;
        const size_t first = available - used;
        for (size_t position = 0U; position < used; ++position) {
            context[position] = sequence.ids[first + position];
        }
        for (size_t position = used; position < config.context_length; ++position) {
            context[position] = tokenizer_size;
        }
        status = llm_tensor_write(backend, &input_ids, context, sizeof(context));
        if (status == LLM_OK) {
            status = lm_model_forward(model, &input_ids, &logits);
        }
        if (status == LLM_OK) {
            status = llm_tensor_read(backend, &logits, host_logits,
                                     logits.element_count * sizeof(*host_logits));
        }
        if (status == LLM_OK) {
            const float *row = host_logits + (used - 1U) * config.vocabulary_size;
            sequence.ids[available] =
                sample_generation_token(row, tokenizer_size, sequence.ids, available,
                                        config.context_length, candidates, recent_tokens, &options);
        }
    }
    if (status == LLM_OK) {
        sequence.length += generated_count;
        unsigned char *output = NULL;
        size_t output_length = 0U;
        tokenizer_result = tokenizer_decode(tokenizer, &sequence, &output, &output_length);
        if (tokenizer_result == TOKENIZER_OK) {
            write_generation_bytes(output, output_length);
            fputc('\n', stdout);
        } else {
            status = LLM_BACKEND_ERROR;
        }
        tokenizer_bytes_destroy(output);
    }
    if (status != LLM_OK) {
        fprintf(stderr, "Generating text failed: %s\n", llm_status_string(status));
    }
    free(recent_tokens);
    free(candidates);
    free(host_logits);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&input_ids);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    token_sequence_destroy(&sequence);
    tokenizer_destroy(tokenizer);
    return status == LLM_OK ? 0 : 1;
}

static int run_model_evaluate(int argc, char **argv) {
    size_t steps = 0U;
    size_t batch_size = 2U;
    uint64_t seed = UINT64_C(1);
    if (parse_positive_size(argv[5], &steps) == 0) {
        fprintf(stderr, "STEPS must be a positive integer supported by this system.\n");
        return 1;
    }
    for (int index = 6; index < argc; index += 2) {
        if (index + 1 >= argc) {
            fprintf(stderr, "Model evaluation options require a value.\n");
            return 1;
        }
        int parsed = 0;
        if (strcmp(argv[index], "--batch-size") == 0) {
            parsed = parse_positive_size(argv[index + 1], &batch_size);
        } else if (strcmp(argv[index], "--seed") == 0) {
            parsed = parse_seed(argv[index + 1], &seed);
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid model evaluation option: %s %s\n", argv[index],
                    argv[index + 1]);
            return 1;
        }
    }
    cli_model_dataset_open_progress dataset_progress = {
        .started_at = current_time_seconds(),
        .interactive = standard_error_is_terminal(),
    };
    lm_dataset *dataset = NULL;
    const lm_dataset_status dataset_status = lm_dataset_open_with_progress(
        argv[3], show_model_dataset_open_progress, &dataset_progress, &dataset);
    finish_model_dataset_open_progress(&dataset_progress);
    if (dataset_status != LM_DATASET_OK || lm_dataset_get_split(dataset) != LM_DATASET_VALIDATION) {
        fprintf(stderr, "Model evaluation requires a validation split artifact.\n");
        lm_dataset_close(dataset);
        return 1;
    }
    llm_backend *backend = NULL;
    lm_model *model = NULL;
    llm_status status = llm_backend_cpu_create(&backend);
    if (status == LLM_OK) {
        status = lm_trainer_load_checkpoint(backend, NULL, argv[4], &model, NULL);
    }
    lm_model_config config = {0};
    if (status == LLM_OK) {
        status = lm_model_get_config(model, &config);
    }
    if (status == LLM_OK && config.vocabulary_size != lm_dataset_model_vocabulary_size(dataset)) {
        status = LLM_INVALID_SHAPE;
    }
    if (status != LLM_OK || batch_size > SIZE_MAX / config.context_length) {
        fprintf(stderr, "Loading evaluation model failed: %s\n", llm_status_string(status));
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_dataset_close(dataset);
        return 1;
    }
    const size_t token_count = batch_size * config.context_length;
    token_id *host_inputs = malloc(token_count * sizeof(*host_inputs));
    token_id *host_targets = malloc(token_count * sizeof(*host_targets));
    const size_t input_shape[] = {batch_size, config.context_length};
    const size_t target_shape[] = {token_count};
    const size_t logits_shape[] = {token_count, config.vocabulary_size};
    llm_tensor inputs = {0};
    llm_tensor targets = {0};
    llm_tensor logits = {0};
    llm_tensor loss = {0};
    lm_batcher *batcher = NULL;
    if (host_inputs == NULL || host_targets == NULL) {
        status = LLM_ALLOCATION_FAILED;
    }
    if (status == LLM_OK && lm_batcher_create(dataset, batch_size, config.context_length, seed,
                                              &batcher) != LM_DATASET_OK) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &inputs);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &targets);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits);
    }
    if (status == LLM_OK) {
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss);
    }
    double loss_sum = 0.0;
    fprintf(stderr, "Valutazione modello: %zu batch...\n", steps);
    for (size_t index = 0U; status == LLM_OK && index < steps; ++index) {
        if (lm_batcher_next(batcher, host_inputs, host_targets) != LM_DATASET_OK) {
            status = LLM_BACKEND_ERROR;
            break;
        }
        status =
            llm_tensor_write(backend, &inputs, host_inputs, token_count * sizeof(*host_inputs));
        if (status == LLM_OK) {
            status = llm_tensor_write(backend, &targets, host_targets,
                                      token_count * sizeof(*host_targets));
        }
        if (status == LLM_OK) {
            status = lm_model_forward(model, &inputs, &logits);
        }
        if (status == LLM_OK) {
            status = llm_cross_entropy_forward(backend, &logits, &targets, &loss);
        }
        float batch_loss = 0.0F;
        if (status == LLM_OK) {
            status = llm_tensor_read(backend, &loss, &batch_loss, sizeof(batch_loss));
        }
        if (status == LLM_OK) {
            loss_sum += batch_loss;
        }
    }
    if (status == LLM_OK) {
        const double mean_loss = loss_sum / (double)steps;
        printf("{\"schema\":\"llm-lab-model-evaluation-v1\",\"batches\":%zu,"
               "\"loss\":%.8f,\"perplexity\":%.8f,\"layer_count\":%zu}\n",
               steps, mean_loss, exp(mean_loss), config.layer_count);
    } else {
        fprintf(stderr, "Model evaluation failed: %s\n", llm_status_string(status));
    }
    lm_batcher_destroy(batcher);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&inputs);
    free(host_targets);
    free(host_inputs);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    lm_dataset_close(dataset);
    return status == LLM_OK ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "tokenizer") == 0 && strcmp(argv[2], "train") == 0) {
        return run_tokenizer_train(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "tokenizer") == 0 && strcmp(argv[2], "evaluate") == 0) {
        return run_tokenizer_evaluate(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "dataset") == 0 && strcmp(argv[2], "prepare") == 0) {
        return run_dataset_prepare(argc, argv);
    }
    if (argc >= 5 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "train") == 0) {
        return run_model_train(argc, argv);
    }
    if (argc >= 7 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "generate") == 0) {
        return run_model_generate(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "evaluate") == 0) {
        return run_model_evaluate(argc, argv);
    }

    print_usage(argv[0]);
    return 1;
}
