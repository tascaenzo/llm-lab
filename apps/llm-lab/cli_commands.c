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

#include "cli_commands.h"
#include "dataset/dataset.h"
#include "dataset/sft_dataset.h"
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

typedef struct cli_model_generation_progress {
    double started_at;
    double last_update_at;
    size_t next_log_token;
    int interactive;
    int has_output;
} cli_model_generation_progress;

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

typedef struct cli_next_token_diagnostics {
    uint64_t token_count;
    uint64_t regular_token_count;
    uint64_t end_of_document_count;
    uint64_t top_1_count;
    uint64_t top_5_count;
    uint64_t top_20_count;
    uint64_t top_100_count;
    uint64_t regular_top_1_count;
    uint64_t regular_top_5_count;
    uint64_t regular_top_20_count;
    uint64_t regular_top_100_count;
    double reciprocal_rank_sum;
    double rank_sum;
    double loss_sum;
} cli_next_token_diagnostics;

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

static void show_model_training_progress(size_t completed, size_t total,
                                         unsigned long long global_step,
                                         uint64_t total_training_steps, float loss, void *context) {
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
    char total_progress[96] = {0};
    if (total_training_steps != 0U) {
        const double overall_fraction = (double)global_step / (double)total_training_steps;
        (void)snprintf(total_progress, sizeof(total_progress),
                       " | totale %.2f%% (step %llu/%" PRIu64 ")", overall_fraction * 100.0,
                       global_step, total_training_steps);
    }
    if (progress->interactive != 0) {
        const size_t bar_width = 26U;
        const size_t filled = (size_t)(fraction * (double)bar_width);
        fprintf(stderr, "\rTraining modello [");
        for (size_t index = 0U; index < bar_width; ++index) {
            fputc(index < filled ? '#' : '-', stderr);
        }
        if (completed == 0U) {
            fprintf(stderr, "] %5.1f%% step 0/%zu%s | preparazione...", fraction * 100.0, total,
                    total_progress);
        } else {
            fprintf(stderr, "] %5.1f%% step %zu/%zu%s | loss %.6f | %.2f step/s | ETA %s",
                    fraction * 100.0, completed, total, total_progress, loss, steps_per_second,
                    eta);
        }
        fflush(stderr);
    } else if (completed == 0U) {
        fprintf(stderr, "Training modello: preparazione di %zu step%s...\n", total, total_progress);
    } else {
        fprintf(stderr,
                "Training modello: %5.1f%% step %zu/%zu%s, loss %.6f, %.2f step/s, ETA %s\n",
                fraction * 100.0, completed, total, total_progress, loss, steps_per_second, eta);
    }
    progress->last_update_at = now;
    progress->has_output = 1;
}

static void finish_model_training_progress(const cli_model_training_progress *progress) {
    if (progress->interactive != 0 && progress->has_output != 0) {
        fputc('\n', stderr);
    }
}

static void show_model_generation_progress(size_t completed, size_t total, void *context) {
    cli_model_generation_progress *progress = context;
    const double now = current_time_seconds();
    const int is_complete = completed >= total;
    const double fraction = total == 0U ? 1.0 : (double)completed / (double)total;
    const double elapsed = now - progress->started_at;
    const double tokens_per_second = elapsed > 0.0 ? (double)completed / elapsed : 0.0;
    const double remaining_seconds = tokens_per_second > 0.0 && is_complete == 0
                                         ? (double)(total - completed) / tokens_per_second
                                         : 0.0;
    if (progress->interactive == 0) {
        const size_t interval = total < 20U ? 1U : total / 20U;
        if (completed != 0U && is_complete == 0 && completed < progress->next_log_token) {
            return;
        }
        progress->next_log_token =
            completed > SIZE_MAX - interval ? SIZE_MAX : completed + interval;
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
        fprintf(stderr, "\rGenerazione [");
        for (size_t index = 0U; index < bar_width; ++index) {
            fputc(index < filled ? '#' : '-', stderr);
        }
        if (completed == 0U) {
            fprintf(stderr, "] %5.1f%% token 0/%zu | preparazione...", fraction * 100.0, total);
        } else {
            fprintf(stderr, "] %5.1f%% token %zu/%zu | %.2f token/s | ETA %s", fraction * 100.0,
                    completed, total, tokens_per_second, eta);
        }
        fflush(stderr);
    } else if (completed == 0U) {
        fprintf(stderr, "Generazione: preparazione di %zu token...\n", total);
    } else {
        fprintf(stderr, "Generazione: %5.1f%% token %zu/%zu, %.2f token/s, ETA %s\n",
                fraction * 100.0, completed, total, tokens_per_second, eta);
    }
    progress->last_update_at = now;
    progress->has_output = 1;
}

static void finish_model_generation_progress(const cli_model_generation_progress *progress) {
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
            "  %s dataset sft-prepare MODEL.llmtok CONVERSATIONS.jsonl OUTPUT_PREFIX"
            " --context T\n"
            "  %s model train TRAIN.llmdat STEPS [--batch-size B] [--context T]"
            " [--hidden C] [--layers L] [--heads H] [--ffn F]"
            " [--learning-rate LR] [--gradient-accumulation N]"
            " [--warmup-steps N] [--total-steps N] [--min-learning-rate LR] [--seed N]"
            " [--beta1 B] [--beta2 B] [--epsilon E] [--weight-decay W]"
            " [--sampling shuffled|random] [--gradient-clip N]"
            " [--backend cpu|metal|cuda]"
            " [--checkpoint FILE] [--checkpoint-every N] [--resume FILE]"
            " [--validation FILE] [--validation-every N] [--validation-batches N]"
            " [--best-checkpoint FILE] [--log FILE]\n"
            "  %s model sft TRAIN.llmsft STEPS (--base FILE|--resume FILE)"
            " [--batch-size B] [--gradient-accumulation N] [--learning-rate LR]"
            " [--warmup-steps N] [--total-steps N] [--min-learning-rate LR]"
            " [--beta1 B] [--beta2 B] [--epsilon E] [--weight-decay W]"
            " [--gradient-clip N] [--seed N] [--backend cpu|metal|cuda]"
            " [--checkpoint FILE] [--checkpoint-every N] [--validation FILE]"
            " [--validation-every N] [--validation-batches N]"
            " [--best-checkpoint FILE] [--log FILE]\n"
            "  %s model generate CHECKPOINT.llmckpt TOKENIZER.llmtok TOKENS PROMPT"
            " [--temperature T] [--top-k K] [--repetition-penalty P] [--seed N]"
            " [--backend cpu|metal|cuda]\n"
            "  %s model chat CHECKPOINT.llmckpt TOKENIZER.llmtok TOKENS PROMPT"
            " [--system TEXT] [--temperature T] [--top-k K]"
            " [--repetition-penalty P] [--seed N] [--backend cpu|metal|cuda]\n"
            "  %s model evaluate VALIDATION.llmdat CHECKPOINT.llmckpt BATCHES"
            " [--batch-size B] [--seed N] [--backend cpu|metal|cuda]\n"
            "  %s model diagnose VALIDATION.llmdat CHECKPOINT.llmckpt TOKENIZER.llmtok BATCHES"
            " [--batch-size B] [--seed N] [--backend cpu|metal|cuda]\n",
            program, program, program, program, program, program, program, program, program,
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

static token_id sample_chat_token(const float *logits, uint32_t tokenizer_vocabulary_size,
                                  token_id end_token, const token_id *tokens, size_t token_count,
                                  size_t context_length, cli_generation_candidate *candidates,
                                  unsigned char *recent_tokens, cli_generation_options *options) {
    const size_t candidate_count = (size_t)tokenizer_vocabulary_size;
    (void)memset(recent_tokens, 0, tokenizer_vocabulary_size * sizeof(*recent_tokens));
    const size_t recent_start = token_count > context_length ? token_count - context_length : 0U;
    for (size_t index = recent_start; index < token_count; ++index) {
        if (tokens[index] < tokenizer_vocabulary_size) {
            recent_tokens[tokens[index]] = 1U;
        }
    }
    size_t candidate = 0U;
    for (uint32_t token = 1U; token < tokenizer_vocabulary_size; ++token) {
        float adjusted = logits[token];
        if (recent_tokens[token] != 0U) {
            adjusted = adjusted >= 0.0F ? adjusted / options->repetition_penalty
                                        : adjusted * options->repetition_penalty;
        }
        candidates[candidate++] = (cli_generation_candidate){.token = token, .logit = adjusted};
    }
    candidates[candidate++] =
        (cli_generation_candidate){.token = end_token, .logit = logits[end_token]};
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

static int run_sft_dataset_prepare(int argc, char **argv) {
    size_t context_length = 0U;
    for (int index = 6; index < argc; index += 2) {
        if (index + 1 >= argc || strcmp(argv[index], "--context") != 0 ||
            parse_positive_size(argv[index + 1], &context_length) == 0) {
            fprintf(stderr, "Invalid SFT dataset option: use --context T.\n");
            return 1;
        }
    }
    if (context_length == 0U) {
        fprintf(stderr, "SFT dataset preparation requires --context T.\n");
        return 1;
    }
    lm_sft_prepare_report report = {0};
    const lm_dataset_status status =
        lm_sft_dataset_prepare_jsonl(argv[3], argv[4], argv[5], context_length, &report);
    if (status != LM_DATASET_OK) {
        fprintf(stderr, "SFT dataset preparation failed: %s\n", lm_dataset_status_string(status));
        return 1;
    }
    printf("{\"schema\":\"llm-lab-sft-dataset-report-v1\","
           "\"tokenizer_vocabulary_size\":%" PRIu32 ","
           "\"model_vocabulary_size\":%" PRIu32 ",\"context_length\":%zu,"
           "\"protocol\":{\"system\":%" PRIu32 ",\"user\":%" PRIu32 ",\"assistant\":%" PRIu32
           ",\"end\":%" PRIu32 ",\"pad\":%" PRIu32 "},"
           "\"splits\":{\"train\":{\"examples\":%" PRIu64 ",\"supervised_tokens\":%" PRIu64 "},"
           "\"validation\":{\"examples\":%" PRIu64 ",\"supervised_tokens\":%" PRIu64 "},"
           "\"test\":{\"examples\":%" PRIu64 ",\"supervised_tokens\":%" PRIu64 "}}}\n",
           report.tokenizer_vocabulary_size, report.model_vocabulary_size, report.context_length,
           report.protocol.system_token, report.protocol.user_token,
           report.protocol.assistant_token, report.protocol.end_token,
           report.protocol.padding_token, report.example_counts[LM_DATASET_TRAIN],
           report.supervised_token_counts[LM_DATASET_TRAIN],
           report.example_counts[LM_DATASET_VALIDATION],
           report.supervised_token_counts[LM_DATASET_VALIDATION],
           report.example_counts[LM_DATASET_TEST], report.supervised_token_counts[LM_DATASET_TEST]);
    return 0;
}

static const char *model_sampling_name(lm_batcher_sampling sampling) {
    if (sampling == LM_BATCHER_SHUFFLED_BLOCKS)
        return "shuffled";
    if (sampling == LM_BATCHER_SHUFFLED_WINDOWS)
        return "shuffled-windows";
    return "random";
}

typedef enum cli_backend_choice {
    CLI_BACKEND_CPU = 0,
    CLI_BACKEND_METAL,
    CLI_BACKEND_CUDA
} cli_backend_choice;

/** Maps a --backend value to a choice. Returns 0 when the name is unknown. */
static int parse_backend_choice(const char *value, cli_backend_choice *out_choice) {
    if (strcmp(value, "cpu") == 0) {
        *out_choice = CLI_BACKEND_CPU;
        return 1;
    }
    if (strcmp(value, "metal") == 0) {
        *out_choice = CLI_BACKEND_METAL;
        return 1;
    }
    if (strcmp(value, "cuda") == 0) {
        *out_choice = CLI_BACKEND_CUDA;
        return 1;
    }
    return 0;
}

/** Resolves the default backend after options were parsed, so --backend can
 * deliberately override even an invalid environment setting. */
static int resolve_environment_backend(cli_backend_choice *out_choice) {
    const char *value = getenv("LLM_LAB_BACKEND");
    if (value == NULL || value[0] == '\0') {
        *out_choice = CLI_BACKEND_CPU;
        return 1;
    }
    if (parse_backend_choice(value, out_choice) != 0) {
        return 1;
    }
    fprintf(stderr,
            "Invalid LLM_LAB_BACKEND: %s (expected cpu, metal, or cuda); "
            "override it with --backend.\n",
            value);
    return 0;
}

static const char *backend_choice_name(cli_backend_choice choice) {
    if (choice == CLI_BACKEND_METAL)
        return "metal";
    if (choice == CLI_BACKEND_CUDA)
        return "cuda";
    return "cpu";
}

static llm_status create_backend_choice(cli_backend_choice choice, llm_backend **out_backend) {
    if (choice == CLI_BACKEND_METAL)
        return llm_backend_metal_create(out_backend);
    if (choice == CLI_BACKEND_CUDA)
        return llm_backend_cuda_create(out_backend);
    return llm_backend_cpu_create(out_backend);
}

static const char *backend_choice_device_name(cli_backend_choice choice,
                                              const llm_backend *backend) {
    if (choice == CLI_BACKEND_METAL) {
        const char *name = llm_backend_metal_device_name(backend);
        return name != NULL ? name : "Metal";
    }
    if (choice == CLI_BACKEND_CUDA) {
        const char *name = llm_backend_cuda_device_name(backend);
        return name != NULL ? name : "CUDA";
    }
    return "CPU";
}

/** Portable accelerator counters written to each training JSONL event. */
typedef struct cli_accelerator_metrics {
    size_t active_buffer_count;
    size_t active_buffer_bytes;
    size_t peak_active_buffer_bytes;
    size_t cached_buffer_count;
    size_t cached_buffer_bytes;
    unsigned long long dispatches;
    unsigned long long synchronizations;
    unsigned long long reused_buffer_allocations;
    double total_gpu_seconds;
    double last_gpu_seconds;
} cli_accelerator_metrics;

/** Reads the buffer-pool, synchronization and timing counters of an accelerator. */
static void read_accelerator_metrics(cli_backend_choice choice, const llm_backend *backend,
                                     cli_accelerator_metrics *out_metrics) {
    *out_metrics = (cli_accelerator_metrics){0};
    if (choice == CLI_BACKEND_METAL) {
        llm_metal_backend_metrics metrics = {0};
        if (llm_backend_metal_get_metrics(backend, &metrics) == LLM_OK) {
            *out_metrics = (cli_accelerator_metrics){
                .active_buffer_count = metrics.active_buffer_count,
                .active_buffer_bytes = metrics.active_buffer_bytes,
                .peak_active_buffer_bytes = metrics.peak_active_buffer_bytes,
                .cached_buffer_count = metrics.cached_buffer_count,
                .cached_buffer_bytes = metrics.cached_buffer_bytes,
                .dispatches = metrics.kernel_dispatches,
                .synchronizations = metrics.submitted_command_buffers,
                .reused_buffer_allocations = metrics.reused_buffer_allocations,
                .total_gpu_seconds = metrics.total_gpu_seconds,
                .last_gpu_seconds = metrics.last_gpu_seconds,
            };
        }
    } else if (choice == CLI_BACKEND_CUDA) {
        llm_cuda_backend_metrics metrics = {0};
        if (llm_backend_cuda_get_metrics(backend, &metrics) == LLM_OK) {
            *out_metrics = (cli_accelerator_metrics){
                .active_buffer_count = metrics.active_buffer_count,
                .active_buffer_bytes = metrics.active_buffer_bytes,
                .peak_active_buffer_bytes = metrics.peak_active_buffer_bytes,
                .cached_buffer_count = metrics.cached_buffer_count,
                .cached_buffer_bytes = metrics.cached_buffer_bytes,
                .dispatches = metrics.kernel_launches,
                .synchronizations = metrics.synchronizations,
                .reused_buffer_allocations = metrics.reused_buffer_allocations,
                .total_gpu_seconds = metrics.total_gpu_seconds,
                .last_gpu_seconds = metrics.last_gpu_seconds,
            };
        }
    }
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

static int is_resume_forbidden_model_option(const char *option) {
    return strcmp(option, "--context") == 0 || strcmp(option, "--hidden") == 0 ||
           strcmp(option, "--layers") == 0 || strcmp(option, "--heads") == 0 ||
           strcmp(option, "--ffn") == 0 || strcmp(option, "--learning-rate") == 0 ||
           strcmp(option, "--warmup-steps") == 0 || strcmp(option, "--total-steps") == 0 ||
           strcmp(option, "--min-learning-rate") == 0 || strcmp(option, "--beta1") == 0 ||
           strcmp(option, "--beta2") == 0 || strcmp(option, "--epsilon") == 0 ||
           strcmp(option, "--weight-decay") == 0 || strcmp(option, "--sampling") == 0 ||
           strcmp(option, "--gradient-clip") == 0 || strcmp(option, "--seed") == 0;
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
    cli_backend_choice backend_choice = CLI_BACKEND_CPU;
    int backend_was_explicit = 0;
    int batch_size_was_explicit = 0;
    int accumulation_was_explicit = 0;
    int has_resume_forbidden_model_options = 0;
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
            batch_size_was_explicit = parsed;
        } else if (strcmp(option, "--context") == 0) {
            parsed = parse_positive_size(value, &trainer_config.context_length);
        } else if (strcmp(option, "--hidden") == 0) {
            parsed = parse_positive_size(value, &hidden_size);
        } else if (strcmp(option, "--layers") == 0) {
            parsed = parse_size(value, &layer_count);
        } else if (strcmp(option, "--heads") == 0) {
            parsed = parse_positive_size(value, &head_count);
        } else if (strcmp(option, "--ffn") == 0) {
            parsed = parse_size(value, &feed_forward_size);
        } else if (strcmp(option, "--learning-rate") == 0) {
            parsed = parse_positive_float(value, &trainer_config.learning_rate);
        } else if (strcmp(option, "--gradient-accumulation") == 0) {
            parsed = parse_positive_size(value, &trainer_config.gradient_accumulation_steps);
            accumulation_was_explicit = parsed;
        } else if (strcmp(option, "--warmup-steps") == 0) {
            parsed = parse_seed(value, &trainer_config.warmup_steps);
        } else if (strcmp(option, "--total-steps") == 0) {
            parsed = parse_seed(value, &trainer_config.total_steps);
        } else if (strcmp(option, "--min-learning-rate") == 0) {
            parsed = parse_positive_float(value, &trainer_config.minimum_learning_rate);
        } else if (strcmp(option, "--beta1") == 0) {
            parsed = parse_positive_float(value, &trainer_config.beta1);
        } else if (strcmp(option, "--beta2") == 0) {
            parsed = parse_positive_float(value, &trainer_config.beta2);
        } else if (strcmp(option, "--epsilon") == 0) {
            parsed = parse_positive_float(value, &trainer_config.epsilon);
        } else if (strcmp(option, "--weight-decay") == 0) {
            parsed = parse_positive_float(value, &trainer_config.weight_decay);
        } else if (strcmp(option, "--sampling") == 0) {
            if (strcmp(value, "shuffled") == 0) {
                trainer_config.sampling = LM_BATCHER_SHUFFLED_BLOCKS;
                parsed = 1;
            } else if (strcmp(value, "random") == 0) {
                trainer_config.sampling = LM_BATCHER_RANDOM_WINDOWS;
                parsed = 1;
            }
        } else if (strcmp(option, "--gradient-clip") == 0) {
            parsed = parse_positive_float(value, &trainer_config.gradient_clip_norm);
        } else if (strcmp(option, "--seed") == 0) {
            parsed = parse_seed(value, &trainer_config.seed);
        } else if (strcmp(option, "--backend") == 0) {
            parsed = parse_backend_choice(value, &backend_choice);
            backend_was_explicit = parsed;
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
        if (is_resume_forbidden_model_option(option) != 0) {
            has_resume_forbidden_model_options = 1;
        }
    }
    if (backend_was_explicit == 0 && resolve_environment_backend(&backend_choice) == 0) {
        return 1;
    }
    if (resume_path != NULL && has_resume_forbidden_model_options != 0) {
        fprintf(stderr,
                "--resume restores model and trainer settings; only an equivalent --batch-size "
                "and --gradient-accumulation override is allowed.\n");
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
    llm_status status = create_backend_choice(backend_choice, &backend);
    lm_model_config model_config = {.vocabulary_size = lm_dataset_model_vocabulary_size(dataset),
                                    .context_length = trainer_config.context_length,
                                    .hidden_size = hidden_size,
                                    .layer_count = layer_count,
                                    .head_count = layer_count == 0U ? 0U : head_count,
                                    .feed_forward_size = layer_count == 0U ? 0U : feed_forward_size,
                                    .seed = trainer_config.seed};
    if (status == LLM_OK && resume_path != NULL) {
        const lm_trainer_resume_options resume_options = {
            .batch_size = batch_size_was_explicit != 0 ? trainer_config.batch_size : 0U,
            .gradient_accumulation_steps =
                accumulation_was_explicit != 0 ? trainer_config.gradient_accumulation_steps : 0U};
        status = lm_trainer_load_checkpoint_with_options(backend, dataset, resume_path,
                                                         &resume_options, &model, &trainer);
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
                backend_choice_name(backend_choice), model_sampling_name(trainer_config.sampling),
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
    show_model_training_progress(0U, steps, lm_trainer_step_count(trainer),
                                 trainer_config.total_steps, loss, &training_progress);
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
        cli_accelerator_metrics accelerator_metrics = {0};
        read_accelerator_metrics(backend_choice, backend, &accelerator_metrics);
        const uint64_t tokens_seen = global_step <= UINT64_MAX / (uint64_t)tokens_per_update
                                         ? (uint64_t)global_step * (uint64_t)tokens_per_update
                                         : UINT64_MAX;
        if (log_file != NULL) {
            fprintf(
                log_file,
                "{\"schema\":\"llm-lab-training-event-v1\",\"event\":\"train\","
                "\"step\":%llu,\"tokens\":%" PRIu64 ",\"loss\":%.9g,"
                "\"learning_rate\":%.9g,\"gradient_norm\":%.9g,"
                "\"step_seconds\":%.9g,\"steps_per_second\":%.9g,"
                "\"tokens_per_second\":%.9g,"
                "\"accelerator_backend\":\"%s\","
                "\"accelerator_device\":\"%s\","
                "\"accelerator_active_buffer_count\":%zu,"
                "\"accelerator_active_bytes\":%zu,"
                "\"accelerator_peak_active_bytes\":%zu,"
                "\"accelerator_cached_buffer_count\":%zu,"
                "\"accelerator_cached_bytes\":%zu,"
                "\"accelerator_dispatches\":%llu,"
                "\"accelerator_synchronizations\":%llu,"
                "\"accelerator_reused_buffer_allocations\":%llu,"
                "\"accelerator_total_gpu_seconds\":%.9g,"
                "\"accelerator_last_gpu_seconds\":%.9g,"
                "\"metal_active_bytes\":%zu,\"metal_peak_active_bytes\":%zu,"
                "\"metal_total_gpu_seconds\":%.9g}\n",
                global_step, tokens_seen, loss, lm_trainer_learning_rate(trainer),
                lm_trainer_gradient_norm(trainer), step_seconds,
                step_seconds > 0.0 ? 1.0 / step_seconds : 0.0, tokens_per_second,
                backend_choice_name(backend_choice),
                backend_choice_device_name(backend_choice, backend),
                accelerator_metrics.active_buffer_count, accelerator_metrics.active_buffer_bytes,
                accelerator_metrics.peak_active_buffer_bytes,
                accelerator_metrics.cached_buffer_count, accelerator_metrics.cached_buffer_bytes,
                accelerator_metrics.dispatches, accelerator_metrics.synchronizations,
                accelerator_metrics.reused_buffer_allocations,
                accelerator_metrics.total_gpu_seconds, accelerator_metrics.last_gpu_seconds,
                accelerator_metrics.active_buffer_bytes,
                accelerator_metrics.peak_active_buffer_bytes,
                accelerator_metrics.total_gpu_seconds);
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
        show_model_training_progress(index + 1U, steps, global_step, trainer_config.total_steps,
                                     loss, &training_progress);
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
           backend_choice_name(backend_choice));
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

static int run_model_sft(int argc, char **argv) {
    size_t requested_steps = 0U;
    if (parse_positive_size(argv[4], &requested_steps) == 0) {
        fprintf(stderr, "STEPS must be a positive integer supported by this system.\n");
        return 1;
    }
    lm_trainer_config config = {.batch_size = 2U,
                                .context_length = 0U,
                                .seed = UINT64_C(1),
                                .learning_rate = 3.0e-5F,
                                .beta1 = 0.9F,
                                .beta2 = 0.95F,
                                .epsilon = 1.0e-8F,
                                .weight_decay = 0.01F,
                                .gradient_accumulation_steps = 1U,
                                .warmup_steps = 0U,
                                .total_steps = 0U,
                                .minimum_learning_rate = 3.0e-6F,
                                .sampling = LM_BATCHER_SHUFFLED_BLOCKS,
                                .gradient_clip_norm = 1.0F};
    const char *base_path = NULL;
    const char *resume_path = NULL;
    const char *checkpoint_path = NULL;
    const char *validation_path = NULL;
    const char *best_checkpoint_path = NULL;
    const char *log_path = NULL;
    size_t checkpoint_every = 0U;
    size_t validation_every = 0U;
    size_t validation_batches = 100U;
    cli_backend_choice backend_choice = CLI_BACKEND_CPU;
    int backend_was_explicit = 0;
    int batch_was_explicit = 0;
    int accumulation_was_explicit = 0;
    int resume_forbidden_option = 0;
    for (int index = 5; index < argc; index += 2) {
        if (index + 1 >= argc) {
            fprintf(stderr, "SFT options require a value.\n");
            return 1;
        }
        const char *option = argv[index];
        const char *value = argv[index + 1];
        int parsed = 0;
        int changes_training = 0;
        if (strcmp(option, "--base") == 0 && base_path == NULL) {
            base_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--resume") == 0 && resume_path == NULL) {
            resume_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--batch-size") == 0) {
            parsed = parse_positive_size(value, &config.batch_size);
            batch_was_explicit = parsed;
        } else if (strcmp(option, "--gradient-accumulation") == 0) {
            parsed = parse_positive_size(value, &config.gradient_accumulation_steps);
            accumulation_was_explicit = parsed;
        } else if (strcmp(option, "--learning-rate") == 0) {
            parsed = parse_positive_float(value, &config.learning_rate);
            changes_training = 1;
        } else if (strcmp(option, "--warmup-steps") == 0) {
            parsed = parse_seed(value, &config.warmup_steps);
            changes_training = 1;
        } else if (strcmp(option, "--total-steps") == 0) {
            parsed = parse_seed(value, &config.total_steps);
            changes_training = 1;
        } else if (strcmp(option, "--min-learning-rate") == 0) {
            parsed = parse_positive_float(value, &config.minimum_learning_rate);
            changes_training = 1;
        } else if (strcmp(option, "--beta1") == 0) {
            parsed = parse_positive_float(value, &config.beta1);
            changes_training = 1;
        } else if (strcmp(option, "--beta2") == 0) {
            parsed = parse_positive_float(value, &config.beta2);
            changes_training = 1;
        } else if (strcmp(option, "--epsilon") == 0) {
            parsed = parse_positive_float(value, &config.epsilon);
            changes_training = 1;
        } else if (strcmp(option, "--weight-decay") == 0) {
            parsed = parse_positive_float(value, &config.weight_decay);
            changes_training = 1;
        } else if (strcmp(option, "--gradient-clip") == 0) {
            parsed = parse_positive_float(value, &config.gradient_clip_norm);
            changes_training = 1;
        } else if (strcmp(option, "--seed") == 0) {
            parsed = parse_seed(value, &config.seed);
            changes_training = 1;
        } else if (strcmp(option, "--backend") == 0) {
            parsed = parse_backend_choice(value, &backend_choice);
            backend_was_explicit = parsed;
        } else if (strcmp(option, "--checkpoint") == 0 && checkpoint_path == NULL) {
            checkpoint_path = value;
            parsed = value[0] != '\0';
        } else if (strcmp(option, "--checkpoint-every") == 0) {
            parsed = parse_positive_size(value, &checkpoint_every);
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
            fprintf(stderr, "Invalid SFT option: %s %s\n", option, value);
            return 1;
        }
        resume_forbidden_option |= changes_training;
    }
    if ((base_path == NULL) == (resume_path == NULL)) {
        fprintf(stderr, "SFT requires exactly one of --base FILE or --resume FILE.\n");
        return 1;
    }
    if (resume_path != NULL && resume_forbidden_option != 0) {
        fprintf(stderr,
                "--resume restores SFT settings; only equivalent batch overrides are allowed.\n");
        return 1;
    }
    if (backend_was_explicit == 0 && resolve_environment_backend(&backend_choice) == 0) {
        return 1;
    }
    if (checkpoint_every != 0U && checkpoint_path == NULL) {
        fprintf(stderr, "--checkpoint-every requires --checkpoint FILE.\n");
        return 1;
    }
    if (validation_path == NULL && (validation_every != 0U || best_checkpoint_path != NULL)) {
        fprintf(stderr, "SFT validation options require --validation FILE.\n");
        return 1;
    }
    if (validation_path != NULL && validation_every == 0U) {
        validation_every = 200U;
    }
    if (config.minimum_learning_rate > config.learning_rate) {
        fprintf(stderr, "--min-learning-rate must not exceed --learning-rate.\n");
        return 1;
    }
    lm_sft_dataset *dataset = NULL;
    lm_dataset_status dataset_status = lm_sft_dataset_open(argv[3], &dataset);
    if (dataset_status != LM_DATASET_OK || lm_sft_dataset_get_split(dataset) != LM_DATASET_TRAIN) {
        fprintf(stderr, "Opening SFT training dataset failed: %s\n",
                lm_dataset_status_string(dataset_status));
        lm_sft_dataset_close(dataset);
        return 1;
    }
    config.context_length = lm_sft_dataset_context_length(dataset);
    lm_sft_dataset *validation_dataset = NULL;
    if (validation_path != NULL) {
        dataset_status = lm_sft_dataset_open(validation_path, &validation_dataset);
        if (dataset_status != LM_DATASET_OK ||
            lm_sft_dataset_get_split(validation_dataset) != LM_DATASET_VALIDATION) {
            fprintf(stderr, "Opening SFT validation dataset failed: %s\n",
                    lm_dataset_status_string(dataset_status));
            lm_sft_dataset_close(validation_dataset);
            lm_sft_dataset_close(dataset);
            return 1;
        }
    }

    llm_backend *backend = NULL;
    lm_model *model = NULL;
    lm_trainer *trainer = NULL;
    llm_status status = create_backend_choice(backend_choice, &backend);
    if (status == LLM_OK && resume_path != NULL) {
        const lm_trainer_resume_options resume_options = {
            .batch_size = batch_was_explicit != 0 ? config.batch_size : 0U,
            .gradient_accumulation_steps =
                accumulation_was_explicit != 0 ? config.gradient_accumulation_steps : 0U};
        status = lm_sft_trainer_load_checkpoint_with_options(backend, dataset, resume_path,
                                                             &resume_options, &model, &trainer);
    } else if (status == LLM_OK) {
        status = lm_trainer_load_checkpoint(backend, NULL, base_path, &model, NULL);
        if (status == LLM_OK) {
            status = lm_model_reset_optimizer_state(model);
        }
        if (status == LLM_OK) {
            status = lm_sft_trainer_create(model, dataset, &config, &trainer);
        }
    }
    lm_model_config model_config = {0};
    if (status == LLM_OK)
        status = lm_model_get_config(model, &model_config);
    if (status == LLM_OK)
        status = lm_trainer_get_config(trainer, &config);
    if (status == LLM_OK &&
        (model_config.vocabulary_size != lm_sft_dataset_model_vocabulary_size(dataset) ||
         model_config.context_length != lm_sft_dataset_context_length(dataset))) {
        status = LLM_INVALID_SHAPE;
    }
    if (status != LLM_OK) {
        fprintf(stderr, "Creating SFT training failed: %s\n", llm_status_string(status));
        lm_trainer_destroy(trainer);
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_sft_dataset_close(validation_dataset);
        lm_sft_dataset_close(dataset);
        return 1;
    }
    FILE *log_file = log_path == NULL ? NULL : fopen(log_path, "a");
    if (log_path != NULL && log_file == NULL) {
        fprintf(stderr, "Opening SFT log failed.\n");
        lm_trainer_destroy(trainer);
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_sft_dataset_close(validation_dataset);
        lm_sft_dataset_close(dataset);
        return 1;
    }
    if (log_file != NULL) {
        (void)setvbuf(log_file, NULL, _IOLBF, 0U);
        fprintf(log_file,
                "{\"schema\":\"llm-lab-sft-event-v1\",\"event\":\"run\","
                "\"start_step\":%llu,\"requested_updates\":%zu,\"backend\":\"%s\","
                "\"batch_size\":%zu,\"context_length\":%zu,\"gradient_accumulation\":%zu}\n",
                lm_trainer_step_count(trainer), requested_steps,
                backend_choice_name(backend_choice), config.batch_size, config.context_length,
                config.gradient_accumulation_steps);
    }
    double best_loss = INFINITY;
    if (best_checkpoint_path != NULL)
        (void)load_best_validation_loss(best_checkpoint_path, &best_loss);
    double latest_validation_loss = INFINITY;
    int has_validation = 0;
    int saved_best = 0;
    unsigned long long last_checkpoint_step = ULLONG_MAX;
    model_training_interrupted = 0;
    install_model_training_signals();
    float loss = 0.0F;
    cli_model_training_progress progress = {.started_at = current_time_seconds(),
                                            .interactive = standard_error_is_terminal()};
    show_model_training_progress(0U, requested_steps, lm_trainer_step_count(trainer),
                                 config.total_steps, loss, &progress);
    for (size_t index = 0U; index < requested_steps; ++index) {
        const double started_at = current_time_seconds();
        status = lm_trainer_step(trainer, &loss);
        if (status != LLM_OK) {
            fprintf(stderr, "SFT failed at update %zu: %s\n", index + 1U,
                    llm_status_string(status));
            break;
        }
        const unsigned long long step = lm_trainer_step_count(trainer);
        const double seconds = current_time_seconds() - started_at;
        if (log_file != NULL) {
            fprintf(log_file,
                    "{\"schema\":\"llm-lab-sft-event-v1\",\"event\":\"train\","
                    "\"step\":%llu,\"assistant_loss\":%.9g,\"learning_rate\":%.9g,"
                    "\"gradient_norm\":%.9g,\"step_seconds\":%.9g}\n",
                    step, loss, lm_trainer_learning_rate(trainer),
                    lm_trainer_gradient_norm(trainer), seconds);
        }
        const int final_step = index + 1U == requested_steps || model_training_interrupted != 0;
        if (validation_dataset != NULL &&
            (step % (unsigned long long)validation_every == 0U || final_step != 0)) {
            float validation_loss = 0.0F;
            status =
                lm_model_evaluate_sft_validation(model, validation_dataset, config.batch_size,
                                                 validation_batches, config.seed, &validation_loss);
            if (status != LLM_OK) {
                fprintf(stderr, "SFT validation failed: %s\n", llm_status_string(status));
                break;
            }
            latest_validation_loss = validation_loss;
            has_validation = 1;
            const int improved = latest_validation_loss < best_loss;
            if (improved != 0) {
                best_loss = latest_validation_loss;
            }
            if (improved != 0 && best_checkpoint_path != NULL) {
                status = lm_sft_trainer_save_checkpoint(trainer, dataset, best_checkpoint_path);
                if (status == LLM_OK &&
                    save_best_validation_metadata(best_checkpoint_path, step, best_loss) == 0) {
                    status = LLM_BACKEND_ERROR;
                }
                if (status != LLM_OK) {
                    fprintf(stderr, "Saving best SFT checkpoint failed: %s\n",
                            llm_status_string(status));
                    break;
                }
                saved_best = 1;
            }
            fprintf(stderr, "SFT validation: assistant loss %.6f, perplexity %.3f%s\n",
                    latest_validation_loss, exp(latest_validation_loss),
                    improved != 0 ? ", nuovo best" : "");
            if (log_file != NULL) {
                fprintf(log_file,
                        "{\"schema\":\"llm-lab-sft-event-v1\",\"event\":\"validation\","
                        "\"step\":%llu,\"assistant_loss\":%.9g,\"perplexity\":%.9g,"
                        "\"improved\":%s}\n",
                        step, latest_validation_loss, exp(latest_validation_loss),
                        improved != 0 ? "true" : "false");
            }
        }
        if (status == LLM_OK && checkpoint_every != 0U &&
            step % (unsigned long long)checkpoint_every == 0U) {
            status = lm_sft_trainer_save_checkpoint(trainer, dataset, checkpoint_path);
            if (status != LLM_OK) {
                fprintf(stderr, "Saving periodic SFT checkpoint failed: %s\n",
                        llm_status_string(status));
                break;
            }
            last_checkpoint_step = step;
        }
        show_model_training_progress(index + 1U, requested_steps, step, config.total_steps, loss,
                                     &progress);
        if (model_training_interrupted != 0)
            break;
    }
    finish_model_training_progress(&progress);
    if (status == LLM_OK && checkpoint_path != NULL &&
        last_checkpoint_step != lm_trainer_step_count(trainer)) {
        status = lm_sft_trainer_save_checkpoint(trainer, dataset, checkpoint_path);
    }
    if (status == LLM_OK) {
        printf("{\"schema\":\"llm-lab-sft-training-v1\",\"steps\":%llu,"
               "\"assistant_loss\":%.8f,\"validation_loss\":",
               lm_trainer_step_count(trainer), loss);
        if (has_validation != 0)
            printf("%.8f", latest_validation_loss);
        else
            printf("null");
        printf(",\"checkpoint_saved\":%s,\"best_checkpoint_saved\":%s,"
               "\"interrupted\":%s}\n",
               checkpoint_path != NULL ? "true" : "false", saved_best != 0 ? "true" : "false",
               model_training_interrupted != 0 ? "true" : "false");
    }
    if (log_file != NULL)
        (void)fclose(log_file);
    lm_trainer_destroy(trainer);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    lm_sft_dataset_close(validation_dataset);
    lm_sft_dataset_close(dataset);
    return status == LLM_OK ? 0 : 1;
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

static void write_generation_bytes_to(FILE *output, const unsigned char *bytes, size_t length) {
    for (size_t index = 0U; index < length; ++index) {
        const unsigned char value = bytes[index];
        if (value >= 32U && value <= 126U) {
            fputc(value, output);
        } else if (value == '\n') {
            fputs("\\n", output);
        } else if (value == '\r') {
            fputs("\\r", output);
        } else if (value == '\t') {
            fputs("\\t", output);
        } else if (value >= 0x80U) {
            const size_t sequence_length =
                valid_utf8_sequence_length(bytes + index, length - index);
            if (sequence_length != 0U) {
                (void)fwrite(bytes + index, 1U, sequence_length, output);
                index += sequence_length - 1U;
            } else {
                fprintf(output, "\\x%02x", value);
            }
        } else {
            fprintf(output, "\\x%02x", value);
        }
    }
}

static void write_generation_bytes(const unsigned char *bytes, size_t length) {
    write_generation_bytes_to(stdout, bytes, length);
}

static int run_model_generate(int argc, char **argv) {
    size_t generated_count = 0U;
    if (parse_positive_size(argv[5], &generated_count) == 0) {
        fprintf(stderr, "TOKENS must be a positive integer supported by this system.\n");
        return 1;
    }
    cli_generation_options options = {
        .temperature = 0.8F, .repetition_penalty = 1.1F, .top_k = 40U, .random_state = UINT64_C(1)};
    cli_backend_choice backend_choice = CLI_BACKEND_CPU;
    int backend_was_explicit = 0;
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
        } else if (strcmp(option, "--backend") == 0) {
            parsed = parse_backend_choice(value, &backend_choice);
            backend_was_explicit = parsed;
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid model generation option: %s %s\n", option, value);
            return 1;
        }
    }
    if (backend_was_explicit == 0 && resolve_environment_backend(&backend_choice) == 0) {
        return 1;
    }
    if (options.random_state == 0U) {
        options.random_state = UINT64_C(0x9e3779b97f4a7c15);
    }
    fprintf(stderr, "Generazione: carico tokenizer e checkpoint su %s...\n",
            backend_choice_name(backend_choice));
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
    llm_status status = create_backend_choice(backend_choice, &backend);
    if (status == LLM_OK) {
        status = lm_model_load_checkpoint_for_inference(backend, argv[3], &model);
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

    lm_decode_session *decode_session = NULL;
    status = lm_decode_session_create(model, 0U, &decode_session);
    float *host_logits = NULL;
    if (status == LLM_OK) {
        host_logits = malloc((size_t)config.vocabulary_size * sizeof(*host_logits));
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
    const size_t prompt_used =
        sequence.length < config.context_length ? sequence.length : config.context_length;
    const size_t prompt_first = sequence.length - prompt_used;
    const double prefill_started = current_time_seconds();
    if (status == LLM_OK)
        status =
            lm_decode_session_prefill(decode_session, sequence.ids + prompt_first, prompt_used);
    const double prefill_seconds = current_time_seconds() - prefill_started;
    cli_model_generation_progress generation_progress = {
        .started_at = current_time_seconds(), .interactive = standard_error_is_terminal()};
    show_model_generation_progress(0U, generated_count, &generation_progress);
    for (size_t generated = 0U; status == LLM_OK && generated < generated_count; ++generated) {
        const size_t available = sequence.length + generated;
        const llm_tensor *device_logits = lm_decode_session_logits(decode_session);
        if (device_logits == NULL)
            status = LLM_BACKEND_ERROR;
        if (status == LLM_OK)
            status = llm_tensor_read(backend, device_logits, host_logits,
                                     (size_t)config.vocabulary_size * sizeof(*host_logits));
        if (status == LLM_OK) {
            sequence.ids[available] =
                sample_generation_token(host_logits, tokenizer_size, sequence.ids, available,
                                        config.context_length, candidates, recent_tokens, &options);
        }
        if (status == LLM_OK) {
            show_model_generation_progress(generated + 1U, generated_count, &generation_progress);
        }
        if (status == LLM_OK && generated + 1U < generated_count) {
            if (lm_decode_session_token_count(decode_session) < config.context_length) {
                status = lm_decode_session_decode(decode_session, sequence.ids[available]);
            } else {
                const size_t refreshed_available = available + 1U;
                const size_t refreshed_used = refreshed_available < config.context_length
                                                  ? refreshed_available
                                                  : config.context_length;
                status = lm_decode_session_prefill(
                    decode_session, sequence.ids + refreshed_available - refreshed_used,
                    refreshed_used);
            }
        }
    }
    finish_model_generation_progress(&generation_progress);
    const double decode_seconds = current_time_seconds() - generation_progress.started_at;
    if (status == LLM_OK) {
        fprintf(stderr,
                "Inferenza cached: prefill %zu token in %.3fs (%.2f token/s), "
                "decode %zu token in %.3fs (%.2f token/s).\n",
                prompt_used, prefill_seconds,
                prefill_seconds > 0.0 ? (double)prompt_used / prefill_seconds : 0.0,
                generated_count, decode_seconds,
                decode_seconds > 0.0 ? (double)generated_count / decode_seconds : 0.0);
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
    lm_decode_session_destroy(decode_session);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    token_sequence_destroy(&sequence);
    tokenizer_destroy(tokenizer);
    return status == LLM_OK ? 0 : 1;
}

static int run_model_chat(int argc, char **argv) {
    size_t maximum_generated = 0U;
    if (parse_positive_size(argv[5], &maximum_generated) == 0) {
        fprintf(stderr, "TOKENS must be a positive integer supported by this system.\n");
        return 1;
    }
    const char *system_prompt = NULL;
    cli_generation_options options = {
        .temperature = 0.7F, .repetition_penalty = 1.1F, .top_k = 40U, .random_state = UINT64_C(1)};
    cli_backend_choice backend_choice = CLI_BACKEND_CPU;
    int backend_was_explicit = 0;
    for (int index = 7; index < argc; index += 2) {
        if (index + 1 >= argc) {
            fprintf(stderr, "Chat options require a value.\n");
            return 1;
        }
        const char *option = argv[index];
        const char *value = argv[index + 1];
        int parsed = 0;
        if (strcmp(option, "--system") == 0) {
            system_prompt = value;
            parsed = 1;
        } else if (strcmp(option, "--temperature") == 0) {
            parsed = parse_positive_float(value, &options.temperature);
        } else if (strcmp(option, "--top-k") == 0) {
            parsed = parse_positive_size(value, &options.top_k);
        } else if (strcmp(option, "--repetition-penalty") == 0) {
            parsed = parse_positive_float(value, &options.repetition_penalty) &&
                     options.repetition_penalty >= 1.0F;
        } else if (strcmp(option, "--seed") == 0) {
            parsed = parse_seed(value, &options.random_state);
        } else if (strcmp(option, "--backend") == 0) {
            parsed = parse_backend_choice(value, &backend_choice);
            backend_was_explicit = parsed;
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid chat option: %s %s\n", option, value);
            return 1;
        }
    }
    if (backend_was_explicit == 0 && resolve_environment_backend(&backend_choice) == 0) {
        return 1;
    }
    if (options.random_state == 0U) {
        options.random_state = UINT64_C(0x9e3779b97f4a7c15);
    }
    tokenizer *text_tokenizer = NULL;
    tokenizer_status tokenizer_result = tokenizer_load(argv[4], &text_tokenizer);
    if (tokenizer_result != TOKENIZER_OK) {
        fprintf(stderr, "Loading tokenizer failed: %s\n",
                tokenizer_status_string(tokenizer_result));
        return 1;
    }
    llm_backend *backend = NULL;
    lm_model *model = NULL;
    llm_status status = create_backend_choice(backend_choice, &backend);
    if (status == LLM_OK)
        status = lm_model_load_checkpoint_for_inference(backend, argv[3], &model);
    lm_model_config config = {0};
    if (status == LLM_OK)
        status = lm_model_get_config(model, &config);
    const uint32_t tokenizer_size = tokenizer_vocabulary_size(text_tokenizer);
    lm_chat_protocol protocol = {0};
    if (status == LLM_OK &&
        (tokenizer_size > UINT32_MAX - 8U || config.vocabulary_size != tokenizer_size + 8U ||
         lm_chat_protocol_v1(tokenizer_size, config.vocabulary_size, &protocol) != LM_DATASET_OK)) {
        status = LLM_INVALID_SHAPE;
    }
    token_sequence system_tokens = {0}, user_tokens = {0};
    if (status == LLM_OK && system_prompt != NULL) {
        tokenizer_result = tokenizer_encode(text_tokenizer, (const unsigned char *)system_prompt,
                                            strlen(system_prompt), &system_tokens);
        status = tokenizer_result == TOKENIZER_OK ? LLM_OK : LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK) {
        tokenizer_result = tokenizer_encode(text_tokenizer, (const unsigned char *)argv[6],
                                            strlen(argv[6]), &user_tokens);
        status = tokenizer_result == TOKENIZER_OK ? LLM_OK : LLM_BACKEND_ERROR;
    }
    size_t prefix_length = 0U;
    const size_t role_tokens = system_prompt != NULL ? 5U : 3U;
    if (status == LLM_OK &&
        (system_tokens.length > SIZE_MAX - user_tokens.length - role_tokens ||
         system_tokens.length + user_tokens.length + role_tokens > config.context_length ||
         maximum_generated > SIZE_MAX - system_tokens.length - user_tokens.length - role_tokens))
        status = LLM_INVALID_SHAPE;
    token_id *sequence = NULL;
    if (status == LLM_OK) {
        prefix_length = system_tokens.length + user_tokens.length + 5U;
        sequence = malloc((prefix_length + maximum_generated) * sizeof(*sequence));
        status = sequence == NULL ? LLM_ALLOCATION_FAILED : LLM_OK;
    }
    if (status == LLM_OK) {
        size_t position = 0U;
        if (system_prompt != NULL) {
            sequence[position++] = protocol.system_token;
            memcpy(sequence + position, system_tokens.ids,
                   system_tokens.length * sizeof(*sequence));
            position += system_tokens.length;
            sequence[position++] = protocol.end_token;
        }
        sequence[position++] = protocol.user_token;
        memcpy(sequence + position, user_tokens.ids, user_tokens.length * sizeof(*sequence));
        position += user_tokens.length;
        sequence[position++] = protocol.end_token;
        sequence[position++] = protocol.assistant_token;
        prefix_length = position;
    }
    token_sequence_destroy(&user_tokens);
    token_sequence_destroy(&system_tokens);
    if (status != LLM_OK) {
        fprintf(stderr, "Preparing chat prompt failed: %s\n", llm_status_string(status));
        free(sequence);
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        tokenizer_destroy(text_tokenizer);
        return 1;
    }

    lm_decode_session *decode_session = NULL;
    status = lm_decode_session_create(model, 0U, &decode_session);
    float *host_logits = NULL;
    cli_generation_candidate *candidates = NULL;
    unsigned char *recent_tokens = NULL;
    if (status == LLM_OK) {
        host_logits = malloc((size_t)config.vocabulary_size * sizeof(*host_logits));
        candidates = malloc((size_t)tokenizer_size * sizeof(*candidates));
        recent_tokens = calloc(tokenizer_size, sizeof(*recent_tokens));
        if (host_logits == NULL || candidates == NULL || recent_tokens == NULL)
            status = LLM_ALLOCATION_FAILED;
    }
    const double prefill_started = current_time_seconds();
    if (status == LLM_OK)
        status = lm_decode_session_prefill(decode_session, sequence, prefix_length);
    const double prefill_seconds = current_time_seconds() - prefill_started;
    size_t generated_count = 0U;
    cli_model_generation_progress progress = {.started_at = current_time_seconds(),
                                              .interactive = standard_error_is_terminal()};
    show_model_generation_progress(0U, maximum_generated, &progress);
    while (status == LLM_OK && generated_count < maximum_generated) {
        const size_t available = prefix_length + generated_count;
        const llm_tensor *device_logits = lm_decode_session_logits(decode_session);
        if (device_logits == NULL)
            status = LLM_BACKEND_ERROR;
        if (status == LLM_OK)
            status = llm_tensor_read(backend, device_logits, host_logits,
                                     (size_t)config.vocabulary_size * sizeof(*host_logits));
        if (status == LLM_OK) {
            const token_id next = sample_chat_token(host_logits, tokenizer_size, protocol.end_token,
                                                    sequence, available, config.context_length,
                                                    candidates, recent_tokens, &options);
            if (next == protocol.end_token)
                break;
            sequence[prefix_length + generated_count++] = next;
            show_model_generation_progress(generated_count, maximum_generated, &progress);
            if (generated_count < maximum_generated) {
                if (lm_decode_session_token_count(decode_session) < config.context_length) {
                    status = lm_decode_session_decode(decode_session, next);
                } else {
                    const size_t refreshed_available = available + 1U;
                    const size_t refreshed_used = refreshed_available < config.context_length
                                                      ? refreshed_available
                                                      : config.context_length;
                    status = lm_decode_session_prefill(
                        decode_session, sequence + refreshed_available - refreshed_used,
                        refreshed_used);
                }
            }
        }
    }
    finish_model_generation_progress(&progress);
    const double decode_seconds = current_time_seconds() - progress.started_at;
    if (status == LLM_OK) {
        fprintf(stderr,
                "Inferenza cached: prefill %zu token in %.3fs (%.2f token/s), "
                "decode %zu token in %.3fs (%.2f token/s).\n",
                prefix_length, prefill_seconds,
                prefill_seconds > 0.0 ? (double)prefix_length / prefill_seconds : 0.0,
                generated_count, decode_seconds,
                decode_seconds > 0.0 ? (double)generated_count / decode_seconds : 0.0);
    }
    if (status == LLM_OK) {
        const token_sequence answer = {.ids = sequence + prefix_length, .length = generated_count};
        unsigned char *bytes = NULL;
        size_t byte_count = 0U;
        tokenizer_result = tokenizer_decode(text_tokenizer, &answer, &bytes, &byte_count);
        if (tokenizer_result == TOKENIZER_OK) {
            write_generation_bytes(bytes, byte_count);
            fputc('\n', stdout);
        } else {
            status = LLM_BACKEND_ERROR;
        }
        tokenizer_bytes_destroy(bytes);
    }
    if (status != LLM_OK)
        fprintf(stderr, "Chat generation failed: %s\n", llm_status_string(status));
    free(recent_tokens);
    free(candidates);
    free(host_logits);
    lm_decode_session_destroy(decode_session);
    free(sequence);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    tokenizer_destroy(text_tokenizer);
    return status == LLM_OK ? 0 : 1;
}

static int run_model_evaluate(int argc, char **argv) {
    size_t steps = 0U;
    size_t batch_size = 2U;
    uint64_t seed = UINT64_C(1);
    cli_backend_choice backend_choice = CLI_BACKEND_CPU;
    int backend_was_explicit = 0;
    if (parse_positive_size(argv[5], &steps) == 0) {
        fprintf(stderr, "BATCHES must be a positive integer supported by this system.\n");
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
        } else if (strcmp(argv[index], "--backend") == 0) {
            parsed = parse_backend_choice(argv[index + 1], &backend_choice);
            backend_was_explicit = parsed;
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid model evaluation option: %s %s\n", argv[index],
                    argv[index + 1]);
            return 1;
        }
    }
    if (backend_was_explicit == 0 && resolve_environment_backend(&backend_choice) == 0) {
        return 1;
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
    llm_status status = create_backend_choice(backend_choice, &backend);
    if (status == LLM_OK) {
        status = lm_model_load_checkpoint_for_inference(backend, argv[4], &model);
    }
    lm_model_config config = {0};
    if (status == LLM_OK) {
        status = lm_model_get_config(model, &config);
    }
    if (status == LLM_OK && config.vocabulary_size != lm_dataset_model_vocabulary_size(dataset)) {
        status = LLM_INVALID_SHAPE;
    }
    if (status != LLM_OK) {
        fprintf(stderr, "Loading evaluation model failed: %s\n", llm_status_string(status));
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        lm_dataset_close(dataset);
        return 1;
    }
    fprintf(stderr, "Valutazione modello su %s: %zu batch...\n",
            backend_choice_name(backend_choice), steps);
    float mean_loss = 0.0F;
    status = lm_model_evaluate_validation(model, dataset, batch_size, steps, seed, &mean_loss);
    if (status == LLM_OK) {
        printf("{\"schema\":\"llm-lab-model-evaluation-v1\",\"batches\":%zu,"
               "\"loss\":%.8f,\"perplexity\":%.8f,\"layer_count\":%zu,"
               "\"backend\":\"%s\",\"device\":\"%s\"}\n",
               steps, (double)mean_loss, exp((double)mean_loss), config.layer_count,
               backend_choice_name(backend_choice),
               backend_choice_device_name(backend_choice, backend));
    } else {
        fprintf(stderr, "Model evaluation failed: %s\n", llm_status_string(status));
    }
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    lm_dataset_close(dataset);
    return status == LLM_OK ? 0 : 1;
}

static void diagnostic_record_rank(cli_next_token_diagnostics *diagnostics, size_t rank,
                                   int is_end_of_document) {
    ++diagnostics->token_count;
    diagnostics->rank_sum += (double)rank;
    diagnostics->reciprocal_rank_sum += 1.0 / (double)rank;
    if (rank <= 1U)
        ++diagnostics->top_1_count;
    if (rank <= 5U)
        ++diagnostics->top_5_count;
    if (rank <= 20U)
        ++diagnostics->top_20_count;
    if (rank <= 100U)
        ++diagnostics->top_100_count;
    if (is_end_of_document != 0) {
        ++diagnostics->end_of_document_count;
        return;
    }
    ++diagnostics->regular_token_count;
    if (rank <= 1U)
        ++diagnostics->regular_top_1_count;
    if (rank <= 5U)
        ++diagnostics->regular_top_5_count;
    if (rank <= 20U)
        ++diagnostics->regular_top_20_count;
    if (rank <= 100U)
        ++diagnostics->regular_top_100_count;
}

static double diagnostic_fraction(uint64_t count, uint64_t total) {
    return total == 0U ? 0.0 : (double)count / (double)total;
}

static void diagnostic_print_tokens(const char *label, const tokenizer *active_tokenizer,
                                    token_id end_of_document, const token_id *tokens,
                                    size_t count) {
    fprintf(stderr, "%s", label);
    const uint32_t tokenizer_size = tokenizer_vocabulary_size(active_tokenizer);
    size_t index = 0U;
    while (index < count) {
        if (tokens[index] >= tokenizer_size) {
            if (tokens[index] == end_of_document)
                fputs("<EOD>", stderr);
            else
                fprintf(stderr, "<ID:%" PRIu32 ">", tokens[index]);
            ++index;
            continue;
        }
        const size_t first = index;
        while (index < count && tokens[index] < tokenizer_size)
            ++index;
        const token_sequence sequence = {.ids = (token_id *)(tokens + first),
                                         .length = index - first};
        unsigned char *bytes = NULL;
        size_t byte_count = 0U;
        if (tokenizer_decode(active_tokenizer, &sequence, &bytes, &byte_count) != TOKENIZER_OK) {
            fputs("<INVALID-SEQUENCE>", stderr);
            tokenizer_bytes_destroy(bytes);
            continue;
        }
        write_generation_bytes_to(stderr, bytes, byte_count);
        tokenizer_bytes_destroy(bytes);
    }
    fputc('\n', stderr);
}

static int run_model_diagnose(int argc, char **argv) {
    size_t batches = 0U;
    size_t batch_size = 1U;
    uint64_t seed = UINT64_C(1);
    cli_backend_choice backend_choice = CLI_BACKEND_CPU;
    int backend_was_explicit = 0;
    if (parse_positive_size(argv[6], &batches) == 0) {
        fprintf(stderr, "BATCHES must be a positive integer supported by this system.\n");
        return 1;
    }
    for (int index = 7; index < argc; index += 2) {
        if (index + 1 >= argc) {
            fprintf(stderr, "Model diagnostic options require a value.\n");
            return 1;
        }
        int parsed = 0;
        if (strcmp(argv[index], "--batch-size") == 0) {
            parsed = parse_positive_size(argv[index + 1], &batch_size);
        } else if (strcmp(argv[index], "--seed") == 0) {
            parsed = parse_seed(argv[index + 1], &seed);
        } else if (strcmp(argv[index], "--backend") == 0) {
            parsed = parse_backend_choice(argv[index + 1], &backend_choice);
            backend_was_explicit = parsed;
        }
        if (parsed == 0) {
            fprintf(stderr, "Invalid model diagnostic option: %s %s\n", argv[index],
                    argv[index + 1]);
            return 1;
        }
    }
    if (backend_was_explicit == 0 && resolve_environment_backend(&backend_choice) == 0)
        return 1;

    cli_model_dataset_open_progress dataset_progress = {
        .started_at = current_time_seconds(), .interactive = standard_error_is_terminal()};
    lm_dataset *dataset = NULL;
    const lm_dataset_status dataset_status = lm_dataset_open_with_progress(
        argv[3], show_model_dataset_open_progress, &dataset_progress, &dataset);
    finish_model_dataset_open_progress(&dataset_progress);
    if (dataset_status != LM_DATASET_OK || lm_dataset_get_split(dataset) != LM_DATASET_VALIDATION) {
        fprintf(stderr, "Model diagnostics require a validation split artifact.\n");
        lm_dataset_close(dataset);
        return 1;
    }

    int tokenizer_matches = 0;
    const lm_dataset_status tokenizer_match_status =
        lm_dataset_tokenizer_matches(dataset, argv[5], &tokenizer_matches);
    if (tokenizer_match_status != LM_DATASET_OK || tokenizer_matches == 0) {
        fprintf(stderr,
                "Diagnostic tokenizer does not match the tokenizer recorded in the dataset.\n");
        lm_dataset_close(dataset);
        return 1;
    }

    tokenizer *active_tokenizer = NULL;
    tokenizer_status tokenizer_result = tokenizer_load(argv[5], &active_tokenizer);
    llm_backend *backend = NULL;
    lm_model *model = NULL;
    llm_status status = tokenizer_result == TOKENIZER_OK
                            ? create_backend_choice(backend_choice, &backend)
                            : LLM_BACKEND_ERROR;
    if (status == LLM_OK)
        status = lm_model_load_checkpoint_for_inference(backend, argv[4], &model);
    lm_model_config config = {0};
    if (status == LLM_OK)
        status = lm_model_get_config(model, &config);
    const uint32_t tokenizer_size = tokenizer_vocabulary_size(active_tokenizer);
    if (status == LLM_OK &&
        (tokenizer_size == UINT32_MAX || config.vocabulary_size < tokenizer_size + 1U ||
         tokenizer_size != lm_dataset_tokenizer_vocabulary_size(dataset) ||
         config.vocabulary_size != lm_dataset_model_vocabulary_size(dataset))) {
        status = LLM_INVALID_SHAPE;
    }
    if (status != LLM_OK || batch_size > SIZE_MAX / config.context_length) {
        fprintf(stderr, "Loading diagnostic model failed: %s\n",
                status == LLM_OK ? "invalid batch shape" : llm_status_string(status));
        lm_model_destroy(model);
        llm_backend_destroy(backend);
        tokenizer_destroy(active_tokenizer);
        lm_dataset_close(dataset);
        return 1;
    }

    const size_t token_count = batch_size * config.context_length;
    const size_t input_shape[] = {batch_size, config.context_length};
    const size_t target_shape[] = {token_count};
    const size_t logits_shape[] = {token_count, config.vocabulary_size};
    token_id *host_inputs = malloc(token_count * sizeof(*host_inputs));
    token_id *host_targets = malloc(token_count * sizeof(*host_targets));
    token_id *sample_inputs = malloc(config.context_length * sizeof(*sample_inputs));
    token_id *sample_targets = malloc(config.context_length * sizeof(*sample_targets));
    token_id *sample_predictions = malloc(config.context_length * sizeof(*sample_predictions));
    float *host_logits = NULL;
    llm_tensor inputs = {0};
    llm_tensor targets = {0};
    llm_tensor logits = {0};
    llm_tensor loss = {0};
    lm_batcher *batcher = NULL;
    if (host_inputs == NULL || host_targets == NULL || sample_inputs == NULL ||
        sample_targets == NULL || sample_predictions == NULL) {
        status = LLM_ALLOCATION_FAILED;
    }
    if (status == LLM_OK && lm_batcher_create(dataset, batch_size, config.context_length, seed,
                                              &batcher) != LM_DATASET_OK) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 2U, input_shape, &inputs);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_U32, 1U, target_shape, &targets);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 2U, logits_shape, &logits);
    if (status == LLM_OK)
        status = llm_tensor_create(backend, LLM_DTYPE_F32, 0U, NULL, &loss);
    if (status == LLM_OK && logits.element_count <= SIZE_MAX / sizeof(*host_logits)) {
        host_logits = malloc(logits.element_count * sizeof(*host_logits));
        status = host_logits == NULL ? LLM_ALLOCATION_FAILED : LLM_OK;
    } else if (status == LLM_OK) {
        status = LLM_OVERFLOW;
    }

    fprintf(stderr, "Diagnostica next-token su %s: %zu batch x %zu...\n",
            backend_choice_name(backend_choice), batches, batch_size);
    cli_next_token_diagnostics diagnostics = {0};
    const token_id end_of_document = lm_dataset_end_of_document_token(dataset);
    for (size_t batch = 0U; status == LLM_OK && batch < batches; ++batch) {
        if (lm_batcher_next(batcher, host_inputs, host_targets) != LM_DATASET_OK) {
            status = LLM_BACKEND_ERROR;
            break;
        }
        if (batch == 0U) {
            (void)memcpy(sample_inputs, host_inputs,
                         config.context_length * sizeof(*sample_inputs));
            (void)memcpy(sample_targets, host_targets,
                         config.context_length * sizeof(*sample_targets));
        }
        status = llm_backend_begin_batch(backend);
        if (status == LLM_OK)
            status =
                llm_tensor_write(backend, &inputs, host_inputs, token_count * sizeof(*host_inputs));
        if (status == LLM_OK)
            status = llm_tensor_write(backend, &targets, host_targets,
                                      token_count * sizeof(*host_targets));
        if (status == LLM_OK)
            status = lm_model_forward(model, &inputs, &logits);
        if (status == LLM_OK)
            status = llm_cross_entropy_forward(backend, &logits, &targets, &loss);
        const llm_status batch_status = llm_backend_end_batch(backend);
        if (status == LLM_OK)
            status = batch_status;
        float batch_loss = 0.0F;
        if (status == LLM_OK)
            status = llm_tensor_read(backend, &loss, &batch_loss, sizeof(batch_loss));
        if (status == LLM_OK)
            status = llm_tensor_read(backend, &logits, host_logits,
                                     logits.element_count * sizeof(*host_logits));
        if (status == LLM_OK && isfinite(batch_loss) == 0)
            status = LLM_NUMERICAL_ERROR;
        if (status != LLM_OK)
            break;
        diagnostics.loss_sum += (double)batch_loss;
        for (size_t row = 0U; row < token_count; ++row) {
            const token_id target = host_targets[row];
            if (target >= config.vocabulary_size) {
                status = LLM_INVALID_SHAPE;
                break;
            }
            const float *row_logits = host_logits + row * config.vocabulary_size;
            const float target_logit = row_logits[target];
            if (isfinite(target_logit) == 0) {
                status = LLM_NUMERICAL_ERROR;
                break;
            }
            size_t rank = 1U;
            token_id best = 0U;
            for (uint32_t token = 0U; token < config.vocabulary_size; ++token) {
                if (row_logits[token] > target_logit)
                    ++rank;
                if (row_logits[token] > row_logits[best])
                    best = token;
            }
            diagnostic_record_rank(&diagnostics, rank, target == end_of_document);
            if (batch == 0U && row < config.context_length)
                sample_predictions[row] = best;
        }
    }

    if (status == LLM_OK && config.context_length >= 2U) {
        const size_t prompt_count =
            config.context_length < 256U ? config.context_length / 2U : 128U;
        const size_t available = config.context_length - prompt_count;
        const size_t continuation_count = available < 64U ? available : 64U;
        diagnostic_print_tokens("Sample reale, prompt: ", active_tokenizer, end_of_document,
                                sample_inputs, prompt_count);
        diagnostic_print_tokens("Sample reale, continuazione: ", active_tokenizer, end_of_document,
                                sample_targets + prompt_count - 1U, continuation_count);
        diagnostic_print_tokens("Teacher-forced top-1: ", active_tokenizer, end_of_document,
                                sample_predictions + prompt_count - 1U, continuation_count);
    }
    if (status == LLM_OK && diagnostics.token_count != 0U) {
        const double mean_loss = diagnostics.loss_sum / (double)batches;
        printf(
            "{\"schema\":\"llm-lab-model-diagnostics-v1\",\"batches\":%zu,"
            "\"batch_size\":%zu,\"tokens\":%" PRIu64 ",\"loss\":%.8f,"
            "\"perplexity\":%.8f,\"top_1_accuracy\":%.8f,"
            "\"top_5_accuracy\":%.8f,\"top_20_accuracy\":%.8f,"
            "\"top_100_accuracy\":%.8f,\"mean_rank\":%.8f,\"mrr\":%.8f,"
            "\"regular_tokens\":%" PRIu64 ",\"regular_top_1_accuracy\":%.8f,"
            "\"regular_top_5_accuracy\":%.8f,\"regular_top_20_accuracy\":%.8f,"
            "\"regular_top_100_accuracy\":%.8f,\"end_of_document_tokens\":%" PRIu64
            ",\"backend\":\"%s\",\"device\":\"%s\"}\n",
            batches, batch_size, diagnostics.token_count, mean_loss, exp(mean_loss),
            diagnostic_fraction(diagnostics.top_1_count, diagnostics.token_count),
            diagnostic_fraction(diagnostics.top_5_count, diagnostics.token_count),
            diagnostic_fraction(diagnostics.top_20_count, diagnostics.token_count),
            diagnostic_fraction(diagnostics.top_100_count, diagnostics.token_count),
            diagnostics.rank_sum / (double)diagnostics.token_count,
            diagnostics.reciprocal_rank_sum / (double)diagnostics.token_count,
            diagnostics.regular_token_count,
            diagnostic_fraction(diagnostics.regular_top_1_count, diagnostics.regular_token_count),
            diagnostic_fraction(diagnostics.regular_top_5_count, diagnostics.regular_token_count),
            diagnostic_fraction(diagnostics.regular_top_20_count, diagnostics.regular_token_count),
            diagnostic_fraction(diagnostics.regular_top_100_count, diagnostics.regular_token_count),
            diagnostics.end_of_document_count, backend_choice_name(backend_choice),
            backend_choice_device_name(backend_choice, backend));
    } else if (status != LLM_OK) {
        fprintf(stderr, "Model diagnostics failed: %s\n", llm_status_string(status));
    }

    lm_batcher_destroy(batcher);
    llm_tensor_destroy(&loss);
    llm_tensor_destroy(&logits);
    llm_tensor_destroy(&targets);
    llm_tensor_destroy(&inputs);
    free(host_logits);
    free(sample_predictions);
    free(sample_targets);
    free(sample_inputs);
    free(host_targets);
    free(host_inputs);
    lm_model_destroy(model);
    llm_backend_destroy(backend);
    tokenizer_destroy(active_tokenizer);
    lm_dataset_close(dataset);
    return status == LLM_OK ? 0 : 1;
}

int llm_lab_run_command(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "tokenizer") == 0 && strcmp(argv[2], "train") == 0) {
        return run_tokenizer_train(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "tokenizer") == 0 && strcmp(argv[2], "evaluate") == 0) {
        return run_tokenizer_evaluate(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "dataset") == 0 && strcmp(argv[2], "prepare") == 0) {
        return run_dataset_prepare(argc, argv);
    }
    if (argc >= 8 && strcmp(argv[1], "dataset") == 0 && strcmp(argv[2], "sft-prepare") == 0) {
        return run_sft_dataset_prepare(argc, argv);
    }
    if (argc >= 5 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "train") == 0) {
        return run_model_train(argc, argv);
    }
    if (argc >= 7 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "sft") == 0) {
        return run_model_sft(argc, argv);
    }
    if (argc >= 7 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "generate") == 0) {
        return run_model_generate(argc, argv);
    }
    if (argc >= 7 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "chat") == 0) {
        return run_model_chat(argc, argv);
    }
    if (argc >= 6 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "evaluate") == 0) {
        return run_model_evaluate(argc, argv);
    }
    if (argc >= 7 && strcmp(argv[1], "model") == 0 && strcmp(argv[2], "diagnose") == 0) {
        return run_model_diagnose(argc, argv);
    }

    print_usage(argv[0]);
    return 1;
}
