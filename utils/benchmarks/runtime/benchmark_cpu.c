#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark_suite.h"
#include "runtime/runtime.h"

#define CPU_BENCHMARK_MAX_THREAD_CONFIGS 64U

typedef enum benchmark_output_format {
    BENCHMARK_OUTPUT_HUMAN = 0,
    BENCHMARK_OUTPUT_JSONL,
} benchmark_output_format;

typedef struct benchmark_cli_config {
    cpu_benchmark_config workload;
    int selected_operations[CPU_BENCHMARK_OPERATION_COUNT];
    int selected_backends[2];
    size_t requested_threads[CPU_BENCHMARK_MAX_THREAD_CONFIGS];
    size_t thread_config_count;
    benchmark_output_format output_format;
} benchmark_cli_config;

static void print_usage(const char *program) {
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  --backend NAME      cpu (default), metal, or all\n");
    printf("  --operations LIST   all or comma-separated kernel names\n");
    printf("                      memory: zero,fill,copy\n");
    printf("                      math: add,multiply,scale,accumulate,reduce_*,matmul*\n");
    printf("                      training: silu*,rms_norm*,rope*,attention*,adamw\n");
    printf("  --threads LIST      comma-separated thread counts; use auto for detection\n");
    printf("  --elements N        vector elements for copy/add (default: 1048576)\n");
    printf("  --rows N            matrix rows (default: 256)\n");
    printf("  --columns N         matrix columns/vocabulary (default: 256)\n");
    printf("  --inner N           matmul inner dimension (default: 256)\n");
    printf("  --batch N           Transformer batch size (default: 1)\n");
    printf("  --sequence N        Transformer sequence length (default: 32)\n");
    printf("  --query-heads N     attention query heads (default: 4)\n");
    printf("  --kv-heads N        attention key/value heads (default: 2)\n");
    printf("  --head-dim N        even attention head dimension (default: 32)\n");
    printf("  --warmup N          warm-up iterations (default: 2)\n");
    printf("  --iterations N      measured iterations (default: 10)\n");
    printf("  --sample-ms N       minimum duration per timing sample (default: 10)\n");
    printf("  --format FORMAT     human (default) or jsonl for automation\n");
    printf("  --help              show this help\n\n");
    printf("Legacy matmul syntax remains accepted: %s rows inner columns iterations\n", program);
}

static int parse_size(const char *text, int allow_zero, size_t *out_value) {
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || (allow_zero == 0 && parsed == 0ULL) ||
        parsed > SIZE_MAX) {
        return 0;
    }
    *out_value = (size_t)parsed;
    return 1;
}

static int parse_positive_double(const char *text, double *out_value) {
    errno = 0;
    char *end = NULL;
    const double parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0.0) {
        return 0;
    }
    *out_value = parsed;
    return 1;
}

static char *copy_text(const char *text) {
    const size_t length = strlen(text);
    if (length == SIZE_MAX) {
        return NULL;
    }
    char *copy = malloc(length + 1U);
    if (copy != NULL) {
        (void)memcpy(copy, text, length + 1U);
    }
    return copy;
}

static int parse_operations(const char *text, benchmark_cli_config *config) {
    for (size_t index = 0U; index < CPU_BENCHMARK_OPERATION_COUNT; ++index) {
        config->selected_operations[index] = 0;
    }
    if (strcmp(text, "all") == 0) {
        for (size_t index = 0U; index < CPU_BENCHMARK_OPERATION_COUNT; ++index) {
            config->selected_operations[index] = 1;
        }
        return 1;
    }

    char *list = copy_text(text);
    if (list == NULL) {
        return 0;
    }
    size_t selected_count = 0U;
    for (char *token = strtok(list, ","); token != NULL; token = strtok(NULL, ",")) {
        cpu_benchmark_operation operation = CPU_BENCHMARK_OPERATION_COUNT;
        if (cpu_benchmark_operation_parse(token, &operation) == 0) {
            free(list);
            return 0;
        }
        if (config->selected_operations[operation] == 0) {
            config->selected_operations[operation] = 1;
            ++selected_count;
        }
    }
    free(list);
    return selected_count > 0U;
}

static int parse_threads(const char *text, benchmark_cli_config *config) {
    config->thread_config_count = 0U;
    char *list = copy_text(text);
    if (list == NULL) {
        return 0;
    }
    for (char *token = strtok(list, ","); token != NULL; token = strtok(NULL, ",")) {
        if (config->thread_config_count == CPU_BENCHMARK_MAX_THREAD_CONFIGS) {
            free(list);
            return 0;
        }
        size_t thread_count = 0U;
        if (strcmp(token, "auto") != 0 && parse_size(token, 0, &thread_count) == 0) {
            free(list);
            return 0;
        }
        config->requested_threads[config->thread_config_count] = thread_count;
        ++config->thread_config_count;
    }
    free(list);
    return config->thread_config_count > 0U;
}

static int parse_legacy_arguments(int argc, char **argv, benchmark_cli_config *config) {
    if (argc != 5 || argv[1][0] == '-') {
        return 0;
    }
    if (parse_size(argv[1], 0, &config->workload.rows) == 0 ||
        parse_size(argv[2], 0, &config->workload.inner_size) == 0 ||
        parse_size(argv[3], 0, &config->workload.columns) == 0 ||
        parse_size(argv[4], 0, &config->workload.measured_iterations) == 0) {
        return -1;
    }
    for (size_t index = 0U; index < CPU_BENCHMARK_OPERATION_COUNT; ++index) {
        config->selected_operations[index] = 0;
    }
    config->selected_operations[CPU_BENCHMARK_MATMUL] = 1;
    return 1;
}

static int parse_arguments(int argc, char **argv, benchmark_cli_config *config) {
    const int legacy_status = parse_legacy_arguments(argc, argv, config);
    if (legacy_status != 0) {
        return legacy_status > 0;
    }
    for (int index = 1; index < argc; ++index) {
        const char *option = argv[index];
        if (strcmp(option, "--help") == 0) {
            print_usage(argv[0]);
            return 2;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", option);
            return 0;
        }
        const char *value = argv[++index];
        if (strcmp(option, "--backend") == 0) {
            config->selected_backends[RUNTIME_BENCHMARK_CPU] =
                strcmp(value, "cpu") == 0 || strcmp(value, "all") == 0;
            config->selected_backends[RUNTIME_BENCHMARK_METAL] =
                strcmp(value, "metal") == 0 || strcmp(value, "all") == 0;
            if (config->selected_backends[RUNTIME_BENCHMARK_CPU] == 0 &&
                config->selected_backends[RUNTIME_BENCHMARK_METAL] == 0) {
                fprintf(stderr, "invalid backend: %s (expected cpu, metal, or all)\n", value);
                return 0;
            }
        } else if (strcmp(option, "--operations") == 0) {
            if (parse_operations(value, config) == 0) {
                fprintf(stderr, "invalid operation list: %s\n", value);
                return 0;
            }
        } else if (strcmp(option, "--threads") == 0) {
            if (parse_threads(value, config) == 0) {
                fprintf(stderr, "invalid thread list: %s\n", value);
                return 0;
            }
        } else if (strcmp(option, "--elements") == 0) {
            if (parse_size(value, 0, &config->workload.elements) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--rows") == 0) {
            if (parse_size(value, 0, &config->workload.rows) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--columns") == 0) {
            if (parse_size(value, 0, &config->workload.columns) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--inner") == 0) {
            if (parse_size(value, 0, &config->workload.inner_size) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--batch") == 0) {
            if (parse_size(value, 0, &config->workload.batch_size) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--sequence") == 0) {
            if (parse_size(value, 0, &config->workload.sequence_length) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--query-heads") == 0) {
            if (parse_size(value, 0, &config->workload.query_head_count) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--kv-heads") == 0) {
            if (parse_size(value, 0, &config->workload.key_value_head_count) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--head-dim") == 0) {
            if (parse_size(value, 0, &config->workload.head_dimension) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--warmup") == 0) {
            if (parse_size(value, 0, &config->workload.warmup_iterations) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--iterations") == 0) {
            if (parse_size(value, 0, &config->workload.measured_iterations) == 0) {
                return 0;
            }
        } else if (strcmp(option, "--sample-ms") == 0) {
            double milliseconds = 0.0;
            if (parse_positive_double(value, &milliseconds) == 0) {
                return 0;
            }
            config->workload.minimum_sample_seconds = milliseconds / 1000.0;
        } else if (strcmp(option, "--format") == 0) {
            if (strcmp(value, "human") == 0) {
                config->output_format = BENCHMARK_OUTPUT_HUMAN;
            } else if (strcmp(value, "jsonl") == 0) {
                config->output_format = BENCHMARK_OUTPUT_JSONL;
            } else {
                fprintf(stderr, "invalid output format: %s (expected human or jsonl)\n", value);
                return 0;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", option);
            return 0;
        }
    }
    return 1;
}

static const char *operating_system_name(void) {
#if defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

static const char *architecture_name(void) {
#if defined(__aarch64__)
    return "arm64";
#elif defined(__x86_64__)
    return "x86_64";
#else
    return "unknown";
#endif
}

static const char *compiler_name(void) {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#else
    return "unknown";
#endif
}

static const char *build_type_name(void) {
#ifdef NDEBUG
    return "release";
#else
    return "debug";
#endif
}

static size_t detect_hardware_threads(void) {
    llm_backend *backend = NULL;
    if (llm_backend_cpu_create(&backend) != LLM_OK) {
        return 0U;
    }
    const size_t thread_count = llm_backend_cpu_thread_count(backend);
    llm_backend_destroy(backend);
    return thread_count;
}

static void print_jsonl_metadata(const benchmark_cli_config *config) {
    printf("{\"type\":\"metadata\",\"schema_version\":1,\"os\":\"%s\","
           "\"architecture\":\"%s\",\"compiler\":\"%s\",\"build\":\"%s\","
           "\"detected_hardware_threads\":%zu,\"metal_available\":%s,"
           "\"warmup_iterations\":%zu,\"measured_iterations\":%zu,"
           "\"minimum_sample_seconds\":%.6f}\n",
           operating_system_name(), architecture_name(), compiler_name(), build_type_name(),
           detect_hardware_threads(), llm_backend_metal_is_available() != 0 ? "true" : "false",
           config->workload.warmup_iterations, config->workload.measured_iterations,
           config->workload.minimum_sample_seconds);
    (void)fflush(stdout);
}

static int operation_uses_vector_shape(cpu_benchmark_operation operation) {
    return operation <= CPU_BENCHMARK_ACCUMULATE || operation == CPU_BENCHMARK_SILU ||
           operation == CPU_BENCHMARK_SILU_BACKWARD || operation == CPU_BENCHMARK_ADAMW;
}

static int operation_uses_matmul_shape(cpu_benchmark_operation operation) {
    return operation == CPU_BENCHMARK_MATMUL || operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT ||
           operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT;
}

static int operation_uses_transformer_shape(cpu_benchmark_operation operation) {
    return operation == CPU_BENCHMARK_ROPE || operation == CPU_BENCHMARK_ROPE_BACKWARD ||
           operation == CPU_BENCHMARK_ATTENTION || operation == CPU_BENCHMARK_ATTENTION_BACKWARD;
}

static void print_jsonl_dimensions(cpu_benchmark_operation operation,
                                   const cpu_benchmark_config *config) {
    if (operation_uses_vector_shape(operation) != 0) {
        printf("{\"elements\":%zu}", config->elements);
    } else if (operation_uses_matmul_shape(operation) != 0) {
        printf("{\"rows\":%zu,\"inner\":%zu,\"columns\":%zu}", config->rows, config->inner_size,
               config->columns);
    } else if (operation_uses_transformer_shape(operation) != 0) {
        printf("{\"batch\":%zu,\"sequence\":%zu,\"query_heads\":%zu,"
               "\"kv_heads\":%zu,\"head_dimension\":%zu}",
               config->batch_size, config->sequence_length, config->query_head_count,
               config->key_value_head_count, config->head_dimension);
    } else {
        printf("{\"rows\":%zu,\"columns\":%zu}", config->rows, config->columns);
    }
}

static void print_jsonl_result(const cpu_benchmark_result *result,
                               const cpu_benchmark_config *config, double baseline_seconds,
                               size_t baseline_threads) {
    const double speedup = baseline_seconds / result->median_seconds;
    const double efficiency = result->actual_threads > 0U ? speedup * (double)baseline_threads /
                                                                (double)result->actual_threads
                                                          : 0.0;
    const double variation = result->mean_seconds > 0.0
                                 ? result->standard_deviation_seconds / result->mean_seconds
                                 : 0.0;
    printf("{\"type\":\"result\",\"schema_version\":4,\"backend\":\"%s\","
           "\"device\":\"%s\",\"dtype\":\"%s\",\"operation\":\"%s\","
           "\"requested_threads\":%zu,\"actual_threads\":%zu,\"baseline_threads\":%zu,"
           "\"repetitions_per_sample\":%zu,\"dimensions\":",
           runtime_benchmark_backend_name(result->backend), result->device_name,
           runtime_benchmark_dtype_name(result->dtype),
           cpu_benchmark_operation_name(result->operation), result->requested_threads,
           result->actual_threads, baseline_threads, result->repetitions_per_sample);
    print_jsonl_dimensions(result->operation, config);
    printf(",\"minimum_seconds\":%.9f,\"median_seconds\":%.9f,\"p95_seconds\":%.9f,"
           "\"mean_seconds\":%.9f,\"standard_deviation_seconds\":%.9f,"
           "\"nanoseconds_per_call\":%.3f,\"calls_per_second\":%.3f,"
           "\"coefficient_of_variation\":%.6f,\"throughput\":%.6f,"
           "\"throughput_unit\":\"%s\",\"speedup\":%.6f,"
           "\"parallel_efficiency\":%.6f,\"gpu_median_seconds\":%.9f,"
           "\"gpu_p95_seconds\":%.9f,\"pipeline_compilation_seconds\":%.9f,"
           "\"guard\":%.9g}\n",
           result->minimum_seconds, result->median_seconds, result->p95_seconds,
           result->mean_seconds, result->standard_deviation_seconds, result->nanoseconds_per_call,
           result->calls_per_second, variation, result->throughput, result->throughput_unit,
           speedup, efficiency, result->gpu_median_seconds, result->gpu_p95_seconds,
           result->pipeline_compilation_seconds, result->guard_value);
    (void)fflush(stdout);
}

static void print_human_metadata(const benchmark_cli_config *config) {
    printf("Tensor runtime benchmark\n");
    printf("System: %s / %s | compiler: %s | build: %s | CPU threads: %zu\n",
           operating_system_name(), architecture_name(), compiler_name(), build_type_name(),
           detect_hardware_threads());
    printf("Samples: %zu warm-up, %zu measured, minimum %.1f ms each\n",
           config->workload.warmup_iterations, config->workload.measured_iterations,
           config->workload.minimum_sample_seconds * 1000.0);
    printf("Metal: %s\n", llm_backend_metal_is_available() != 0 ? "available" : "unavailable");
}

static void print_human_operation_title(cpu_benchmark_operation operation,
                                        const cpu_benchmark_config *config) {
    printf("\n%s (", cpu_benchmark_operation_name(operation));
    if (operation_uses_vector_shape(operation) != 0) {
        printf("%zu elements", config->elements);
    } else if (operation_uses_matmul_shape(operation) != 0) {
        printf("%zu x %zu x %zu", config->rows, config->inner_size, config->columns);
    } else if (operation_uses_transformer_shape(operation) != 0) {
        printf("B%zu S%zu Hq%zu Hkv%zu D%zu", config->batch_size, config->sequence_length,
               config->query_head_count, config->key_value_head_count, config->head_dimension);
    } else {
        printf("%zu x %zu", config->rows, config->columns);
    }
    printf(")\n");
}

static void print_human_cpu_header(void) {
    printf("%-10s %12s %12s %20s %10s %12s\n", "Threads", "Median", "P95", "Throughput", "Speedup",
           "Efficiency");
    printf("%-10s %12s %12s %20s %10s %12s\n", "----------", "------------", "------------",
           "--------------------", "----------", "------------");
}

static void print_human_result(const cpu_benchmark_result *result, double baseline_seconds,
                               size_t baseline_threads) {
    const double speedup = baseline_seconds / result->median_seconds;
    const double efficiency =
        speedup * (double)baseline_threads / (double)result->actual_threads * 100.0;
    char throughput[64];
    const int written = snprintf(throughput, sizeof(throughput), "%.3f %s", result->throughput,
                                 result->throughput_unit);
    if (written < 0 || (size_t)written >= sizeof(throughput)) {
        (void)snprintf(throughput, sizeof(throughput), "unavailable");
    }
    printf("%-10zu %9.3f ms %9.3f ms %20s %9.2fx %11.1f%%\n", result->actual_threads,
           result->median_seconds * 1000.0, result->p95_seconds * 1000.0, throughput, speedup,
           efficiency);
}

static void print_human_metal_header(const cpu_benchmark_result *result) {
    printf("Metal device: %s | dtype: %s | pipeline startup: %.3f ms\n", result->device_name,
           runtime_benchmark_dtype_name(result->dtype),
           result->pipeline_compilation_seconds * 1000.0);
    printf("%-10s %12s %12s %12s %20s\n", "Backend", "Median", "P95", "GPU median", "Throughput");
    printf("%-10s %12s %12s %12s %20s\n", "----------", "------------", "------------",
           "------------", "--------------------");
}

static void print_human_metal_result(const cpu_benchmark_result *result) {
    char throughput[64];
    const int written = snprintf(throughput, sizeof(throughput), "%.3f %s", result->throughput,
                                 result->throughput_unit);
    if (written < 0 || (size_t)written >= sizeof(throughput)) {
        (void)snprintf(throughput, sizeof(throughput), "unavailable");
    }
    printf("%-10s %9.3f ms %9.3f ms %9.3f ms %20s\n", "metal", result->median_seconds * 1000.0,
           result->p95_seconds * 1000.0, result->gpu_median_seconds * 1000.0, throughput);
}

int main(int argc, char **argv) {
    benchmark_cli_config config = {
        .workload =
            {
                .elements = 1048576U,
                .rows = 256U,
                .columns = 256U,
                .inner_size = 256U,
                .batch_size = 1U,
                .sequence_length = 32U,
                .query_head_count = 4U,
                .key_value_head_count = 2U,
                .head_dimension = 32U,
                .warmup_iterations = 2U,
                .measured_iterations = 10U,
                .minimum_sample_seconds = 0.01,
            },
        .selected_backends = {1, 0},
        .requested_threads = {1U, 0U},
        .thread_config_count = 2U,
        .output_format = BENCHMARK_OUTPUT_HUMAN,
    };
    for (size_t index = 0U; index < CPU_BENCHMARK_OPERATION_COUNT; ++index) {
        config.selected_operations[index] = 1;
    }

    const int parse_status = parse_arguments(argc, argv, &config);
    if (parse_status == 2) {
        return EXIT_SUCCESS;
    }
    if (parse_status == 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (config.output_format == BENCHMARK_OUTPUT_JSONL) {
        print_jsonl_metadata(&config);
    } else {
        print_human_metadata(&config);
    }
    for (size_t operation_index = 0U; operation_index < CPU_BENCHMARK_OPERATION_COUNT;
         ++operation_index) {
        if (config.selected_operations[operation_index] == 0) {
            continue;
        }
        if (config.output_format == BENCHMARK_OUTPUT_HUMAN) {
            print_human_operation_title((cpu_benchmark_operation)operation_index, &config.workload);
            if (config.selected_backends[RUNTIME_BENCHMARK_CPU] != 0) {
                print_human_cpu_header();
            }
        }
        double baseline_seconds = 0.0;
        size_t baseline_threads = 0U;
        for (size_t thread_index = 0U; config.selected_backends[RUNTIME_BENCHMARK_CPU] != 0 &&
                                       thread_index < config.thread_config_count;
             ++thread_index) {
            cpu_benchmark_result result = {0};
            if (runtime_benchmark_run(
                    RUNTIME_BENCHMARK_CPU, (cpu_benchmark_operation)operation_index,
                    config.requested_threads[thread_index], &config.workload, &result) == 0) {
                return EXIT_FAILURE;
            }
            if (thread_index == 0U) {
                baseline_seconds = result.median_seconds;
                baseline_threads = result.actual_threads;
            }
            if (config.output_format == BENCHMARK_OUTPUT_JSONL) {
                print_jsonl_result(&result, &config.workload, baseline_seconds, baseline_threads);
            } else {
                print_human_result(&result, baseline_seconds, baseline_threads);
            }
        }
        if (config.selected_backends[RUNTIME_BENCHMARK_METAL] != 0) {
            if (runtime_benchmark_operation_supported(
                    RUNTIME_BENCHMARK_METAL, (cpu_benchmark_operation)operation_index) == 0) {
                if (config.output_format == BENCHMARK_OUTPUT_HUMAN) {
                    printf("Metal: operation not implemented, skipped\n");
                }
                continue;
            }
            if (llm_backend_metal_is_available() == 0) {
                if (config.output_format == BENCHMARK_OUTPUT_HUMAN) {
                    printf("Metal unavailable: operation skipped\n");
                }
                continue;
            }
            cpu_benchmark_result result = {0};
            if (runtime_benchmark_run(RUNTIME_BENCHMARK_METAL,
                                      (cpu_benchmark_operation)operation_index, 0U,
                                      &config.workload, &result) == 0) {
                return EXIT_FAILURE;
            }
            if (config.output_format == BENCHMARK_OUTPUT_JSONL) {
                print_jsonl_result(&result, &config.workload, result.median_seconds, 0U);
            } else {
                print_human_metal_header(&result);
                print_human_metal_result(&result);
            }
        }
    }
    return EXIT_SUCCESS;
}
