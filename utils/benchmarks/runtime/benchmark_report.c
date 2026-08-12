#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "benchmark_suite.h"
#include "runtime/runtime.h"

typedef enum report_profile {
    REPORT_PROFILE_QUICK = 0,
    REPORT_PROFILE_STANDARD,
    REPORT_PROFILE_FULL,
} report_profile;

typedef struct machine_information {
    char hostname[256];
    char cpu_name[256];
    unsigned long long memory_bytes;
    size_t cpu_threads;
    int metal_available;
    char metal_device[128];
} machine_information;

typedef struct report_summary {
    size_t successful_results;
    size_t failed_results;
    size_t successful_by_backend[2];
    size_t failed_by_backend[2];
    int has_fastest_matmul;
    double fastest_matmul_seconds;
    runtime_benchmark_backend fastest_matmul_backend;
    double cpu_f32_matmul_seconds;
    double metal_f32_matmul_seconds;
    double matmul_seconds[2];
    double matmul_throughput[2];
} report_summary;

static const char *operating_system_name(void) {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "Unknown";
#endif
}

static const char *architecture_name(void) {
#if defined(__aarch64__)
    return "arm64";
#elif defined(__x86_64__)
    return "x86_64";
#elif defined(__i386__)
    return "x86";
#else
    return "unknown";
#endif
}

static const char *compiler_name(void) {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#else
    return "unknown";
#endif
}

static const char *build_type_name(void) {
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

static void copy_text(char *destination, size_t capacity, const char *source) {
    if (destination == NULL || capacity == 0U) {
        return;
    }
    (void)snprintf(destination, capacity, "%s", source == NULL ? "unknown" : source);
}

static void detect_hostname(machine_information *information) {
    if (gethostname(information->hostname, sizeof(information->hostname)) != 0) {
        copy_text(information->hostname, sizeof(information->hostname), "unknown");
    }
    information->hostname[sizeof(information->hostname) - 1U] = '\0';
}

static void detect_cpu_and_memory(machine_information *information) {
#if defined(__APPLE__)
    size_t name_size = sizeof(information->cpu_name);
    if (sysctlbyname("machdep.cpu.brand_string", information->cpu_name, &name_size, NULL, 0U) !=
        0) {
        name_size = sizeof(information->cpu_name);
        if (sysctlbyname("hw.model", information->cpu_name, &name_size, NULL, 0U) != 0) {
            copy_text(information->cpu_name, sizeof(information->cpu_name), "unknown");
        }
    }
    uint64_t memory = 0U;
    size_t memory_size = sizeof(memory);
    if (sysctlbyname("hw.memsize", &memory, &memory_size, NULL, 0U) == 0) {
        information->memory_bytes = memory;
    }
#elif defined(__linux__)
    copy_text(information->cpu_name, sizeof(information->cpu_name), "unknown");
    FILE *cpu_information = fopen("/proc/cpuinfo", "r");
    if (cpu_information != NULL) {
        char line[512];
        while (fgets(line, sizeof(line), cpu_information) != NULL) {
            const char *separator = strchr(line, ':');
            if (separator == NULL ||
                (strncmp(line, "model name", 10U) != 0 && strncmp(line, "Hardware", 8U) != 0)) {
                continue;
            }
            const char *value = separator + 1;
            while (*value == ' ' || *value == '\t') {
                ++value;
            }
            size_t length = strcspn(value, "\r\n");
            if (length >= sizeof(information->cpu_name)) {
                length = sizeof(information->cpu_name) - 1U;
            }
            (void)memcpy(information->cpu_name, value, length);
            information->cpu_name[length] = '\0';
            break;
        }
        (void)fclose(cpu_information);
    }
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0 &&
        (unsigned long long)pages <= ULLONG_MAX / (unsigned long long)page_size) {
        information->memory_bytes = (unsigned long long)pages * (unsigned long long)page_size;
    }
#else
    copy_text(information->cpu_name, sizeof(information->cpu_name), "unknown");
#endif
}

static void detect_runtime_devices(machine_information *information) {
    llm_backend *cpu = NULL;
    if (llm_backend_cpu_create(&cpu) == LLM_OK) {
        information->cpu_threads = llm_backend_cpu_thread_count(cpu);
        llm_backend_destroy(cpu);
    }
    information->metal_available = llm_backend_metal_is_available();
    if (information->metal_available != 0) {
        llm_backend *metal = NULL;
        if (llm_backend_metal_create(&metal) == LLM_OK) {
            copy_text(information->metal_device, sizeof(information->metal_device),
                      llm_backend_metal_device_name(metal));
            llm_backend_destroy(metal);
        }
    }
    if (information->metal_device[0] == '\0') {
        copy_text(information->metal_device, sizeof(information->metal_device), "unavailable");
    }
}

static machine_information detect_machine(void) {
    machine_information information = {0};
    detect_hostname(&information);
    detect_cpu_and_memory(&information);
    detect_runtime_devices(&information);
    return information;
}

static const char *profile_name(report_profile profile) {
    return profile == REPORT_PROFILE_QUICK      ? "quick"
           : profile == REPORT_PROFILE_STANDARD ? "standard"
                                                : "full";
}

static cpu_benchmark_config profile_config(report_profile profile) {
    cpu_benchmark_config config = {
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
        .measured_iterations = 7U,
        .minimum_sample_seconds = 0.005,
    };
    if (profile == REPORT_PROFILE_QUICK) {
        config.elements = 65536U;
        config.rows = 64U;
        config.columns = 128U;
        config.inner_size = 64U;
        config.sequence_length = 16U;
        config.warmup_iterations = 1U;
        config.measured_iterations = 3U;
        config.minimum_sample_seconds = 0.001;
    } else if (profile == REPORT_PROFILE_FULL) {
        config.elements = 8388608U;
        config.rows = 1024U;
        config.columns = 1024U;
        config.inner_size = 1024U;
        config.sequence_length = 128U;
        config.query_head_count = 8U;
        config.key_value_head_count = 2U;
        config.head_dimension = 64U;
        config.warmup_iterations = 3U;
        config.measured_iterations = 15U;
        config.minimum_sample_seconds = 0.02;
    }
    return config;
}

static void print_usage(const char *program) {
    printf("Usage: %s [--quick|--standard|--full]\n\n", program);
    printf("  --quick     short functional performance report\n");
    printf("  --standard  balanced report (default)\n");
    printf("  --full      larger, slower workloads for stable measurements\n");
    printf("  --help      show this help\n");
}

static int parse_arguments(int argc, char **argv, report_profile *out_profile) {
    *out_profile = REPORT_PROFILE_STANDARD;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--quick") == 0) {
            *out_profile = REPORT_PROFILE_QUICK;
        } else if (strcmp(argv[index], "--standard") == 0) {
            *out_profile = REPORT_PROFILE_STANDARD;
        } else if (strcmp(argv[index], "--full") == 0) {
            *out_profile = REPORT_PROFILE_FULL;
        } else if (strcmp(argv[index], "--help") == 0) {
            print_usage(argv[0]);
            return 2;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[index]);
            return 0;
        }
    }
    return 1;
}

static void print_machine(const machine_information *machine, report_profile profile,
                          const cpu_benchmark_config *config) {
    const double memory_gib = (double)machine->memory_bytes / (1024.0 * 1024.0 * 1024.0);
    printf("Report benchmark hardware\n\n");
    printf("Hardware rilevato\n");
    printf("  Macchina:     %s\n", machine->hostname);
    printf("  Sistema:      %s / %s\n", operating_system_name(), architecture_name());
    printf("  Memoria:      %.1f GiB\n", memory_gib);
    printf("  Compilatore:  %s\n", compiler_name());
    printf("  Build:        %s\n", build_type_name());
    printf("  CPU:          %s (%zu thread logici)\n", machine->cpu_name, machine->cpu_threads);
    printf("  GPU Metal:    %s\n", machine->metal_device);
    printf("  Profilo:      %s | %zu warm-up | %zu campioni | minimo %.1f ms\n",
           profile_name(profile), config->warmup_iterations, config->measured_iterations,
           config->minimum_sample_seconds * 1000.0);
    (void)fflush(stdout);
}

static void print_backend_header(const machine_information *machine,
                                 runtime_benchmark_backend backend) {
    if (backend == RUNTIME_BENCHMARK_CPU) {
        printf("\nCPU — %s\n", machine->cpu_name);
        printf("Esecuzione con %zu thread logici.\n", machine->cpu_threads);
    } else {
        printf("\nGPU Metal — %s\n", machine->metal_device);
        printf("Tempo GPU e tempo end-to-end vengono mostrati separatamente.\n");
    }
    printf("%-25s %-5s %-17s %11s %11s %11s %20s\n", "Kernel", "Tipo", "Forma", "Mediana", "P95",
           "Tempo GPU", "Velocita'");
    printf("%-25s %-5s %-17s %11s %11s %11s %20s\n", "-------------------------", "-----",
           "-----------------", "-----------", "-----------", "-----------",
           "--------------------");
    (void)fflush(stdout);
}

static void format_shape(cpu_benchmark_operation operation, const cpu_benchmark_config *config,
                         char *output, size_t capacity) {
    if (operation <= CPU_BENCHMARK_ACCUMULATE || operation == CPU_BENCHMARK_SILU ||
        operation == CPU_BENCHMARK_SILU_BACKWARD || operation == CPU_BENCHMARK_ADAMW) {
        (void)snprintf(output, capacity, "%zu elementi", config->elements);
    } else if (operation == CPU_BENCHMARK_MATMUL ||
               operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_LEFT ||
               operation == CPU_BENCHMARK_MATMUL_TRANSPOSE_RIGHT) {
        (void)snprintf(output, capacity, "%zux%zux%zu", config->rows, config->inner_size,
                       config->columns);
    } else if (operation == CPU_BENCHMARK_ROPE || operation == CPU_BENCHMARK_ROPE_BACKWARD ||
               operation == CPU_BENCHMARK_ATTENTION ||
               operation == CPU_BENCHMARK_ATTENTION_BACKWARD) {
        (void)snprintf(output, capacity, "B%zu S%zu H%zu D%zu", config->batch_size,
                       config->sequence_length, config->query_head_count, config->head_dimension);
    } else {
        (void)snprintf(output, capacity, "%zux%zu", config->rows, config->columns);
    }
}

static void update_summary(const cpu_benchmark_result *result, report_summary *summary) {
    ++summary->successful_results;
    ++summary->successful_by_backend[result->backend];
    if (result->operation != CPU_BENCHMARK_MATMUL) {
        return;
    }
    summary->matmul_seconds[result->backend] = result->median_seconds;
    summary->matmul_throughput[result->backend] = result->throughput;
    if (summary->has_fastest_matmul == 0 ||
        result->median_seconds < summary->fastest_matmul_seconds) {
        summary->has_fastest_matmul = 1;
        summary->fastest_matmul_seconds = result->median_seconds;
        summary->fastest_matmul_backend = result->backend;
    }
    if (result->dtype == LLM_DTYPE_F32 && result->backend == RUNTIME_BENCHMARK_CPU) {
        summary->cpu_f32_matmul_seconds = result->median_seconds;
    } else if (result->dtype == LLM_DTYPE_F32 && result->backend == RUNTIME_BENCHMARK_METAL) {
        summary->metal_f32_matmul_seconds = result->median_seconds;
    }
}

static int run_one(runtime_benchmark_backend backend, cpu_benchmark_operation operation,
                   cpu_benchmark_config *config, report_summary *summary) {
    cpu_benchmark_result result = {0};
    char shape[64];
    format_shape(operation, config, shape, sizeof(shape));
    if (runtime_benchmark_run(backend, operation, 0U, config, &result) == 0) {
        ++summary->failed_results;
        ++summary->failed_by_backend[backend];
        printf("%-25s %-5s %-17s %11s %11s %11s %20s\n", cpu_benchmark_operation_name(operation),
               "f32", shape, "ERRORE", "-", "-", "-");
        (void)fflush(stdout);
        return 0;
    }
    char throughput[64];
    const int written = snprintf(throughput, sizeof(throughput), "%.3f %s", result.throughput,
                                 result.throughput_unit);
    if (written < 0 || (size_t)written >= sizeof(throughput)) {
        copy_text(throughput, sizeof(throughput), "unavailable");
    }
    char gpu_time[32];
    if (backend == RUNTIME_BENCHMARK_METAL) {
        (void)snprintf(gpu_time, sizeof(gpu_time), "%.3f ms", result.gpu_median_seconds * 1000.0);
    } else {
        copy_text(gpu_time, sizeof(gpu_time), "-");
    }
    printf("%-25s %-5s %-17s %8.3f ms %8.3f ms %11s %20s\n",
           cpu_benchmark_operation_name(operation), runtime_benchmark_dtype_name(result.dtype),
           shape, result.median_seconds * 1000.0, result.p95_seconds * 1000.0, gpu_time,
           throughput);
    (void)fflush(stdout);
    update_summary(&result, summary);
    return 1;
}

static void print_backend_summary(runtime_benchmark_backend backend,
                                  const report_summary *summary) {
    printf("Risultato %s: %zu kernel riusciti, %zu falliti.\n",
           backend == RUNTIME_BENCHMARK_CPU ? "CPU" : "GPU Metal",
           summary->successful_by_backend[backend], summary->failed_by_backend[backend]);
    if (summary->matmul_seconds[backend] > 0.0) {
        printf("  Matmul f32 : %8.3f GFLOP/s (%8.3f ms)\n", summary->matmul_throughput[backend],
               summary->matmul_seconds[backend] * 1000.0);
    }
}

static void run_backend(const machine_information *machine, runtime_benchmark_backend backend,
                        cpu_benchmark_config *config, report_summary *summary) {
    print_backend_header(machine, backend);
    for (size_t operation = 0U; operation < CPU_BENCHMARK_OPERATION_COUNT; ++operation) {
        if (runtime_benchmark_operation_supported(backend, (cpu_benchmark_operation)operation) ==
            0) {
            continue;
        }
        (void)run_one(backend, (cpu_benchmark_operation)operation, config, summary);
    }
    print_backend_summary(backend, summary);
}

static void print_summary(const machine_information *machine, const report_summary *summary) {
    printf("\nConfronto hardware\n");
    printf("Totale: %zu kernel riusciti, %zu falliti.\n", summary->successful_results,
           summary->failed_results);
    if (summary->has_fastest_matmul != 0) {
        printf("Matmul F32 piu' veloce nel profilo: %s in %.3f ms end-to-end.\n",
               runtime_benchmark_backend_name(summary->fastest_matmul_backend),
               summary->fastest_matmul_seconds * 1000.0);
    }
    if (summary->cpu_f32_matmul_seconds > 0.0 && summary->metal_f32_matmul_seconds > 0.0) {
        const double speedup = summary->cpu_f32_matmul_seconds / summary->metal_f32_matmul_seconds;
        if (speedup >= 1.0) {
            printf("Matmul FP32: Metal e' %.2fx piu' veloce della CPU su questa forma.\n", speedup);
        } else {
            printf("Matmul FP32: la CPU e' %.2fx piu' veloce di Metal su questa forma.\n",
                   1.0 / speedup);
        }
    } else if (machine->metal_available == 0) {
        printf("Metal non disponibile: sono stati misurati solo i kernel CPU portabili.\n");
    }
    printf("I risultati valgono per questa macchina e questo profilo; temperatura e carico "
           "del sistema possono modificarli.\n");
}

int main(int argc, char **argv) {
    report_profile profile = REPORT_PROFILE_STANDARD;
    const int parse_status = parse_arguments(argc, argv, &profile);
    if (parse_status == 2) {
        return EXIT_SUCCESS;
    }
    if (parse_status == 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    machine_information machine = detect_machine();
    cpu_benchmark_config config = profile_config(profile);
    report_summary summary = {0};
    print_machine(&machine, profile, &config);
    run_backend(&machine, RUNTIME_BENCHMARK_CPU, &config, &summary);
    if (machine.metal_available != 0) {
        run_backend(&machine, RUNTIME_BENCHMARK_METAL, &config, &summary);
    }
    print_summary(&machine, &summary);
    return summary.failed_results == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
}
