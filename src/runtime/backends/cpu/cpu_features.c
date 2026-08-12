#include "cpu_internal.h"

#include <unistd.h>

size_t llm_cpu_detect_thread_count(void) {
    size_t detected = 1U;
    const long processor_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (processor_count > 0L) {
        detected = (size_t)processor_count;
    }
    return detected > LLM_CPU_MAX_THREADS ? LLM_CPU_MAX_THREADS : detected;
}
