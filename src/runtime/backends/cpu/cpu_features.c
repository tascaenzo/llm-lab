#include "cpu_internal.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

size_t llm_cpu_detect_thread_count(void) {
    size_t detected = 1U;
#ifdef _WIN32
    const DWORD processor_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (processor_count > 0U) {
        detected = (size_t)processor_count;
    }
#elif defined(__unix__) || defined(__APPLE__)
    const long processor_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (processor_count > 0L) {
        detected = (size_t)processor_count;
    }
#endif
    return detected > LLM_CPU_MAX_THREADS ? LLM_CPU_MAX_THREADS : detected;
}
