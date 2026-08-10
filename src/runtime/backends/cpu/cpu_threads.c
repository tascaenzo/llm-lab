#include <stdlib.h>

#include "cpu_threads.h"

typedef struct cpu_thread_start {
    llm_cpu_thread_fn function;
    void *context;
} cpu_thread_start;

#ifdef _WIN32
static DWORD WINAPI cpu_thread_entry(LPVOID argument) {
    cpu_thread_start *start = argument;
    llm_cpu_thread_fn function = start->function;
    void *context = start->context;
    free(start);
    return (DWORD)function(context);
}
#else
static void *cpu_thread_entry(void *argument) {
    cpu_thread_start *start = argument;
    llm_cpu_thread_fn function = start->function;
    void *context = start->context;
    free(start);
    (void)function(context);
    return NULL;
}
#endif

int llm_cpu_thread_create(llm_cpu_thread *thread, llm_cpu_thread_fn function, void *context) {
    if (thread == NULL || function == NULL) {
        return -1;
    }
    cpu_thread_start *start = malloc(sizeof(*start));
    if (start == NULL) {
        return -1;
    }
    start->function = function;
    start->context = context;
#ifdef _WIN32
    *thread = CreateThread(NULL, 0U, cpu_thread_entry, start, 0U, NULL);
    if (*thread == NULL) {
        free(start);
        return -1;
    }
    return 0;
#else
    const int status = pthread_create(thread, NULL, cpu_thread_entry, start);
    if (status != 0) {
        free(start);
    }
    return status;
#endif
}

int llm_cpu_thread_join(llm_cpu_thread thread) {
#ifdef _WIN32
    const DWORD wait_status = WaitForSingleObject(thread, INFINITE);
    const BOOL close_status = CloseHandle(thread);
    return wait_status == WAIT_OBJECT_0 && close_status != 0 ? 0 : -1;
#else
    return pthread_join(thread, NULL);
#endif
}

int llm_cpu_mutex_init(llm_cpu_mutex *mutex) {
#ifdef _WIN32
    InitializeCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_init(mutex, NULL);
#endif
}

void llm_cpu_mutex_destroy(llm_cpu_mutex *mutex) {
#ifdef _WIN32
    DeleteCriticalSection(mutex);
#else
    (void)pthread_mutex_destroy(mutex);
#endif
}

int llm_cpu_mutex_lock(llm_cpu_mutex *mutex) {
#ifdef _WIN32
    EnterCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_lock(mutex);
#endif
}

int llm_cpu_mutex_unlock(llm_cpu_mutex *mutex) {
#ifdef _WIN32
    LeaveCriticalSection(mutex);
    return 0;
#else
    return pthread_mutex_unlock(mutex);
#endif
}

int llm_cpu_condition_init(llm_cpu_condition *condition) {
#ifdef _WIN32
    InitializeConditionVariable(condition);
    return 0;
#else
    return pthread_cond_init(condition, NULL);
#endif
}

void llm_cpu_condition_destroy(llm_cpu_condition *condition) {
#ifdef _WIN32
    (void)condition;
#else
    (void)pthread_cond_destroy(condition);
#endif
}

int llm_cpu_condition_wait(llm_cpu_condition *condition, llm_cpu_mutex *mutex) {
#ifdef _WIN32
    return SleepConditionVariableCS(condition, mutex, INFINITE) != 0 ? 0 : -1;
#else
    return pthread_cond_wait(condition, mutex);
#endif
}

int llm_cpu_condition_signal(llm_cpu_condition *condition) {
#ifdef _WIN32
    WakeConditionVariable(condition);
    return 0;
#else
    return pthread_cond_signal(condition);
#endif
}

int llm_cpu_condition_broadcast(llm_cpu_condition *condition) {
#ifdef _WIN32
    WakeAllConditionVariable(condition);
    return 0;
#else
    return pthread_cond_broadcast(condition);
#endif
}
