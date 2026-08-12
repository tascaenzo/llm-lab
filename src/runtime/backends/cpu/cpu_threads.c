#include <stdlib.h>

#include "cpu_threads.h"

typedef struct cpu_thread_start {
    llm_cpu_thread_fn function;
    void *context;
} cpu_thread_start;

static void *cpu_thread_entry(void *argument) {
    cpu_thread_start *start = argument;
    llm_cpu_thread_fn function = start->function;
    void *context = start->context;
    free(start);
    (void)function(context);
    return NULL;
}

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
    const int status = pthread_create(thread, NULL, cpu_thread_entry, start);
    if (status != 0) {
        free(start);
    }
    return status;
}

int llm_cpu_thread_join(llm_cpu_thread thread) { return pthread_join(thread, NULL); }

int llm_cpu_mutex_init(llm_cpu_mutex *mutex) { return pthread_mutex_init(mutex, NULL); }

void llm_cpu_mutex_destroy(llm_cpu_mutex *mutex) { (void)pthread_mutex_destroy(mutex); }

int llm_cpu_mutex_lock(llm_cpu_mutex *mutex) { return pthread_mutex_lock(mutex); }

int llm_cpu_mutex_unlock(llm_cpu_mutex *mutex) { return pthread_mutex_unlock(mutex); }

int llm_cpu_condition_init(llm_cpu_condition *condition) {
    return pthread_cond_init(condition, NULL);
}

void llm_cpu_condition_destroy(llm_cpu_condition *condition) {
    (void)pthread_cond_destroy(condition);
}

int llm_cpu_condition_wait(llm_cpu_condition *condition, llm_cpu_mutex *mutex) {
    return pthread_cond_wait(condition, mutex);
}

int llm_cpu_condition_signal(llm_cpu_condition *condition) {
    return pthread_cond_signal(condition);
}

int llm_cpu_condition_broadcast(llm_cpu_condition *condition) {
    return pthread_cond_broadcast(condition);
}
