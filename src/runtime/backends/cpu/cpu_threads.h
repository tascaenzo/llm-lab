#ifndef LLM_LAB_CPU_THREADS_H
#define LLM_LAB_CPU_THREADS_H

#include <pthread.h>
typedef pthread_t llm_cpu_thread;
typedef pthread_mutex_t llm_cpu_mutex;
typedef pthread_cond_t llm_cpu_condition;

typedef int (*llm_cpu_thread_fn)(void *context);

int llm_cpu_thread_create(llm_cpu_thread *thread, llm_cpu_thread_fn function, void *context);
int llm_cpu_thread_join(llm_cpu_thread thread);

int llm_cpu_mutex_init(llm_cpu_mutex *mutex);
void llm_cpu_mutex_destroy(llm_cpu_mutex *mutex);
int llm_cpu_mutex_lock(llm_cpu_mutex *mutex);
int llm_cpu_mutex_unlock(llm_cpu_mutex *mutex);

int llm_cpu_condition_init(llm_cpu_condition *condition);
void llm_cpu_condition_destroy(llm_cpu_condition *condition);
int llm_cpu_condition_wait(llm_cpu_condition *condition, llm_cpu_mutex *mutex);
int llm_cpu_condition_signal(llm_cpu_condition *condition);
int llm_cpu_condition_broadcast(llm_cpu_condition *condition);

#endif
