#ifndef LLM_LAB_CPU_EXECUTOR_H
#define LLM_LAB_CPU_EXECUTOR_H

#include <stddef.h>

#include "runtime/types.h"

typedef struct llm_cpu_executor llm_cpu_executor;

typedef llm_status (*llm_cpu_range_fn)(void *context, size_t begin, size_t end);

llm_status llm_cpu_executor_create(size_t thread_count, llm_cpu_executor **out_executor);
void llm_cpu_executor_destroy(llm_cpu_executor *executor);
size_t llm_cpu_executor_thread_count(const llm_cpu_executor *executor);

llm_status llm_cpu_parallel_for(llm_cpu_executor *executor, size_t item_count,
                                size_t minimum_items_per_task, llm_cpu_range_fn function,
                                void *context);

#endif
