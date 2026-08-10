#include <stdlib.h>

#include "cpu_atomic.h"
#include "cpu_executor.h"
#include "cpu_threads.h"

#define LLM_CPU_TASKS_PER_THREAD 4U

struct llm_cpu_executor {
    llm_cpu_mutex state_mutex;
    llm_cpu_mutex submission_mutex;
    llm_cpu_condition work_available;
    llm_cpu_condition work_complete;
    llm_cpu_thread *workers;
    size_t worker_count;
    int stopping;
    size_t generation;
    llm_cpu_range_fn function;
    void *function_context;
    size_t item_count;
    size_t chunk_size;
    llm_cpu_atomic_size next_item;
    size_t completed_workers;
    llm_cpu_atomic_int status;
};

static llm_status execute_available_chunks(llm_cpu_executor *executor) {
    for (;;) {
        if ((llm_status)llm_cpu_atomic_int_load(&executor->status) != LLM_OK) {
            return LLM_OK;
        }

        size_t begin = llm_cpu_atomic_size_load(&executor->next_item);
        size_t end = 0U;
        do {
            if (begin >= executor->item_count) {
                return LLM_OK;
            }
            end = executor->item_count - begin < executor->chunk_size
                      ? executor->item_count
                      : begin + executor->chunk_size;
        } while (llm_cpu_atomic_size_compare_exchange_weak(&executor->next_item, &begin, end) == 0);

        const llm_status status = executor->function(executor->function_context, begin, end);
        if (status != LLM_OK) {
            int expected = LLM_OK;
            if (llm_cpu_atomic_int_compare_exchange_strong(&executor->status, &expected,
                                                           (int)status) != 0) {
                llm_cpu_atomic_size_store(&executor->next_item, executor->item_count);
            }
            return LLM_OK;
        }
    }
}

static int cpu_worker_main(void *argument) {
    llm_cpu_executor *executor = argument;
    size_t observed_generation = 0U;

    if (llm_cpu_mutex_lock(&executor->state_mutex) != 0) {
        return EXIT_FAILURE;
    }
    for (;;) {
        while (executor->stopping == 0 && executor->generation == observed_generation) {
            if (llm_cpu_condition_wait(&executor->work_available, &executor->state_mutex) != 0) {
                (void)llm_cpu_mutex_unlock(&executor->state_mutex);
                return EXIT_FAILURE;
            }
        }
        if (executor->stopping != 0) {
            (void)llm_cpu_mutex_unlock(&executor->state_mutex);
            return EXIT_SUCCESS;
        }

        observed_generation = executor->generation;
        (void)llm_cpu_mutex_unlock(&executor->state_mutex);
        const llm_status execution_status = execute_available_chunks(executor);

        if (llm_cpu_mutex_lock(&executor->state_mutex) != 0) {
            return EXIT_FAILURE;
        }
        if (execution_status != LLM_OK) {
            int expected = LLM_OK;
            if (llm_cpu_atomic_int_compare_exchange_strong(&executor->status, &expected,
                                                           (int)execution_status) != 0) {
                llm_cpu_atomic_size_store(&executor->next_item, executor->item_count);
            }
        }
        ++executor->completed_workers;
        if (executor->completed_workers == executor->worker_count) {
            (void)llm_cpu_condition_signal(&executor->work_complete);
        }
    }
}

static void stop_and_join_workers(llm_cpu_executor *executor) {
    if (llm_cpu_mutex_lock(&executor->state_mutex) == 0) {
        executor->stopping = 1;
        (void)llm_cpu_condition_broadcast(&executor->work_available);
        (void)llm_cpu_mutex_unlock(&executor->state_mutex);
    }
    for (size_t index = 0U; index < executor->worker_count; ++index) {
        (void)llm_cpu_thread_join(executor->workers[index]);
    }
}

llm_status llm_cpu_executor_create(size_t thread_count, llm_cpu_executor **out_executor) {
    if (thread_count == 0U || out_executor == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_executor = NULL;

    llm_cpu_executor *executor = calloc(1U, sizeof(*executor));
    if (executor == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    if (llm_cpu_mutex_init(&executor->state_mutex) != 0) {
        free(executor);
        return LLM_BACKEND_ERROR;
    }
    if (llm_cpu_mutex_init(&executor->submission_mutex) != 0) {
        llm_cpu_mutex_destroy(&executor->state_mutex);
        free(executor);
        return LLM_BACKEND_ERROR;
    }
    if (llm_cpu_condition_init(&executor->work_available) != 0) {
        llm_cpu_mutex_destroy(&executor->submission_mutex);
        llm_cpu_mutex_destroy(&executor->state_mutex);
        free(executor);
        return LLM_BACKEND_ERROR;
    }
    if (llm_cpu_condition_init(&executor->work_complete) != 0) {
        llm_cpu_condition_destroy(&executor->work_available);
        llm_cpu_mutex_destroy(&executor->submission_mutex);
        llm_cpu_mutex_destroy(&executor->state_mutex);
        free(executor);
        return LLM_BACKEND_ERROR;
    }

    const size_t requested_workers = thread_count - 1U;
    if (requested_workers > 0U) {
        executor->workers = calloc(requested_workers, sizeof(*executor->workers));
        if (executor->workers == NULL) {
            llm_cpu_condition_destroy(&executor->work_complete);
            llm_cpu_condition_destroy(&executor->work_available);
            llm_cpu_mutex_destroy(&executor->submission_mutex);
            llm_cpu_mutex_destroy(&executor->state_mutex);
            free(executor);
            return LLM_ALLOCATION_FAILED;
        }
    }

    for (size_t index = 0U; index < requested_workers; ++index) {
        if (llm_cpu_thread_create(&executor->workers[index], cpu_worker_main, executor) != 0) {
            stop_and_join_workers(executor);
            free(executor->workers);
            llm_cpu_condition_destroy(&executor->work_complete);
            llm_cpu_condition_destroy(&executor->work_available);
            llm_cpu_mutex_destroy(&executor->submission_mutex);
            llm_cpu_mutex_destroy(&executor->state_mutex);
            free(executor);
            return LLM_BACKEND_ERROR;
        }
        ++executor->worker_count;
    }

    llm_cpu_atomic_size_init(&executor->next_item, 0U);
    llm_cpu_atomic_int_init(&executor->status, LLM_OK);
    *out_executor = executor;
    return LLM_OK;
}

void llm_cpu_executor_destroy(llm_cpu_executor *executor) {
    if (executor == NULL) {
        return;
    }
    if (llm_cpu_mutex_lock(&executor->submission_mutex) == 0) {
        stop_and_join_workers(executor);
        (void)llm_cpu_mutex_unlock(&executor->submission_mutex);
    }
    free(executor->workers);
    llm_cpu_condition_destroy(&executor->work_complete);
    llm_cpu_condition_destroy(&executor->work_available);
    llm_cpu_mutex_destroy(&executor->submission_mutex);
    llm_cpu_mutex_destroy(&executor->state_mutex);
    free(executor);
}

size_t llm_cpu_executor_thread_count(const llm_cpu_executor *executor) {
    return executor == NULL ? 0U : executor->worker_count + 1U;
}

llm_status llm_cpu_parallel_for(llm_cpu_executor *executor, size_t item_count,
                                size_t minimum_items_per_task, llm_cpu_range_fn function,
                                void *context) {
    if (executor == NULL || item_count == 0U || minimum_items_per_task == 0U || function == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    if (executor->worker_count == 0U || item_count <= minimum_items_per_task) {
        return function(context, 0U, item_count);
    }
    if (llm_cpu_mutex_lock(&executor->submission_mutex) != 0) {
        return LLM_BACKEND_ERROR;
    }
    if (llm_cpu_mutex_lock(&executor->state_mutex) != 0) {
        (void)llm_cpu_mutex_unlock(&executor->submission_mutex);
        return LLM_BACKEND_ERROR;
    }

    const size_t total_threads = executor->worker_count + 1U;
    const size_t target_tasks = total_threads * LLM_CPU_TASKS_PER_THREAD;
    size_t chunk_size = item_count / target_tasks;
    if (item_count % target_tasks != 0U) {
        ++chunk_size;
    }
    if (chunk_size < minimum_items_per_task) {
        chunk_size = minimum_items_per_task;
    }

    executor->function = function;
    executor->function_context = context;
    executor->item_count = item_count;
    executor->chunk_size = chunk_size;
    llm_cpu_atomic_size_store(&executor->next_item, 0U);
    executor->completed_workers = 0U;
    llm_cpu_atomic_int_store(&executor->status, LLM_OK);
    ++executor->generation;
    (void)llm_cpu_condition_broadcast(&executor->work_available);
    (void)llm_cpu_mutex_unlock(&executor->state_mutex);

    const llm_status caller_status = execute_available_chunks(executor);

    if (llm_cpu_mutex_lock(&executor->state_mutex) != 0) {
        (void)llm_cpu_mutex_unlock(&executor->submission_mutex);
        return LLM_BACKEND_ERROR;
    }
    if (caller_status != LLM_OK) {
        int expected = LLM_OK;
        if (llm_cpu_atomic_int_compare_exchange_strong(&executor->status, &expected,
                                                       (int)caller_status) != 0) {
            llm_cpu_atomic_size_store(&executor->next_item, executor->item_count);
        }
    }
    while (executor->completed_workers < executor->worker_count) {
        if (llm_cpu_condition_wait(&executor->work_complete, &executor->state_mutex) != 0) {
            llm_cpu_atomic_int_store(&executor->status, LLM_BACKEND_ERROR);
            break;
        }
    }
    const llm_status status = (llm_status)llm_cpu_atomic_int_load(&executor->status);
    executor->function = NULL;
    executor->function_context = NULL;
    (void)llm_cpu_mutex_unlock(&executor->state_mutex);
    (void)llm_cpu_mutex_unlock(&executor->submission_mutex);
    return status;
}
