#ifndef LLM_LAB_CPU_ATOMIC_H
#define LLM_LAB_CPU_ATOMIC_H

#include <stddef.h>

#include <stdatomic.h>

typedef _Atomic size_t llm_cpu_atomic_size;
typedef _Atomic int llm_cpu_atomic_int;

static inline void llm_cpu_atomic_size_init(llm_cpu_atomic_size *atomic, size_t value) {
    atomic_init(atomic, value);
}

static inline size_t llm_cpu_atomic_size_load(llm_cpu_atomic_size *atomic) {
    return atomic_load(atomic);
}

static inline void llm_cpu_atomic_size_store(llm_cpu_atomic_size *atomic, size_t value) {
    atomic_store(atomic, value);
}

static inline int llm_cpu_atomic_size_compare_exchange_weak(llm_cpu_atomic_size *atomic,
                                                            size_t *expected, size_t desired) {
    return atomic_compare_exchange_weak(atomic, expected, desired);
}

static inline void llm_cpu_atomic_int_init(llm_cpu_atomic_int *atomic, int value) {
    atomic_init(atomic, value);
}

static inline int llm_cpu_atomic_int_load(llm_cpu_atomic_int *atomic) {
    return atomic_load(atomic);
}

static inline void llm_cpu_atomic_int_store(llm_cpu_atomic_int *atomic, int value) {
    atomic_store(atomic, value);
}

static inline int llm_cpu_atomic_int_compare_exchange_strong(llm_cpu_atomic_int *atomic,
                                                             int *expected, int desired) {
    return atomic_compare_exchange_strong(atomic, expected, desired);
}

#endif
