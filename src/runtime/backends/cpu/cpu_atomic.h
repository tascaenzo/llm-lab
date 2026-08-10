#ifndef LLM_LAB_CPU_ATOMIC_H
#define LLM_LAB_CPU_ATOMIC_H

#include <stddef.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef struct llm_cpu_atomic_size {
    volatile LONG64 value;
} llm_cpu_atomic_size;

typedef struct llm_cpu_atomic_int {
    volatile LONG value;
} llm_cpu_atomic_int;

static inline void llm_cpu_atomic_size_init(llm_cpu_atomic_size *atomic, size_t value) {
    atomic->value = (LONG64)value;
}

static inline size_t llm_cpu_atomic_size_load(llm_cpu_atomic_size *atomic) {
    return (size_t)InterlockedCompareExchange64(&atomic->value, 0, 0);
}

static inline void llm_cpu_atomic_size_store(llm_cpu_atomic_size *atomic, size_t value) {
    (void)InterlockedExchange64(&atomic->value, (LONG64)value);
}

static inline int llm_cpu_atomic_size_compare_exchange_weak(llm_cpu_atomic_size *atomic,
                                                            size_t *expected, size_t desired) {
    const LONG64 original =
        InterlockedCompareExchange64(&atomic->value, (LONG64)desired, (LONG64)*expected);
    if (original == (LONG64)*expected) {
        return 1;
    }
    *expected = (size_t)original;
    return 0;
}

static inline void llm_cpu_atomic_int_init(llm_cpu_atomic_int *atomic, int value) {
    atomic->value = (LONG)value;
}

static inline int llm_cpu_atomic_int_load(llm_cpu_atomic_int *atomic) {
    return (int)InterlockedCompareExchange(&atomic->value, 0, 0);
}

static inline void llm_cpu_atomic_int_store(llm_cpu_atomic_int *atomic, int value) {
    (void)InterlockedExchange(&atomic->value, (LONG)value);
}

static inline int llm_cpu_atomic_int_compare_exchange_strong(llm_cpu_atomic_int *atomic,
                                                             int *expected, int desired) {
    const LONG original =
        InterlockedCompareExchange(&atomic->value, (LONG)desired, (LONG)*expected);
    if (original == (LONG)*expected) {
        return 1;
    }
    *expected = (int)original;
    return 0;
}

#else

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

#endif
