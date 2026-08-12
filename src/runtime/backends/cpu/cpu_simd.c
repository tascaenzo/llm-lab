#include "cpu_simd.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#define LLM_CPU_SIMD_NEON 1
#elif defined(__SSE2__)
#include <immintrin.h>
#define LLM_CPU_SIMD_SSE2 1
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define LLM_CPU_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define LLM_CPU_TARGET_AVX512 __attribute__((target("avx512f,fma")))

static int llm_cpu_has_avx2_fma(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

static int llm_cpu_has_avx512_fma(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
}

LLM_CPU_TARGET_AVX2
static size_t llm_cpu_axpy_avx2(const float *input, float scale, float *output,
                                size_t value_count) {
    const __m256 scale_values = _mm256_set1_ps(scale);
    size_t index = 0U;
    for (; value_count - index >= 8U; index += 8U) {
        const __m256 input_values = _mm256_loadu_ps(input + index);
        const __m256 output_values = _mm256_loadu_ps(output + index);
        _mm256_storeu_ps(output + index,
                         _mm256_fmadd_ps(input_values, scale_values, output_values));
    }
    return index;
}

LLM_CPU_TARGET_AVX512
static size_t llm_cpu_axpy_avx512(const float *input, float scale, float *output,
                                  size_t value_count) {
    const __m512 scale_values = _mm512_set1_ps(scale);
    size_t index = 0U;
    for (; value_count - index >= 16U; index += 16U) {
        const __m512 input_values = _mm512_loadu_ps(input + index);
        const __m512 output_values = _mm512_loadu_ps(output + index);
        _mm512_storeu_ps(output + index,
                         _mm512_fmadd_ps(input_values, scale_values, output_values));
    }
    return index;
}

LLM_CPU_TARGET_AVX2
static size_t llm_cpu_axpy4_avx2(const float *input, const float scales[4], float *const outputs[4],
                                 size_t value_count) {
    const __m256 scale_values[4] = {_mm256_set1_ps(scales[0]), _mm256_set1_ps(scales[1]),
                                    _mm256_set1_ps(scales[2]), _mm256_set1_ps(scales[3])};
    size_t index = 0U;
    for (; value_count - index >= 8U; index += 8U) {
        const __m256 input_values = _mm256_loadu_ps(input + index);
        for (size_t row = 0U; row < 4U; ++row) {
            const __m256 output_values = _mm256_loadu_ps(outputs[row] + index);
            _mm256_storeu_ps(outputs[row] + index,
                             _mm256_fmadd_ps(input_values, scale_values[row], output_values));
        }
    }
    return index;
}

LLM_CPU_TARGET_AVX512
static size_t llm_cpu_axpy4_avx512(const float *input, const float scales[4],
                                   float *const outputs[4], size_t value_count) {
    const __m512 scale_values[4] = {_mm512_set1_ps(scales[0]), _mm512_set1_ps(scales[1]),
                                    _mm512_set1_ps(scales[2]), _mm512_set1_ps(scales[3])};
    size_t index = 0U;
    for (; value_count - index >= 16U; index += 16U) {
        const __m512 input_values = _mm512_loadu_ps(input + index);
        for (size_t row = 0U; row < 4U; ++row) {
            const __m512 output_values = _mm512_loadu_ps(outputs[row] + index);
            _mm512_storeu_ps(outputs[row] + index,
                             _mm512_fmadd_ps(input_values, scale_values[row], output_values));
        }
    }
    return index;
}
#endif

void llm_cpu_simd_add_f32(const float *left, const float *right, float *output,
                          size_t value_count) {
    size_t index = 0U;
#if defined(LLM_CPU_SIMD_NEON)
    for (; value_count - index >= 4U; index += 4U) {
        const float32x4_t left_values = vld1q_f32(left + index);
        const float32x4_t right_values = vld1q_f32(right + index);
        vst1q_f32(output + index, vaddq_f32(left_values, right_values));
    }
#elif defined(LLM_CPU_SIMD_SSE2)
    for (; value_count - index >= 4U; index += 4U) {
        const __m128 left_values = _mm_loadu_ps(left + index);
        const __m128 right_values = _mm_loadu_ps(right + index);
        _mm_storeu_ps(output + index, _mm_add_ps(left_values, right_values));
    }
#endif
    for (; index < value_count; ++index) {
        output[index] = left[index] + right[index];
    }
}

void llm_cpu_simd_multiply_f32(const float *left, const float *right, float *output,
                               size_t value_count) {
    size_t index = 0U;
#if defined(LLM_CPU_SIMD_NEON)
    for (; value_count - index >= 4U; index += 4U) {
        const float32x4_t left_values = vld1q_f32(left + index);
        const float32x4_t right_values = vld1q_f32(right + index);
        vst1q_f32(output + index, vmulq_f32(left_values, right_values));
    }
#elif defined(LLM_CPU_SIMD_SSE2)
    for (; value_count - index >= 4U; index += 4U) {
        const __m128 left_values = _mm_loadu_ps(left + index);
        const __m128 right_values = _mm_loadu_ps(right + index);
        _mm_storeu_ps(output + index, _mm_mul_ps(left_values, right_values));
    }
#endif
    for (; index < value_count; ++index) {
        output[index] = left[index] * right[index];
    }
}

void llm_cpu_simd_scale_f32(const float *input, float scale, float *output, size_t value_count) {
    size_t index = 0U;
#if defined(LLM_CPU_SIMD_NEON)
    for (; value_count - index >= 4U; index += 4U) {
        const float32x4_t input_values = vld1q_f32(input + index);
        vst1q_f32(output + index, vmulq_n_f32(input_values, scale));
    }
#elif defined(LLM_CPU_SIMD_SSE2)
    const __m128 scale_values = _mm_set1_ps(scale);
    for (; value_count - index >= 4U; index += 4U) {
        const __m128 input_values = _mm_loadu_ps(input + index);
        _mm_storeu_ps(output + index, _mm_mul_ps(input_values, scale_values));
    }
#endif
    for (; index < value_count; ++index) {
        output[index] = input[index] * scale;
    }
}

void llm_cpu_simd_axpy_f32(const float *input, float scale, float *output, size_t value_count) {
    size_t index = 0U;
#if defined(LLM_CPU_SIMD_NEON)
    for (; value_count - index >= 4U; index += 4U) {
        const float32x4_t input_values = vld1q_f32(input + index);
        const float32x4_t output_values = vld1q_f32(output + index);
        vst1q_f32(output + index, vfmaq_n_f32(output_values, input_values, scale));
    }
#elif defined(LLM_CPU_SIMD_SSE2)
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (llm_cpu_has_avx512_fma() != 0) {
        index = llm_cpu_axpy_avx512(input, scale, output, value_count);
    } else if (llm_cpu_has_avx2_fma() != 0) {
        index = llm_cpu_axpy_avx2(input, scale, output, value_count);
    }
#endif
    const __m128 scale_values = _mm_set1_ps(scale);
    for (; value_count - index >= 4U; index += 4U) {
        const __m128 input_values = _mm_loadu_ps(input + index);
        const __m128 output_values = _mm_loadu_ps(output + index);
        _mm_storeu_ps(output + index,
                      _mm_add_ps(output_values, _mm_mul_ps(input_values, scale_values)));
    }
#endif
    for (; index < value_count; ++index) {
        output[index] += input[index] * scale;
    }
}

void llm_cpu_simd_axpy4_f32(const float *input, const float scales[4], float *const outputs[4],
                            size_t value_count) {
    size_t index = 0U;
#if defined(LLM_CPU_SIMD_NEON)
    for (; value_count - index >= 4U; index += 4U) {
        const float32x4_t input_values = vld1q_f32(input + index);
        for (size_t row = 0U; row < 4U; ++row) {
            const float32x4_t output_values = vld1q_f32(outputs[row] + index);
            vst1q_f32(outputs[row] + index, vfmaq_n_f32(output_values, input_values, scales[row]));
        }
    }
#elif defined(LLM_CPU_SIMD_SSE2)
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (llm_cpu_has_avx512_fma() != 0) {
        index = llm_cpu_axpy4_avx512(input, scales, outputs, value_count);
    } else if (llm_cpu_has_avx2_fma() != 0) {
        index = llm_cpu_axpy4_avx2(input, scales, outputs, value_count);
    }
#endif
    for (; value_count - index >= 4U; index += 4U) {
        const __m128 input_values = _mm_loadu_ps(input + index);
        for (size_t row = 0U; row < 4U; ++row) {
            const __m128 output_values = _mm_loadu_ps(outputs[row] + index);
            const __m128 scale_values = _mm_set1_ps(scales[row]);
            _mm_storeu_ps(outputs[row] + index,
                          _mm_add_ps(output_values, _mm_mul_ps(input_values, scale_values)));
        }
    }
#endif
    for (; index < value_count; ++index) {
        for (size_t row = 0U; row < 4U; ++row) {
            outputs[row][index] += input[index] * scales[row];
        }
    }
}
