#include "cpu_simd.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define LLM_CPU_SIMD_NEON 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define LLM_CPU_SIMD_SSE2 1
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
        vst1q_f32(output + index, vmlaq_n_f32(output_values, input_values, scale));
    }
#elif defined(LLM_CPU_SIMD_SSE2)
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
