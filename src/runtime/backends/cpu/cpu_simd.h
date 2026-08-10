#ifndef LLM_LAB_CPU_SIMD_H
#define LLM_LAB_CPU_SIMD_H

#include <stddef.h>

void llm_cpu_simd_add_f32(const float *left, const float *right, float *output, size_t value_count);
void llm_cpu_simd_multiply_f32(const float *left, const float *right, float *output,
                               size_t value_count);
void llm_cpu_simd_scale_f32(const float *input, float scale, float *output, size_t value_count);
void llm_cpu_simd_axpy_f32(const float *input, float scale, float *output, size_t value_count);

#endif
