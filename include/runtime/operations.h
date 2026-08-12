#ifndef LLM_LAB_RUNTIME_OPERATIONS_H
#define LLM_LAB_RUNTIME_OPERATIONS_H

#include "runtime/tensor.h"

/** Selects whether either stored matrix is read with its two dimensions exchanged. */
typedef struct llm_matmul_options {
    int transpose_left;
    int transpose_right;
} llm_matmul_options;

/** Controls scaling and absolute query positions for causal grouped-query attention. */
typedef struct llm_attention_options {
    float scale;
    size_t query_position_offset;
} llm_attention_options;

/** Hyperparameters and step state for one FP32 AdamW parameter update. */
typedef struct llm_adamw_options {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float gradient_scale;
    unsigned long long step;
} llm_adamw_options;

/** Adds two distinct FP32 tensors with identical shapes. */
llm_status llm_add(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                   llm_tensor *output);

/** Multiplies two distinct FP32 tensors element by element. */
llm_status llm_multiply(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                        llm_tensor *output);

/** Multiplies every FP32 input value by a scalar. */
llm_status llm_scale(llm_backend *backend, const llm_tensor *input, float scale,
                     llm_tensor *output);

/** Reduces the last FP32 dimension by summation. */
llm_status llm_reduce_sum_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Reduces the last FP32 dimension by selecting its maximum. */
llm_status llm_reduce_max_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Reduces the last FP32 dimension to the mean of its squared values. */
llm_status llm_reduce_mean_square_last(llm_backend *backend, const llm_tensor *input,
                                       llm_tensor *output);

/** Computes a two-dimensional FP32 matrix product: [M,K] x [K,N] -> [M,N]. */
llm_status llm_matmul(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                      llm_tensor *output);

/** Computes a two-dimensional FP32 matrix product with optional logical transposes. */
llm_status llm_matmul_ex(llm_backend *backend, const llm_tensor *left, const llm_tensor *right,
                         const llm_matmul_options *options, llm_tensor *output);

/** Multiplies FP16 or BF16 matrices while accumulating into an FP32 output. */
llm_status llm_matmul_mixed_f32(llm_backend *backend, const llm_tensor *left,
                                const llm_tensor *right, llm_tensor *output);

/** Selects rows from an FP32 [V,C] table using a U32 tensor of row indices. */
llm_status llm_gather_rows(llm_backend *backend, const llm_tensor *table, const llm_tensor *indices,
                           llm_tensor *output);

/** Accumulates FP32 source rows into an FP32 [V,C] table using U32 indices. */
llm_status llm_scatter_add_rows(llm_backend *backend, const llm_tensor *source,
                                const llm_tensor *indices, llm_tensor *table);

/** Adds an FP32 source tensor into a distinct destination tensor in place. */
llm_status llm_accumulate(llm_backend *backend, const llm_tensor *source, llm_tensor *destination);

/** Applies SiLU element by element: output = input / (1 + exp(-input)). */
llm_status llm_silu(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Computes the SiLU gradient with respect to its input. */
llm_status llm_silu_backward(llm_backend *backend, const llm_tensor *input,
                             const llm_tensor *output_gradient, llm_tensor *input_gradient);

/** Applies weighted RMS normalization over the last dimension of an FP32 tensor. */
llm_status llm_rms_norm(llm_backend *backend, const llm_tensor *input, const llm_tensor *weight,
                        float epsilon, llm_tensor *output);

/** Computes input and weight gradients for weighted RMS normalization. */
llm_status llm_rms_norm_backward(llm_backend *backend, const llm_tensor *input,
                                 const llm_tensor *weight, const llm_tensor *output_gradient,
                                 float epsilon, llm_tensor *input_gradient,
                                 llm_tensor *weight_gradient);

/** Applies RoPE to adjacent pairs in an FP32 [B,S,H,D] tensor using [P,D/2] tables. */
llm_status llm_rope(llm_backend *backend, const llm_tensor *input, const llm_tensor *cos_table,
                    const llm_tensor *sin_table, size_t position_offset, llm_tensor *output);

/** Applies the inverse RoPE rotation to an output gradient. */
llm_status llm_rope_backward(llm_backend *backend, const llm_tensor *output_gradient,
                             const llm_tensor *cos_table, const llm_tensor *sin_table,
                             size_t position_offset, llm_tensor *input_gradient);

/** Computes FP32 causal grouped-query attention for [B,S,H,D] tensors. */
llm_status llm_attention_forward(llm_backend *backend, const llm_tensor *query,
                                 const llm_tensor *key, const llm_tensor *value,
                                 const llm_attention_options *options, llm_tensor *output);

/** Computes query, key and value gradients for causal grouped-query attention. */
llm_status llm_attention_backward(llm_backend *backend, const llm_tensor *query,
                                  const llm_tensor *key, const llm_tensor *value,
                                  const llm_tensor *output_gradient,
                                  const llm_attention_options *options, llm_tensor *query_gradient,
                                  llm_tensor *key_gradient, llm_tensor *value_gradient);

/** Computes a numerically stable FP32 softmax over the last dimension. */
llm_status llm_softmax_last(llm_backend *backend, const llm_tensor *input, llm_tensor *output);

/** Computes mean next-token cross-entropy from FP32 [N,V] logits and U32 [N] targets. */
llm_status llm_cross_entropy_forward(llm_backend *backend, const llm_tensor *logits,
                                     const llm_tensor *targets, llm_tensor *loss);

/** Computes the FP32 [N,V] gradient of mean cross-entropy with respect to logits. */
llm_status llm_cross_entropy_backward(llm_backend *backend, const llm_tensor *logits,
                                      const llm_tensor *targets, llm_tensor *logits_gradient);

/** Updates an FP32 parameter and its two FP32 AdamW moment tensors in place. */
llm_status llm_adamw_update(llm_backend *backend, llm_tensor *parameter, const llm_tensor *gradient,
                            llm_tensor *first_moment, llm_tensor *second_moment,
                            const llm_adamw_options *options);

#endif
