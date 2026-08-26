#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "model_internal.h"
#include "tokenizer/sha256.h"

#define LM_CHECKPOINT_V1_HEADER_SIZE 160U
#define LM_CHECKPOINT_V2_HEADER_SIZE 192U
/* Versions 3 and 4 share the same header layout: v4 only adds parameters
   (MLP and final norm) to the payload, not fields to the header. */
#define LM_CHECKPOINT_V3_V4_HEADER_SIZE 232U
/* Version 5 appends a SHA-256 of the fixed header and payload. */
#define LM_CHECKPOINT_V5_HEADER_SIZE 264U
#define LM_CHECKPOINT_CHECKSUM_OFFSET 232U
#define LM_CHECKPOINT_VERSION UINT32_C(5)

static const unsigned char checkpoint_magic[8] = {'L', 'L', 'M', 'C', 'K', 'P', 'T', '\n'};

static void store_u32(unsigned char *bytes, uint32_t value) {
    for (size_t index = 0U; index < 4U; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
}

static void store_u64(unsigned char *bytes, uint64_t value) {
    for (size_t index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
}

static uint32_t load_u32(const unsigned char *bytes) {
    uint32_t value = 0U;
    for (size_t index = 0U; index < 4U; ++index) {
        value |= (uint32_t)bytes[index] << (index * 8U);
    }
    return value;
}

static uint64_t load_u64(const unsigned char *bytes) {
    uint64_t value = 0U;
    for (size_t index = 0U; index < 8U; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static void store_f32(unsigned char *bytes, float value) {
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    store_u32(bytes, bits);
}

static float load_f32(const unsigned char *bytes) {
    const uint32_t bits = load_u32(bytes);
    float value = 0.0F;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int write_exact(FILE *file, const void *data, size_t byte_count) {
    return fwrite(data, 1U, byte_count, file) == byte_count;
}

static int read_exact(FILE *file, void *data, size_t byte_count) {
    return fread(data, 1U, byte_count, file) == byte_count;
}

static llm_status payload_size(const lm_model *model, uint64_t *out_size) {
    if (model == NULL || out_size == NULL) {
        return LLM_INVALID_ARGUMENT;
    }
    uint64_t total = 0U;
    for (size_t index = 0U; index < model->parameter_count; ++index) {
        const uint64_t count = model->parameters[index].value.element_count;
        if (count > UINT64_MAX / (UINT64_C(3) * sizeof(float)) ||
            total > UINT64_MAX - count * UINT64_C(3) * sizeof(float)) {
            return LLM_OVERFLOW;
        }
        total += count * UINT64_C(3) * sizeof(float);
    }
    *out_size = total;
    return LLM_OK;
}

static llm_status write_tensor(FILE *file, llm_backend *backend, const llm_tensor *tensor,
                               tokenizer_sha256_context *checksum) {
    if (tensor->element_count > SIZE_MAX / sizeof(float)) {
        return LLM_OVERFLOW;
    }
    const size_t byte_count = tensor->element_count * sizeof(float);
    float *values = malloc(byte_count);
    if (values == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    llm_status status = llm_tensor_read(backend, tensor, values, byte_count);
    if (status == LLM_OK) {
        for (size_t index = 0U; index < tensor->element_count; ++index) {
            if (isfinite(values[index]) == 0) {
                status = LLM_NUMERICAL_ERROR;
                break;
            }
        }
    }
    if (status == LLM_OK) {
        if (write_exact(file, values, byte_count) == 0) {
            status = LLM_BACKEND_ERROR;
        } else if (checksum != NULL) {
            tokenizer_sha256_update(checksum, (const unsigned char *)values, byte_count);
        }
    }
    free(values);
    return status;
}

static llm_status read_tensor(FILE *file, llm_backend *backend, llm_tensor *tensor,
                              tokenizer_sha256_context *checksum) {
    if (tensor->element_count > SIZE_MAX / sizeof(float)) {
        return LLM_OVERFLOW;
    }
    const size_t byte_count = tensor->element_count * sizeof(float);
    float *values = malloc(byte_count);
    if (values == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    llm_status status = read_exact(file, values, byte_count) != 0 ? LLM_OK : LLM_INVALID_ARGUMENT;
    if (status == LLM_OK) {
        for (size_t index = 0U; index < tensor->element_count; ++index) {
            if (isfinite(values[index]) == 0) {
                status = LLM_NUMERICAL_ERROR;
                break;
            }
        }
    }
    if (status == LLM_OK && checksum != NULL) {
        tokenizer_sha256_update(checksum, (const unsigned char *)values, byte_count);
    }
    if (status == LLM_OK) {
        status = llm_tensor_write(backend, tensor, values, byte_count);
    }
    free(values);
    return status;
}

static int checkpoint_header_is_supported(uint32_t version, uint32_t header_size) {
    return version >= 1U && version <= LM_CHECKPOINT_VERSION &&
           (header_size == LM_CHECKPOINT_V1_HEADER_SIZE ||
            header_size == LM_CHECKPOINT_V2_HEADER_SIZE ||
            header_size == LM_CHECKPOINT_V3_V4_HEADER_SIZE ||
            header_size == LM_CHECKPOINT_V5_HEADER_SIZE) &&
           (version < 5U || header_size == LM_CHECKPOINT_V5_HEADER_SIZE);
}

/** Flushes checkpoint bytes before rename so a successful save is durable. */
static llm_status sync_checkpoint_file(FILE *file) {
    if (fflush(file) != 0 || fsync(fileno(file)) != 0) {
        return LLM_BACKEND_ERROR;
    }
    return LLM_OK;
}

/** Makes the rename itself durable on the macOS/Linux filesystems supported by this project. */
static llm_status sync_parent_directory(const char *path) {
    const char *last_slash = strrchr(path, '/');
    const char *directory = ".";
    char *allocated_directory = NULL;
    if (last_slash != NULL) {
        const size_t length = last_slash == path ? 1U : (size_t)(last_slash - path);
        allocated_directory = malloc(length + 1U);
        if (allocated_directory == NULL) {
            return LLM_ALLOCATION_FAILED;
        }
        memcpy(allocated_directory, path, length);
        allocated_directory[length] = '\0';
        directory = allocated_directory;
    }
    const int descriptor = open(directory, O_RDONLY);
    free(allocated_directory);
    if (descriptor < 0) {
        return LLM_BACKEND_ERROR;
    }
    const int sync_result = fsync(descriptor);
    const int close_result = close(descriptor);
    return sync_result == 0 && close_result == 0 ? LLM_OK : LLM_BACKEND_ERROR;
}

static char *temporary_path(const char *path) {
    const size_t length = strlen(path);
    if (length > SIZE_MAX - sizeof(".part")) {
        return NULL;
    }
    char *result = malloc(length + sizeof(".part"));
    if (result != NULL) {
        (void)snprintf(result, length + sizeof(".part"), "%s.part", path);
    }
    return result;
}

static llm_status save_checkpoint(const lm_trainer *trainer, lm_dataset *dataset,
                                  lm_sft_dataset *sft_dataset, const char *path) {
    if (trainer == NULL || trainer->model == NULL || path == NULL ||
        ((dataset == NULL) == (sft_dataset == NULL)) ||
        (dataset != NULL &&
         (trainer->sft_mode != 0 || lm_dataset_get_split(dataset) != LM_DATASET_TRAIN)) ||
        (sft_dataset != NULL &&
         (trainer->sft_mode == 0 ||
          lm_sft_dataset_get_split(sft_dataset) != LM_DATASET_TRAIN))) {
        return LLM_INVALID_ARGUMENT;
    }
    const lm_model *model = trainer->model;
    uint64_t saved_payload_size = 0U;
    llm_status status = payload_size(model, &saved_payload_size);
    if (status != LLM_OK) {
        return status;
    }
    char *part_path = temporary_path(path);
    if (part_path == NULL) {
        return LLM_ALLOCATION_FAILED;
    }
    FILE *file = fopen(part_path, "wb");
    if (file == NULL) {
        free(part_path);
        return LLM_BACKEND_ERROR;
    }
    unsigned char header[LM_CHECKPOINT_V5_HEADER_SIZE] = {0};
    memcpy(header, checkpoint_magic, sizeof(checkpoint_magic));
    store_u32(header + 8U, LM_CHECKPOINT_VERSION);
    store_u32(header + 12U, LM_CHECKPOINT_V5_HEADER_SIZE);
    store_u32(header + 16U, model->config.vocabulary_size);
    store_u32(header + 20U, (uint32_t)LM_DATASET_TRAIN);
    store_u64(header + 24U, model->config.context_length);
    store_u64(header + 32U, model->config.hidden_size);
    store_u64(header + 40U, model->config.layer_count);
    store_u64(header + 48U, model->config.head_count);
    store_u64(header + 56U, model->config.feed_forward_size);
    store_u64(header + 64U, model->config.seed);
    store_u64(header + 72U, trainer->config.batch_size);
    store_u64(header + 80U, trainer->config.context_length);
    store_u64(header + 88U, trainer->config.seed);
    store_f32(header + 96U, trainer->config.learning_rate);
    store_f32(header + 100U, trainer->config.beta1);
    store_f32(header + 104U, trainer->config.beta2);
    store_f32(header + 108U, trainer->config.epsilon);
    store_f32(header + 112U, trainer->config.weight_decay);
    store_u64(header + 116U, trainer->step);
    lm_batcher_state batcher_state = {0};
    const lm_dataset_status batcher_status =
        trainer->sft_mode != 0 ? lm_sft_batcher_get_state(trainer->sft_batcher, &batcher_state)
                               : lm_batcher_get_state(trainer->batcher, &batcher_state);
    if (batcher_status != LM_DATASET_OK) {
        (void)fclose(file);
        (void)remove(part_path);
        free(part_path);
        return LLM_BACKEND_ERROR;
    }
    store_u64(header + 124U, batcher_state.random_state);
    store_u64(header + 132U, dataset != NULL ? lm_dataset_token_count(dataset)
                                             : lm_sft_dataset_supervised_token_count(sft_dataset));
    store_u64(header + 140U, dataset != NULL ? lm_dataset_document_count(dataset)
                                             : lm_sft_dataset_example_count(sft_dataset));
    store_u64(header + 148U, saved_payload_size);
    store_u64(header + 156U, trainer->config.gradient_accumulation_steps);
    store_u64(header + 164U, trainer->config.warmup_steps);
    store_u64(header + 172U, trainer->config.total_steps);
    store_f32(header + 180U, trainer->config.minimum_learning_rate);
    store_u32(header + 188U, (uint32_t)trainer->config.sampling);
    store_u64(header + 192U, batcher_state.epoch);
    store_u64(header + 200U, batcher_state.sample_index);
    store_u64(header + 208U, batcher_state.next_offset);
    store_u64(header + 216U, batcher_state.stride);
    store_f32(header + 224U, trainer->config.gradient_clip_norm);
    tokenizer_sha256_context checksum = {0};
    tokenizer_sha256_init(&checksum);
    tokenizer_sha256_update(&checksum, header, LM_CHECKPOINT_CHECKSUM_OFFSET);
    if (write_exact(file, header, sizeof(header)) == 0) {
        status = LLM_BACKEND_ERROR;
    }
    for (size_t index = 0U; status == LLM_OK && index < model->parameter_count; ++index) {
        const lm_model_parameter *parameter = &model->parameters[index];
        status = write_tensor(file, model->backend, &parameter->value, &checksum);
        if (status == LLM_OK) {
            status = write_tensor(file, model->backend, &parameter->first_moment, &checksum);
        }
        if (status == LLM_OK) {
            status = write_tensor(file, model->backend, &parameter->second_moment, &checksum);
        }
    }
    if (status == LLM_OK) {
        unsigned char digest[TOKENIZER_SHA256_DIGEST_SIZE] = {0};
        tokenizer_sha256_final(&checksum, digest);
        if (fseek(file, LM_CHECKPOINT_CHECKSUM_OFFSET, SEEK_SET) != 0 ||
            write_exact(file, digest, sizeof(digest)) == 0) {
            status = LLM_BACKEND_ERROR;
        }
    }
    if (status == LLM_OK) {
        status = sync_checkpoint_file(file);
    }
    if (fclose(file) != 0 && status == LLM_OK) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK && rename(part_path, path) != 0) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK) {
        status = sync_parent_directory(path);
    }
    if (status != LLM_OK) {
        (void)remove(part_path);
    }
    free(part_path);
    return status;
}

llm_status lm_trainer_save_checkpoint(const lm_trainer *trainer, lm_dataset *dataset,
                                      const char *path) {
    return save_checkpoint(trainer, dataset, NULL, path);
}

llm_status lm_sft_trainer_save_checkpoint(const lm_trainer *trainer, lm_sft_dataset *dataset,
                                          const char *path) {
    return save_checkpoint(trainer, NULL, dataset, path);
}

static llm_status load_checkpoint(llm_backend *backend, lm_dataset *dataset,
                                  lm_sft_dataset *sft_dataset, const char *path,
                                  const lm_trainer_resume_options *options,
                                  lm_model **out_model, lm_trainer **out_trainer) {
    if (backend == NULL || path == NULL || out_model == NULL ||
        (dataset != NULL && sft_dataset != NULL) ||
        (dataset != NULL &&
         (out_trainer == NULL || lm_dataset_get_split(dataset) != LM_DATASET_TRAIN)) ||
        (sft_dataset != NULL &&
         (out_trainer == NULL ||
          lm_sft_dataset_get_split(sft_dataset) != LM_DATASET_TRAIN)) ||
        (dataset == NULL && sft_dataset == NULL && out_trainer != NULL)) {
        return LLM_INVALID_ARGUMENT;
    }
    *out_model = NULL;
    if (out_trainer != NULL) {
        *out_trainer = NULL;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return LLM_BACKEND_ERROR;
    }
    unsigned char header[LM_CHECKPOINT_V5_HEADER_SIZE] = {0};
    llm_status status = read_exact(file, header, 16U) != 0 ? LLM_OK : LLM_INVALID_ARGUMENT;
    const uint32_t version = load_u32(header + 8U);
    const uint32_t header_size = load_u32(header + 12U);
    if (status == LLM_OK && (memcmp(header, checkpoint_magic, sizeof(checkpoint_magic)) != 0 ||
                             checkpoint_header_is_supported(version, header_size) == 0)) {
        status = LLM_INVALID_ARGUMENT;
    }
    if (status == LLM_OK && read_exact(file, header + 16U, header_size - 16U) == 0) {
        status = LLM_INVALID_ARGUMENT;
    }
    const uint32_t split = load_u32(header + 20U);
    lm_model_config model_config = {.vocabulary_size = load_u32(header + 16U),
                                    .context_length = (size_t)load_u64(header + 24U),
                                    .hidden_size = (size_t)load_u64(header + 32U),
                                    .layer_count = (size_t)load_u64(header + 40U),
                                    .head_count = (size_t)load_u64(header + 48U),
                                    .feed_forward_size = (size_t)load_u64(header + 56U),
                                    .seed = load_u64(header + 64U)};
    lm_trainer_config trainer_config = {
        .batch_size = (size_t)load_u64(header + 72U),
        .context_length = (size_t)load_u64(header + 80U),
        .seed = load_u64(header + 88U),
        .learning_rate = load_f32(header + 96U),
        .beta1 = load_f32(header + 100U),
        .beta2 = load_f32(header + 104U),
        .epsilon = load_f32(header + 108U),
        .weight_decay = load_f32(header + 112U),
        .gradient_accumulation_steps = version == 1U ? 1U : (size_t)load_u64(header + 156U),
        .warmup_steps = version == 1U ? 0U : load_u64(header + 164U),
        .total_steps = version == 1U ? 0U : load_u64(header + 172U),
        .minimum_learning_rate = version == 1U ? load_f32(header + 96U) : load_f32(header + 180U),
        .sampling =
            version < 3U ? LM_BATCHER_RANDOM_WINDOWS : (lm_batcher_sampling)load_u32(header + 188U),
        .gradient_clip_norm = version < 3U ? 0.0F : load_f32(header + 224U)};
    if (status == LLM_OK && options != NULL) {
        const size_t saved_batch_size = trainer_config.batch_size;
        const size_t saved_accumulation = trainer_config.gradient_accumulation_steps;
        const size_t resumed_batch_size =
            options->batch_size == 0U ? saved_batch_size : options->batch_size;
        const size_t resumed_accumulation = options->gradient_accumulation_steps == 0U
                                                ? saved_accumulation
                                                : options->gradient_accumulation_steps;
        if (resumed_batch_size == 0U || resumed_accumulation == 0U ||
            saved_batch_size > SIZE_MAX / saved_accumulation ||
            resumed_batch_size > SIZE_MAX / resumed_accumulation ||
            saved_batch_size * saved_accumulation != resumed_batch_size * resumed_accumulation) {
            status = LLM_INVALID_ARGUMENT;
        } else {
            trainer_config.batch_size = resumed_batch_size;
            trainer_config.gradient_accumulation_steps = resumed_accumulation;
        }
    }
    const unsigned long long step = load_u64(header + 116U);
    const uint64_t batcher_state = load_u64(header + 124U);
    const uint64_t token_count = load_u64(header + 132U);
    const uint64_t document_count = load_u64(header + 140U);
    const uint64_t saved_payload_size = load_u64(header + 148U);
    if (status == LLM_OK &&
        (memcmp(header, checkpoint_magic, sizeof(checkpoint_magic)) != 0 ||
         checkpoint_header_is_supported(version, header_size) == 0 || split != LM_DATASET_TRAIN ||
         batcher_state == 0U ||
         (dataset != NULL &&
          (model_config.vocabulary_size != lm_dataset_model_vocabulary_size(dataset) ||
           token_count != lm_dataset_token_count(dataset) ||
           document_count != lm_dataset_document_count(dataset))) ||
         (sft_dataset != NULL &&
          (model_config.vocabulary_size != lm_sft_dataset_model_vocabulary_size(sft_dataset) ||
           model_config.context_length != lm_sft_dataset_context_length(sft_dataset) ||
           token_count != lm_sft_dataset_supervised_token_count(sft_dataset) ||
           document_count != lm_sft_dataset_example_count(sft_dataset))))) {
        status = LLM_INVALID_ARGUMENT;
    }
    lm_model *model = NULL;
    lm_trainer *trainer = NULL;
    if (status == LLM_OK) {
        status = lm_model_create(backend, &model_config, &model);
    }
    uint64_t expected_payload_size = 0U;
    if (status == LLM_OK) {
        status = payload_size(model, &expected_payload_size);
    }
    if (status == LLM_OK && saved_payload_size != expected_payload_size) {
        status = LLM_INVALID_ARGUMENT;
    }
    if (status == LLM_OK && dataset != NULL) {
        status = lm_trainer_create(model, dataset, &trainer_config, &trainer);
    }
    if (status == LLM_OK && sft_dataset != NULL) {
        status = lm_sft_trainer_create(model, sft_dataset, &trainer_config, &trainer);
    }
    tokenizer_sha256_context checksum = {0};
    if (status == LLM_OK && version >= 5U) {
        tokenizer_sha256_init(&checksum);
        tokenizer_sha256_update(&checksum, header, LM_CHECKPOINT_CHECKSUM_OFFSET);
    }
    for (size_t index = 0U; status == LLM_OK && index < model->parameter_count; ++index) {
        lm_model_parameter *parameter = &model->parameters[index];
        status = read_tensor(file, backend, &parameter->value, version >= 5U ? &checksum : NULL);
        if (status == LLM_OK) {
            status = read_tensor(file, backend, &parameter->first_moment,
                                 version >= 5U ? &checksum : NULL);
        }
        if (status == LLM_OK) {
            status = read_tensor(file, backend, &parameter->second_moment,
                                 version >= 5U ? &checksum : NULL);
        }
    }
    if (status == LLM_OK && version >= 5U) {
        unsigned char digest[TOKENIZER_SHA256_DIGEST_SIZE] = {0};
        tokenizer_sha256_final(&checksum, digest);
        if (memcmp(digest, header + LM_CHECKPOINT_CHECKSUM_OFFSET, sizeof(digest)) != 0) {
            status = LLM_INVALID_ARGUMENT;
        }
    }
    if (status == LLM_OK && fgetc(file) != EOF) {
        status = LLM_INVALID_ARGUMENT;
    }
    if (fclose(file) != 0 && status == LLM_OK) {
        status = LLM_BACKEND_ERROR;
    }
    if (status == LLM_OK && trainer != NULL && trainer->sft_mode == 0) {
        if (version < 3U) {
            if (lm_batcher_set_random_state(trainer->batcher, batcher_state) != LM_DATASET_OK) {
                status = LLM_INVALID_ARGUMENT;
            }
        } else {
            const lm_batcher_state restored_state = {
                .random_state = batcher_state,
                .epoch = load_u64(header + 192U),
                .sample_index = load_u64(header + 200U),
                .next_offset = load_u64(header + 208U),
                .stride = load_u64(header + 216U),
            };
            if (lm_batcher_set_state(trainer->batcher, &restored_state) != LM_DATASET_OK) {
                status = LLM_INVALID_ARGUMENT;
            }
        }
    }
    if (status == LLM_OK && trainer != NULL && trainer->sft_mode != 0) {
        if (version < 3U) {
            status = LLM_INVALID_ARGUMENT;
        } else {
            const lm_batcher_state restored_state = {
                .random_state = batcher_state,
                .epoch = load_u64(header + 192U),
                .sample_index = load_u64(header + 200U),
                .next_offset = load_u64(header + 208U),
                .stride = load_u64(header + 216U),
            };
            if (lm_sft_batcher_set_state(trainer->sft_batcher, &restored_state) !=
                LM_DATASET_OK) {
                status = LLM_INVALID_ARGUMENT;
            }
        }
    }
    if (status == LLM_OK) {
        if (trainer != NULL) {
            trainer->step = step;
            *out_trainer = trainer;
        }
        *out_model = model;
        return LLM_OK;
    }
    lm_trainer_destroy(trainer);
    lm_model_destroy(model);
    return status;
}

llm_status lm_trainer_load_checkpoint_with_options(llm_backend *backend, lm_dataset *dataset,
                                                   const char *path,
                                                   const lm_trainer_resume_options *options,
                                                   lm_model **out_model,
                                                   lm_trainer **out_trainer) {
    return load_checkpoint(backend, dataset, NULL, path, options, out_model, out_trainer);
}

llm_status lm_trainer_load_checkpoint(llm_backend *backend, lm_dataset *dataset, const char *path,
                                      lm_model **out_model, lm_trainer **out_trainer) {
    return lm_trainer_load_checkpoint_with_options(backend, dataset, path, NULL, out_model,
                                                   out_trainer);
}

llm_status lm_sft_trainer_load_checkpoint_with_options(
    llm_backend *backend, lm_sft_dataset *dataset, const char *path,
    const lm_trainer_resume_options *options, lm_model **out_model, lm_trainer **out_trainer) {
    return load_checkpoint(backend, NULL, dataset, path, options, out_model, out_trainer);
}

llm_status lm_sft_trainer_load_checkpoint(llm_backend *backend, lm_sft_dataset *dataset,
                                          const char *path, lm_model **out_model,
                                          lm_trainer **out_trainer) {
    return lm_sft_trainer_load_checkpoint_with_options(backend, dataset, path, NULL, out_model,
                                                       out_trainer);
}
