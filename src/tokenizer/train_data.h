#ifndef TOKENIZER_TRAIN_DATA_H
#define TOKENIZER_TRAIN_DATA_H

#include "tokenizer_internal.h"

typedef struct tokenizer_training_data tokenizer_training_data;

tokenizer_status tokenizer_training_data_create(tokenizer_training_data **out_data);
void tokenizer_training_data_destroy(tokenizer_training_data *data);

tokenizer_status tokenizer_training_collect_pretoken(const unsigned char *bytes, size_t length,
                                                     void *context);
tokenizer_status tokenizer_training_prepare(tokenizer_training_data *data,
                                            tokenizer_train_progress_callback progress_callback,
                                            void *progress_context);
tokenizer_status tokenizer_training_merge_next(tokenizer_training_data *data, tokenizer *tokenizer,
                                               int *out_did_merge);

#endif
