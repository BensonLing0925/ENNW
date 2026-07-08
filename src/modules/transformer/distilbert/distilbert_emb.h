#ifndef TK_DISTILBERT_EMB_H
#define TK_DISTILBERT_EMB_H

#include "../embedding/embedding.h"
#include "rt_context.h"

struct tk_distilbert_emb_config {
    int vocab_size;
    int hidden_dim;
    int max_seq_len;
};

struct tk_distilbert_emb {
    struct tk_distilbert_emb_config config;
    struct tk_embedding* word_emb;
    struct tk_embedding* pos_emb;
    struct tk_tensor* ln_gamma;
    struct tk_tensor* ln_beta;
};

struct tk_distilbert_emb* tk_distilbert_emb_create(struct tk_rt_ctx* ctx);
void tk_distilbert_emb_alloc(struct tk_rt_ctx* ctx,
                             struct tk_distilbert_emb* emb);
int tk_distilbert_emb_forward(struct tk_rt_ctx* ctx, 
                              struct tk_distilbert_emb* emb,
                              struct tk_tensor* input,
                              struct tk_tensor** output_ptr);

#endif
