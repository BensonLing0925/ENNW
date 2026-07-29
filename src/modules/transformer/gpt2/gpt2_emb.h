#ifndef TK_GPT2_EMB_H
#define TK_GPT2_EMB_H

#include "../embedding/embedding.h"
#include "../../../runtime/rt_context.h"

struct tk_gpt2_emb_config {
    int vocab_size;
    int hidden_dim;
    int max_seq_len;
};

struct tk_gpt2_emb {
    struct tk_gpt2_emb_config config;
    struct tk_embedding* word_emb;
    struct tk_embedding* pos_emb;
    struct tk_tensor* ln_gamma;
    struct tk_tensor* ln_beta;
};

struct tk_gpt2_emb* tk_gpt2_emb_create(struct tk_rt_ctx* ctx);
void tk_gpt2_emb_alloc(struct tk_rt_ctx* ctx,
                             struct tk_gpt2_emb* emb);
int tk_gpt2_emb_forward(struct tk_rt_ctx* ctx, 
                              struct tk_gpt2_emb* emb,
                              struct tk_tensor* input,
							  int pos_offset,
                              struct tk_tensor** output_ptr);

#endif
