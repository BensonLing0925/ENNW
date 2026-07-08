#ifndef TK_DISTILBERT_H
#define TK_DISTILBERT_H

#include "distilbert_emb.h"
#include "distilbert_block.h"
#include "../../../ops/tensor_ops.h"
#include "../../../runtime/rt_context.h"

struct tk_distilbert {
    struct tk_distilbert_emb* emb;
    struct tk_distilbert_block** blocks;  // array of pointers
    int num_layers;
};

struct tk_distilbert_config {
    int vocab_size;
    int max_seq_len;
    int num_layers;
    int seq_length;
    int hidden_dim;
    int n_heads;
    int inter_dim;      // 0 = automatically set to 4 * hidden_dim
    int use_qkv_bias;
    int use_o_proj;
    int use_o_proj_bias;
    int use_ffn_bias;
};

/* Create and configure model structs (no weight allocation) */
int tk_distilbert_config(struct tk_rt_ctx* ctx,
                         struct tk_distilbert_config config,
                         struct tk_distilbert** distilbert_out);

/* Allocate and randomly initialise all weight tensors */
int tk_distilbert_alloc(struct tk_rt_ctx* ctx,
                        struct tk_distilbert_config config,
                        struct tk_distilbert* distilbert);

/* Forward pass: input_ids [seq_len] (TK_I32) -> hidden [seq_len, hidden_dim] */
int tk_distilbert_forward(struct tk_rt_ctx* ctx,
                          struct tk_distilbert* distilbert,
                          struct tk_tensor* input_ids,
                          struct tk_tensor** output_ptr);

#endif
