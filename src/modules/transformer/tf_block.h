#ifndef BLOCK_H
#define BLOCK_H

#include "../../ops/tensor.h"

/* Forward-declare runtime types to avoid heavy header chain in this header */
struct tk_rt_ctx;

struct TransformerBlock {

    struct {
        int hidden_dim;
        int head_dim;
        int n_heads;
        int inter_dim;
        int seq_length;
    } config;

    /* QKV projection weights: [hidden_dim, hidden_dim] each */
    struct tk_tensor* q_weights;
    struct tk_tensor* k_weights;
    struct tk_tensor* v_weights;
    struct tk_tensor* q_bias;
    struct tk_tensor* k_bias;
    struct tk_tensor* v_bias;

    /* FFN weights: up [hidden_dim, inter_dim], down [inter_dim, hidden_dim] */
    struct tk_tensor* ffn_up_weights;
    struct tk_tensor* ffn_down_weights;

    /* LayerNorm params: [hidden_dim] each */
    struct tk_tensor* ln1_gamma;
    struct tk_tensor* ln1_beta;
    struct tk_tensor* ln2_gamma;
    struct tk_tensor* ln2_beta;
};

/* Allocate a TransformerBlock struct in ctx->meta_arena */
struct TransformerBlock* tf_block_create(struct tk_rt_ctx* ctx);

/*
 * Allocate and randomly initialise all weight tensors.
 *   seq_length : number of tokens in the sequence
 *   hidden_dim : embedding / feature dimension per token
 *   n_heads    : number of attention heads (hidden_dim must be divisible by n_heads)
 */
void tf_block_alloc(struct tk_rt_ctx* ctx,
                    struct TransformerBlock* tf_block,
                    int seq_length,
                    int hidden_dim,
                    int n_heads);

/*
 * Forward pass.  Modifies `input` in-place via residual connections.
 *   input shape: [seq_length, hidden_dim]
 */
int tf_block_forward(struct tk_rt_ctx* ctx,
                     struct TransformerBlock* tf_block,
                     struct tk_tensor* input);

#endif
