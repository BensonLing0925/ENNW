#ifndef BLOCK_H
#define BLOCK_H

#include "../../ops/tensor.h"
#include "tf_config.h"

/* Forward-declare runtime types to avoid heavy header chain in this header */
struct tk_rt_ctx;

struct TransformerBlock {

    struct tk_tf_block_config config;

    /* QKV projection weights: [hidden_dim, hidden_dim] each */
    struct tk_tensor* q_weights;
    struct tk_tensor* k_weights;
    struct tk_tensor* v_weights;
    struct tk_tensor* q_bias;
    struct tk_tensor* k_bias;
    struct tk_tensor* v_bias;

    /* out projection */
    struct tk_tensor* o_proj_weights;
    struct tk_tensor* o_proj_bias;

    /* FFN weights: up [hidden_dim, inter_dim], down [inter_dim, hidden_dim] */
    struct tk_tensor* ffn_up_weights;
    struct tk_tensor* ffn_down_weights;
    struct tk_tensor* ffn_up_bias;
    struct tk_tensor* ffn_down_bias;

    /* LayerNorm params: [hidden_dim] each */
    struct tk_tensor* ln1_gamma;
    struct tk_tensor* ln1_beta;
    struct tk_tensor* ln2_gamma;
    struct tk_tensor* ln2_beta;

    /* Per-tensor weight scales for int8 quantization (dtype TK_I8 weights).
     * w_scale = max(|W_f32|) / 127 */
    float q_scale;
    float k_scale;
    float v_scale;
    float o_proj_scale;
    float ffn_up_scale;
    float ffn_down_scale;

    /* Static activation scales from calibration (0.0 = use dynamic per-batch).
     * Q, K, V share the same input, so one scale covers all three.
     * Populated by the exporter when --calib is used. */
    float attn_in_act_scale;      /* input to Q/K/V projections */
    float o_proj_in_act_scale;    /* input to O_proj            */
    float ffn_up_in_act_scale;    /* input to FFN up            */
    float ffn_down_in_act_scale;  /* input to FFN down (post-GELU) */
};


/* Allocate a TransformerBlock struct in ctx->meta_arena */
struct TransformerBlock* tk_tf_block_create(struct tk_rt_ctx* ctx);

/*
 * Allocate and randomly initialise all weight tensors.
 *   seq_length : number of tokens in the sequence
 *   hidden_dim : embedding / feature dimension per token
 *   n_heads    : number of attention heads (hidden_dim must be divisible by n_heads)
 */
void tk_tf_block_alloc(struct tk_rt_ctx* ctx,
                    struct TransformerBlock* tf);

int tk_tf_ffn_forward(struct tk_rt_ctx* ctx,
                      struct TransformerBlock* tf,
                      struct tk_tensor* input,
                      struct tk_tensor** ffn_out_ptr);

int tk_tf_attention_forward(struct tk_rt_ctx* ctx,
                            struct TransformerBlock* tf,
                            struct tk_tensor* input,
                            struct tk_tensor** attn_out_ptr);

/*
 * Forward pass.  Modifies `input` in-place via residual connections.
 *   input shape: [seq_length, hidden_dim]
 */
int tk_tf_block_forward(struct tk_rt_ctx* ctx,
                     struct TransformerBlock* tf_block,
                     struct tk_tensor* input);

#endif
