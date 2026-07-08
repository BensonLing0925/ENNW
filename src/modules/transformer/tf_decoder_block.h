#ifndef TK_DBLOCK_H
#define TK_DBLOCK_H

#include "tf_block.h"
#include "../../ops/tensor.h"

struct tk_tf_decoder_block {

    struct TransformerBlock* base;

    struct tk_tf_decoder_config config;

    /* cross-attention weights */
    struct tk_tensor* xattn_q_weights;
    struct tk_tensor* xattn_k_weights;
    struct tk_tensor* xattn_v_weights;
    struct tk_tensor* xattn_o_proj_weights;
    struct tk_tensor* xattn_o_proj_bias;
    struct tk_tensor* ln3_gamma;
    struct tk_tensor* ln3_beta;

    float xattn_q_scale;
    float xattn_k_scale;
    float xattn_v_scale;
    float xattn_in_act_scale;
};

#endif
