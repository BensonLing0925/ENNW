#ifndef BLOCK_H
#define BLOCK_H

#include "../../ops/tensor.h"
#include "../../ops/tensor_ops.h"
#include "../runtime/rt_context.h"
#include "../runtime/workspaces/rt_workspaces.h"
#include "../fc/fc.h"

struct TransformerBlock {

    struct {
        int hidden_dim;
        int head_dim;
        int n_heads;
        int inter_dim;
    } config;

    struct tk_tensor* q_weights, *k_weights, *v_weights;
    struct tk_tensor* q_bias, *k_bias, *v_bias;

    struct tk_tensor* ffn_up_weights;
    struct tk_tensor* ffn_down_weights;

    // LayerNorm 1 (Before Attention)
    struct tk_tensor* ln1_gamma; // [hidden_dim]
    struct tk_tensor* ln1_beta;  // [hidden_dim]

    // LayerNorm2 (Before FFN)
    struct tk_tensor* ln2_gamma; // [hidden_dim]
    struct tk_tensor* ln2_beta;  // [hidden_dim]
};



#endif
