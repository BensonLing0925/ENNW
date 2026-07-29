#ifndef TK_TF_CONFIG_H
#define TK_TF_CONFIG_H

#include "../../ops/tensor.h"   // enum tk_dtype
#include "../../ops/tensor_ops_config.h"

struct tk_tf_block_config {

	int max_seq_len;
    int seq_length;
    int hidden_dim;
    int n_heads;
    int inter_dim;      // 0 = automatically set to 4 * hidden_dim
    int use_qkv_bias;
    int use_o_proj;
    int use_o_proj_bias;
    int use_ffn_bias;
    int pre_norm;        // 1 = pre-norm (GPT-style), 0 = post-norm (BERT/DistilBERT-style)
    enum tk_dtype dtype;
    int use_causal;

	struct tk_ops_config ops_config;
};

#endif
