#ifndef TK_GPT2_H
#define TK_GPT2_H

#include "gpt2_emb.h"
#include "gpt2_block.h"
#include "../../../ops/tensor_ops.h"
#include "../../../ops/tk_ops.h"
#include "../../../runtime/rt_context.h"

enum tk_gpt2_forward_mode {
    TK_GPT2_PREFILL,
    TK_GPT2_DECODE,
};

struct tk_gpt2 {
    struct tk_gpt2_emb* emb;
    struct tk_gpt2_block** blocks;  // array of pointers
	struct tk_tensor* final_ln_gamma;
	struct tk_tensor* final_ln_beta;
	struct tk_tensor* lm_head_weight;
	int lm_head_ready;
    int num_layers;
};

struct tk_gpt2_config {
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
	
	int use_pre_norm;
	int use_causal;
};

/* Create and configure model structs (no weight allocation) */
int tk_gpt2_config(struct tk_rt_ctx* ctx,
                         struct tk_gpt2_config config,
                         struct tk_gpt2** gpt2_out);

/* Allocate and randomly initialise all weight tensors */
int tk_gpt2_alloc(struct tk_rt_ctx* ctx,
                        struct tk_gpt2_config config,
                        struct tk_gpt2* gpt2);

/* Forward pass: input_ids [seq_len] (TK_I32) -> hidden [seq_len, hidden_dim] */
int tk_gpt2_forward(struct tk_rt_ctx* ctx,
                          struct tk_gpt2* gpt2,
                          struct tk_tensor* input_ids,
                          struct tk_tensor** output_ptr,
					 	  enum tk_gpt2_forward_mode mode);

int tk_gpt2_generate(struct tk_rt_ctx* ctx, struct tk_gpt2* gpt2,
                     struct tk_tensor* prompt_ids, int max_new_tokens, int eos_token_id,
                     int32_t** out_tokens, int* out_count);

#endif
