#ifndef TK_GPT2_EMB_H
#define TK_GPT2_EMB_H

#include "embedding.h"
#include "rt_context.h"

struct tk_emb_block* tk_gpt2_emb_create(struct tk_rt_ctx* ctx);
void tk_gpt2_emb_alloc(struct tk_rt_ctx* ctx,
                             struct tk_emb_block* emb);

/*
int tk_gpt2_emb_forward(struct tk_rt_ctx* ctx, 
                              struct tk_gpt2_emb* emb,
                              struct tk_tensor* input,
							  int pos_offset,
                              struct tk_tensor** output_ptr);
*/

#endif
