#ifndef TK_GPT2_BLK_H
#define TK_GPT2_BLK_H

#include "../tf_block.h"

struct tk_gpt2_block {
	struct TransformerBlock* base;
};

struct tk_gpt2_block* tk_gpt2_block_create(struct tk_rt_ctx* ctx);
void tk_gpt2_block_alloc(struct tk_rt_ctx* ctx,
                               struct tk_gpt2_block* block);

#endif
