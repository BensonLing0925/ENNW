#ifndef TK_DISTILBERT_BLK_H
#define TK_DISTILBERT_BLK_H

#include "../tf_block.h"

struct tk_distilbert_block {
    struct TransformerBlock* base;
};

struct tk_distilbert_block* tk_distilbert_block_create(struct tk_rt_ctx* ctx);
void tk_distilbert_block_alloc(struct tk_rt_ctx* ctx,
                               struct tk_distilbert_block* block);

#endif
