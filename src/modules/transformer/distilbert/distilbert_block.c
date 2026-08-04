#include "distilbert_block.h"
#include "rt_context.h"

struct tk_distilbert_block* tk_distilbert_block_create(struct tk_rt_ctx* ctx) {
    struct tk_distilbert_block* block = arena_alloc(ctx->meta_arena, sizeof(struct tk_distilbert_block));
    block->base = tk_tf_block_create(ctx);
    return block;
}

void tk_distilbert_block_alloc(struct tk_rt_ctx* ctx,
                               struct tk_distilbert_block* block) {
    tk_tf_block_alloc(ctx, block->base);
}
