#include "gpt2_block.h"
#include "../../../runtime/rt_context.h"

struct tk_gpt2_block* tk_gpt2_block_create(struct tk_rt_ctx* ctx) {
    struct tk_gpt2_block* block = arena_alloc(ctx->meta_arena, sizeof(struct tk_gpt2_block));
    block->base = tk_tf_block_create(ctx);
    return block;
}

void tk_gpt2_block_alloc(struct tk_rt_ctx* ctx,
                               struct tk_gpt2_block* block) {
    tk_tf_block_alloc(ctx, block->base);
}
