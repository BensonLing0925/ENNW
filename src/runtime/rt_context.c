#include "rt_context.h"

struct tk_rt_ctx* tk_runtime_ctx_create(struct arena* root_arena) {
    
    struct tk_rt_ctx* ctx = arena_alloc(root_arena, sizeof(struct tk_rt_ctx));

    ctx->meta_arena = arena_alloc(root_arena, sizeof(struct arena));
    arena_init(ctx->meta_arena);

    ctx->data_arena = arena_alloc(root_arena, sizeof(struct arena));
    arena_init(ctx->data_arena);
    // void* big_block_data = arena_alloc(ctx->data_arena, max_cap + RESERVED);
    ctx->ws = tk_ws_create(root_arena, NULL);
    return ctx;
}

