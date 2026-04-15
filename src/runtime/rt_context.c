#include "rt_context.h"

#define DEFAULT_WS_CAPACITY (256ULL * 1024 * 1024)  /* 256 MB */

struct tk_rt_ctx* tk_runtime_ctx_create(struct arena* root_arena) {

    struct tk_rt_ctx* ctx = arena_alloc(root_arena, sizeof(struct tk_rt_ctx));

    ctx->meta_arena = arena_alloc(root_arena, sizeof(struct arena));
    arena_init(ctx->meta_arena);

    ctx->data_arena = arena_alloc(root_arena, sizeof(struct arena));
    arena_init(ctx->data_arena);

    ctx->ws = tk_ws_create(root_arena, NULL);

    /* Allocate a real workspace buffer so both dry-run and real passes work */
    void* ws_data = arena_alloc(ctx->data_arena, DEFAULT_WS_CAPACITY);
    ctx->ws->arena_base  = ws_data;
    ctx->ws->capacity    = DEFAULT_WS_CAPACITY;
    ctx->ws->is_dryrun   = 0;
    ctx->ws->cur_offset  = 0;
    ctx->ws->peak_offset = 0;

    return ctx;
}
