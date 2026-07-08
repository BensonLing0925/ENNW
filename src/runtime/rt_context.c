#include "rt_context.h"
#include "graph/rt_graph.h"
#include "../ops/tk_ops.h"
#include "../../mem/arena.h"

#define DEFAULT_WS_CAPACITY (256ULL * 1024 * 1024)  /* 256 MB */

static const struct tk_rt_ctx_config DEFAULT_CONFIG = {
    .use_int8 = 0,
    .use_prof = 0,
    .use_graph_optimize = 1,
    .graph_capacity = 1024
};

struct tk_rt_ctx* tk_runtime_ctx_create_config(struct arena* root_arena, struct tk_rt_ctx_config config) {

    struct tk_rt_ctx* ctx = arena_alloc(root_arena, sizeof(struct tk_rt_ctx));

    ctx->meta_arena = arena_alloc(root_arena, sizeof(struct arena));
    arena_init(ctx->meta_arena);

    ctx->data_arena = arena_alloc(root_arena, sizeof(struct arena));
    arena_init(ctx->data_arena);

    ctx->compute_dtype = TK_F64;
    ctx->ws = tk_ws_create(root_arena, NULL);

    void* ws_data = arena_alloc(ctx->data_arena, DEFAULT_WS_CAPACITY);
    ctx->ws->arena_base  = ws_data;
    ctx->ws->capacity    = DEFAULT_WS_CAPACITY;
    ctx->ws->is_dryrun   = 0;
    ctx->ws->cur_offset  = 0;
    ctx->ws->peak_offset = 0;

    ctx->use_int8 = config.use_int8;
    ctx->use_prof = config.use_prof;
    ctx->use_graph_optimize = config.use_graph_optimize;
    ctx->graph_ready = 0;
    ctx->static_graph = NULL;  /* always initialise; set below if optimisation is on */

    if (ctx->use_graph_optimize) {
        int cap = (config.graph_capacity > 0) ? config.graph_capacity : 1024;
        ctx->static_graph = tk_rt_graph_alloc(ctx);
        tk_rt_graph_init(ctx, cap);
        ctx->ops = (struct tk_ops_vtable*)&tk_record_vtable;
    }
    if (ctx->use_prof) {
        int max_threads = 1024;
        int max_events_per_thread = 4096;
        ctx->manager = tk_prof_create(max_threads, max_events_per_thread);
    }
    /*
    if (ctx->use_int8)
        ctx->ops = (struct tk_ops_vtable*)&tk_exec_i8_vtable;
    else {
        if (!ctx->use_graph_optimize)
            ctx->ops = (struct tk_ops_vtable*)&tk_exec_vtable;
    }
    */

    return ctx;
}

struct tk_rt_ctx* tk_runtime_ctx_create(struct arena* root_arena) {
    return tk_runtime_ctx_create_config(root_arena, DEFAULT_CONFIG);
}

/* Called once after the dry run.
 * 1. Runs graph optimisations (ADD+LN fusion, etc.)
 * 2. Prints the (possibly transformed) node list.
 * 3. Switches ctx->ops to the appropriate exec vtable.
 */
int tk_rt_prepare(struct tk_rt_ctx* ctx) {
    if (ctx->rt_type != RT_DRYRUN) {
        printf("[Warning] tk_rt_prepare: already in non-dryrun mode\n");
        return 0;
    }

    if (ctx->use_graph_optimize && ctx->static_graph) {
        /* Optimise */
        tk_rt_graph_optimize(ctx);

        struct tk_rt_node* final_exec_node = NULL;
        /* Print the final graph */
        int active = 0;
        for (int i = 0; i < ctx->static_graph->node_count; i++) {
            struct tk_rt_node* node = &ctx->static_graph->nodes[i];
            if (node->skip) continue;
            final_exec_node = node;
            printf("Node [%2d]: %-22s  ws_before=%6u  In=%p  Out=%p\n",
                   active,
                   tk_rt_op_type(node->op_type),
                   (unsigned)node->ws_cursor_before,
                   (void*)node->inputs[0],
                   (void*)node->outputs[0]);
            /*
            for (int j = 0; j < node->input_count; j++) {
                printf("  In[%d]: %p (Data: %p)\n", 
                       j, (void*)node->inputs[j], (void*)node->inputs[j]->data);
            }
            printf("  Out[0]: %p (Data: %p)\n", 
                   (void*)node->outputs[0], (void*)node->outputs[0]->data);
                    ++active;
            */
            active++;
        }
        ctx->static_graph->last_node = final_exec_node;
        printf("[Info] Graph: %d active nodes (%d total, %d fused away)\n",
               active,
               ctx->static_graph->node_count,
               ctx->static_graph->node_count - active);
    }


    ctx->rt_type = RT_INFERENCE;
    if (ctx->use_int8)
        ctx->ops = (struct tk_ops_vtable*)&tk_exec_i8_vtable;
    else {
        ctx->ops = (struct tk_ops_vtable*)&tk_exec_vtable;
    }

    if (ctx->ws)
        ctx->ws->cur_offset = 0;

    ctx->graph_ready = 1;
    printf("[Info] Runtime prepared. Switched to inference mode.\n");
    return 0;
}

void tk_rt_ctx_destroy(struct tk_rt_ctx* ctx) {
    if (ctx->manager)
    	arena_destroy(ctx->manager->prof_arena);
	arena_destroy(ctx->data_arena);
    arena_destroy(ctx->meta_arena);
}

/* configuration getter */
int tk_rt_use_prof_enabled(struct tk_rt_ctx* ctx) {
    return ctx && ctx->use_prof;
}
