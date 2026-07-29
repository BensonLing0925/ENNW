#ifndef RT_CONTEXT_H
#define RT_CONTEXT_H

#define RESERVED 2048

#include "../../mem/arena.h"
#include "../error/rt_error.h"
#include "../profiler/tk_profiler.h"
#include "workspaces/rt_workspaces.h"

enum rt_type {
    RT_TRAIN,
    RT_INFERENCE,
    RT_DRYRUN
};

struct tk_rt_ctx {
    
    /* rt_type is used for static graph optimization */
    enum rt_type rt_type;
    struct tk_rt_graph* static_graph;
    int use_graph_optimize;
    struct tk_ops_vtable* ops;

    enum tk_dtype compute_dtype;

	/* arena */
    struct arena* prof_arena;
    struct arena* meta_arena;
    struct arena* data_arena;

	/* should kv_arena belongs to ctx is a question worth considering */
	struct arena* kv_arena;

	/* kv cache united current index  */
	int kv_cur_len;

    struct tk_workspace* ws;
    struct Model* model;

    // event profiler and visualizer
    struct tk_prof_manager* manager;
    int use_prof;

    /* Set to 1 before tk_distilbert_alloc to enable int8 weight quantization.
     * Weight matrices are allocated as TK_I8; activations are quantized
     * dynamically per-inference.  Biases and LayerNorm params stay float. */
    int use_int8;

    /* Set to 1 by tk_rt_prepare after dry run + optimisation.
     * Enables the graph-exec path in tk_distilbert_forward. */
    int graph_ready;

};

struct tk_rt_ctx_config {
    int use_int8;
    int use_prof;
    int use_graph_optimize;
    int graph_capacity;
};

struct tk_rt_ctx* tk_runtime_ctx_create(struct arena* root_arena);
struct tk_rt_ctx* tk_runtime_ctx_create_config(struct arena* root_arena, struct tk_rt_ctx_config config);
int tk_rt_prepare(struct tk_rt_ctx* ctx);
void tk_rt_ctx_destroy(struct tk_rt_ctx* ctx);
void tk_rt_ctx_set_mode(struct tk_rt_ctx* ctx, enum rt_type type);
// configuration getter
int tk_rt_is_prof_enabled(struct tk_rt_ctx* ctx);

#endif
