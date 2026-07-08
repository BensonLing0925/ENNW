#ifndef RT_GRAPH_H
#define RT_GRAPH_H

#include "rt_params.h"
#include "../../ops/tensor.h"
#include "../rt_context.h"

struct tk_rt_ctx;

enum rt_op_type {
    TK_OP_GEMM,
    TK_OP_ADD,
    TK_OP_GELU,
    TK_OP_LAYERNORM,
    TK_OP_QUANTIZE,
    TK_OP_ATTENTION,
    TK_OP_FFN,
    TK_OP_FUSED_ADD_NORM,  /* fused residual-add + layer-norm (post-norm only) */
};

struct tk_rt_node {
    enum rt_op_type op_type;
    int input_count;
    int output_count;
    int skip;
    size_t ws_cursor_before;   /* workspace cur_offset when this node was recorded */
    struct tk_tensor** inputs;
    struct tk_tensor** outputs;

    union {
        struct tk_rt_gemm_params        gemm;
        struct tk_rt_add_params         add;
        struct tk_rt_gelu_params        gelu;
        struct tk_rt_layernorm_params   layernorm;
        struct tk_rt_attention_params   attention;
        struct tk_rt_ffn_params         ffn;
        struct tk_rt_quantize_params    quantize;
    } params;
};

struct tk_rt_node_config {
    int input_count;
    int output_count;
    enum rt_op_type op_type;
};

struct tk_rt_graph {
    struct tk_rt_node* nodes;
    struct tk_rt_node* last_node;
    int node_count;
    int capacity;
};

struct tk_rt_graph* tk_rt_graph_alloc(struct tk_rt_ctx* ctx);
int tk_rt_graph_init(struct tk_rt_ctx* ctx, int capacity);
int tk_rt_node_append(struct tk_rt_ctx* ctx, struct tk_rt_graph* g, struct tk_rt_node_config config, struct tk_rt_node** out_node);
const char* tk_rt_op_type(enum rt_op_type op_type);

/* Execute the static graph. Caller must reset ctx->ws->cur_offset = 0 and
 * run embedding forward before calling this. */
int tk_rt_graph_exec(struct tk_rt_ctx* ctx);

/* Apply graph-level optimisations (currently: ADD+LAYERNORM fusion).
 * Called once by tk_rt_prepare after the dry run. */
int tk_rt_graph_optimize(struct tk_rt_ctx* ctx);

#endif
