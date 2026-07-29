#include "tk_ops.h"
#include "tensor_ops_config.h"
#include "../runtime/graph/rt_graph.h"
#include "../runtime/rt_context.h"
#include "../modules/transformer/tf_block.h"

int record_add(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest) {
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");
    struct tk_rt_node_config config = {.op_type = TK_OP_ADD,
                                       .input_count = 2,
                                       .output_count = 1};
    struct tk_rt_node* out_node = NULL;
    tk_rt_node_append(ctx, g, config, &out_node);
    out_node->ws_cursor_before = ctx->ws->cur_offset;
    out_node->inputs[0] = src1;
    out_node->inputs[1] = src2;
    out_node->outputs[0] = dest;
    return 0;
}

int record_gemm(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest) {
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");
    struct tk_rt_node_config config = {.op_type = TK_OP_GEMM,
                                       .input_count = 2,
                                       .output_count = 1};
    struct tk_rt_node* out_node = NULL;
    tk_rt_node_append(ctx, g, config, &out_node);
    out_node->ws_cursor_before = ctx->ws->cur_offset;
    out_node->inputs[0] = src1;
    out_node->inputs[1] = src2;
    out_node->outputs[0] = dest;
    return 0;
}

int record_layernorm(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest) {
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");
    struct tk_rt_node_config config = {.op_type = TK_OP_LAYERNORM,
                                       .input_count = 3,
                                       .output_count = 1};
    struct tk_rt_node* out_node = NULL;
    tk_rt_node_append(ctx, g, config, &out_node);
    out_node->ws_cursor_before = ctx->ws->cur_offset;
    out_node->inputs[0] = src;
    out_node->inputs[1] = gamma;
    out_node->inputs[2] = beta;
    out_node->outputs[0] = dest;
    return 0;
}

int record_gelu(struct tk_rt_ctx* ctx, struct tk_ops_config* oc, 
				struct tk_rt_node* node, struct tk_tensor* src, struct tk_tensor* dest) {
	(void)node;
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");
    struct tk_rt_node_config config = {.op_type = TK_OP_GELU,
                                       .input_count = 1,
                                       .output_count = 1};
    struct tk_rt_node* out_node = NULL;
    tk_rt_node_append(ctx, g, config, &out_node);

	/* flatten oc */
	out_node->params.single.gelu.gelu_variant = oc->gelu_variant;
	out_node->params.single.gelu.gelu_fn = oc->gelu_fn;
	out_node->params.single.gelu.gelu_fn_raw = oc->gelu_fn_raw;

    out_node->ws_cursor_before = ctx->ws->cur_offset;
    out_node->inputs[0] = src;
    out_node->outputs[0] = dest;
    return 0;
}

int record_quantize(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* dest, float calib) {
    struct tk_rt_graph* g = ctx->static_graph;


	if (g->capacity <= g->node_count)
		RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");
	struct tk_rt_node_config config = {.op_type = TK_OP_QUANTIZE,
									   .input_count = 1,
									   .output_count = 1};
	struct tk_rt_node* out_node = NULL;
	tk_rt_node_append(ctx, g, config, &out_node);
	out_node->ws_cursor_before = ctx->ws->cur_offset;
	out_node->inputs[0] = src;
	out_node->outputs[0] = dest;
	out_node->params.single.quantize.calib_scale = calib;

	/*
    if (dest->dtype == TK_I8) {
        if (g->capacity <= g->node_count)
            RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");
        struct tk_rt_node_config config = {.op_type = TK_OP_QUANTIZE,
                                           .input_count = 1,
                                           .output_count = 1};
        struct tk_rt_node* out_node = NULL;
        tk_rt_node_append(ctx, g, config, &out_node);
        out_node->ws_cursor_before = ctx->ws->cur_offset;
        out_node->inputs[0] = src;
        out_node->outputs[0] = dest;
        out_node->params.quantize.calib_scale = calib;
    }
	*/
    return 0;
}

int record_attention(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
                     struct tk_tensor* input, struct tk_tensor** out) {
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");

    /* Capture cursor BEFORE the forward call so graph exec restores it here,
     * causing internal workspace allocations to land at the same offsets. */
    size_t cursor_before = ctx->ws->cur_offset;

    /* Run in dryrun mode: sizes the workspace, skips computation */
    RT_CHECK(tk_tf_attention_forward(ctx, tf, input, out));

    struct tk_rt_node_config config = { .op_type = TK_OP_ATTENTION,
                                        .input_count = 1, .output_count = 1 };
    struct tk_rt_node* node = NULL;
    tk_rt_node_append(ctx, g, config, &node);
    node->ws_cursor_before    = cursor_before;
    node->inputs[0]           = input;
    node->outputs[0]          = *out;
    node->params.single.attention.tf = tf;
    return 0;
}

/*
int record_attention_decode(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
                     		struct tk_tensor* input, struct tk_tensor** out) {
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");

    size_t cursor_before = ctx->ws->cur_offset;

    RT_CHECK(tk_tf_attention_forward(ctx, tf, input, out));

    struct tk_rt_node_config config = { .op_type = TK_OP_ATTENTION,
                                        .input_count = 1, .output_count = 1 };
    struct tk_rt_node* node = NULL;
    tk_rt_node_append(ctx, g, config, &node);
    node->ws_cursor_before    = cursor_before;
    node->inputs[0]           = input;
    node->outputs[0]          = *out;
    node->params.single.attention.tf = tf;
    return 0;
}
*/

int record_ffn(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
               struct tk_tensor* input, struct tk_tensor** out) {
    struct tk_rt_graph* g = ctx->static_graph;
    if (g->capacity <= g->node_count)
        RT_FAIL(RT_EINVAL, "Number of graph node exceeded\n");

    size_t cursor_before = ctx->ws->cur_offset;

    /* Run in dryrun mode: sizes the workspace, skips computation */
    RT_CHECK(tk_tf_ffn_forward(ctx, tf, input, out));

    struct tk_rt_node_config config = { .op_type = TK_OP_FFN,
                                        .input_count = 1, .output_count = 1 };
    struct tk_rt_node* node = NULL;
    tk_rt_node_append(ctx, g, config, &node);
    node->ws_cursor_before = cursor_before;
    node->inputs[0]        = input;
    node->outputs[0]       = *out;
    node->params.single.ffn.tf    = tf;
    return 0;
}

const struct tk_ops_vtable tk_record_vtable = {
    .add       = record_add,
    .gemm      = record_gemm,
    .layernorm = record_layernorm,
    .gelu      = record_gelu,
    .quantize  = record_quantize,
    .attention = record_attention,
    .ffn       = record_ffn
};
