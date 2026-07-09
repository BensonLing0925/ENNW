#include "rt_graph.h"
#include "../../error/rt_error.h"
#include "../../ops/tk_ops.h"
#include "../../ops/tensor_ops.h"
#include "../../modules/transformer/tf_block.h"

struct tk_rt_graph* tk_rt_graph_alloc(struct tk_rt_ctx* ctx) {
    return arena_alloc(ctx->meta_arena, sizeof(struct tk_rt_graph));
}

int tk_rt_graph_init(struct tk_rt_ctx* ctx, int capacity) {
    if (capacity <= 0)
        RT_FAIL(RT_EINVAL, "Node capacity must be larger than 0");
    struct tk_rt_graph* g = ctx->static_graph;
    g->nodes = arena_alloc(ctx->meta_arena, capacity * sizeof(struct tk_rt_node));
    g->node_count = 0;
    g->capacity = capacity;
    return 0;
}

int tk_rt_node_append(struct tk_rt_ctx* ctx, struct tk_rt_graph* g, struct tk_rt_node_config config, struct tk_rt_node** out_node) {
    struct tk_rt_node* node = &g->nodes[g->node_count++];
    node->op_type = config.op_type;
    node->input_count = config.input_count;
    node->output_count = config.output_count;
    node->skip = 0;
    node->ws_cursor_before = 0;

    node->inputs = arena_alloc(ctx->meta_arena, node->input_count * sizeof(struct tk_tensor*));
    node->outputs = arena_alloc(ctx->meta_arena, node->output_count * sizeof(struct tk_tensor*));

    *out_node = node;
    return 0;
}

const char* tk_rt_op_type(enum rt_op_type op_type) {
    switch (op_type) {
        case TK_OP_GEMM:           		return "TK_OP_GEMM";
        case TK_OP_ADD:            		return "TK_OP_ADD";
        case TK_OP_GELU:           		return "TK_OP_GELU";
        case TK_OP_LAYERNORM:      		return "TK_OP_LAYERNORM";
        case TK_OP_QUANTIZE:       		return "TK_OP_QUANTIZE";
        case TK_OP_ATTENTION:      		return "TK_OP_ATTENTION";
        case TK_OP_FFN:            		return "TK_OP_FFN";
        case TK_OP_FUSED_ADD_NORM: 		return "TK_OP_FUSED_ADD_NORM";
        case TK_OP_FUSED_GEMM_ADD_GELU:	return "TK_OP_FUSED_GEMM_ADD_GELU";
        default:                   		return "TK_OP_UNKNOWN";
    }
}

/* ------------------------------------------------------------------ */
/* Graph executor                                                      */
/* ------------------------------------------------------------------ */

/*
 * Execute the static graph recorded during the dry run.
 *
 * Before calling:
 *   - ctx->ws->cur_offset = 0
 *   - embedding forward already run (hidden state at workspace offset 0)
 *   - ctx->rt_type = RT_INFERENCE, ctx->ops = exec or exec_i8 vtable
 *
 * Each node restores ctx->ws->cur_offset to the value captured at recording
 * time, so ATTENTION/FFN forward functions re-allocate their scratch buffers
 * at the exact same workspace positions as during the dry run.  All other
 * workspace tensor data pointers remain valid (memory is never zeroed, only
 * the bump-pointer is moved).
 */
int tk_rt_graph_exec(struct tk_rt_ctx* ctx) {
    struct tk_rt_graph* g = ctx->static_graph;

    for (int i = 0; i < g->node_count; i++) {
        struct tk_rt_node* node = &g->nodes[i];
        if (node->skip) continue;

        /* Restore the workspace cursor to the state when this node was recorded.
         * This guarantees ATTENTION/FFN internal allocations land at the same
         * offsets as during the dry run, keeping recorded output pointers valid. */
        ctx->ws->cur_offset = node->ws_cursor_before;

        switch (node->op_type) {

            case TK_OP_ADD:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "ADD", ctx->ws->cur_offset);
                RT_CHECK(ctx->ops->add(ctx,
                    node->inputs[0], node->inputs[1], node->outputs[0]));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "ADD", ctx->ws->cur_offset);
                break;

            case TK_OP_GEMM:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "GEMM", ctx->ws->cur_offset);
                RT_CHECK(ctx->ops->gemm(ctx,
                    node->inputs[0], node->inputs[1], node->outputs[0]));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "GEMM", ctx->ws->cur_offset);
                break;

            case TK_OP_GELU:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "GELU", ctx->ws->cur_offset);
                RT_CHECK(ctx->ops->gelu(ctx,
                    node->inputs[0], node->outputs[0]));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "GELU", ctx->ws->cur_offset);
                break;

            case TK_OP_LAYERNORM:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "LAYERNORM", ctx->ws->cur_offset);
                RT_CHECK(ctx->ops->layernorm(ctx,
                    node->inputs[0], node->inputs[1], node->inputs[2],
                    node->outputs[0]));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "LAYERNORM", ctx->ws->cur_offset);
                break;

            case TK_OP_QUANTIZE:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "QUANTIZE", ctx->ws->cur_offset);            		
               	RT_CHECK(ctx->ops->quantize(ctx,
                   	node->inputs[0], node->outputs[0],
                   	node->params.quantize.calib_scale));   
				TK_PROF_SCOPE(ctx, TK_EV_OP_END, "QUANTIZE", ctx->ws->cur_offset);         		
				/*
            	if (ctx->use_int8) {
            		TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "QUANTIZE", ctx->ws->cur_offset);            		
                	RT_CHECK(ctx->ops->quantize(ctx,
                    	node->inputs[0], node->outputs[0],
                    	node->params.quantize.calib_scale));   
					TK_PROF_SCOPE(ctx, TK_EV_OP_END, "QUANTIZE", ctx->ws->cur_offset);         		
				}
				*/
                break;

            /* Fused residual-add + LayerNorm (post-norm transformer blocks).
             * inputs: [0]=residual/hidden, [1]=sublayer_out, [2]=gamma, [3]=beta
             * output: [0]=residual (updated in-place) */
            case TK_OP_FUSED_ADD_NORM:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "FUSED_ADD_NORM", ctx->ws->cur_offset);
                RT_CHECK(tk_ops_fused_add_norm(node->inputs[0], node->inputs[1],
                                               node->inputs[2], node->inputs[3],
                                               node->outputs[0]));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "FUSED_ADD_NORM", ctx->ws->cur_offset);
                /*
                RT_CHECK(ctx->ops->add(ctx,
                    node->inputs[0], node->inputs[1], node->outputs[0]));
                RT_CHECK(ctx->ops->layernorm(ctx,
                    node->outputs[0], node->inputs[2], node->inputs[3],
                    node->outputs[0]));
                */
                break;
            case TK_OP_FUSED_GEMM_ADD_GELU:
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "FUSED_GEMM_ADD_GELU", ctx->ws->cur_offset);
                RT_CHECK(tk_ops_fused_gemm_bias_gelu(node->inputs[0], node->inputs[1],
                                               		 node->inputs[2], node->outputs[0]));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "FUSED_GEMM_ADD_GELU", ctx->ws->cur_offset);
                break;

            /* High-level attention node: calling the forward function re-uses
             * the same workspace layout as the dry run (cursor restored above).
             * The local `out` pointer is discarded; downstream nodes reference
             * node->outputs[0] whose data pointer is at the same workspace address. */
            case TK_OP_ATTENTION: {
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "ATTENTION", ctx->ws->cur_offset);
                struct tk_tensor* out = NULL;
                RT_CHECK(ctx->ops->attention(ctx, node->params.attention.tf, node->inputs[0], &out));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "ATTENTION", ctx->ws->cur_offset);
                break;
            }

            case TK_OP_FFN: {
            	TK_PROF_SCOPE(ctx, TK_EV_OP_BEGIN, "FFN", ctx->ws->cur_offset);
                struct tk_tensor* out = NULL;
                RT_CHECK(ctx->ops->ffn(ctx, node->params.ffn.tf, node->inputs[0], &out));
                TK_PROF_SCOPE(ctx, TK_EV_OP_END, "FFN", ctx->ws->cur_offset);
                break;
            }

            default:
                RT_FAIL(RT_EINVAL, "Unknown op type %d in graph exec", node->op_type);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Graph optimiser                                                     */
/* ------------------------------------------------------------------ */

/*
 * ADD + LAYERNORM fusion.
 *
 * Pattern: consecutive nodes
 *   ADD    (src1, src2 → out)
 *   LAYERNORM (out, gamma, beta → out)   [typically in-place: out == src1]
 *
 * Collapsed into a single FUSED_ADD_NORM node so the graph loop has fewer
 * iterations.  The actual computation is unchanged (same two vtable calls);
 * the value comes when a dedicated fused kernel is eventually provided.
 *
 * Only fires when ADD.outputs[0] == LAYERNORM.inputs[0] (same tensor pointer).
 */
static int tk_rt_fuse_add_norm(struct tk_rt_ctx* ctx, struct tk_rt_graph* g) {
    int fused = 0;

    for (int i = 0; i + 1 < g->node_count; i++) {
        struct tk_rt_node* a = &g->nodes[i];
        struct tk_rt_node* n = &g->nodes[i + 1];

        if (a->skip || n->skip)            continue;
        if (a->op_type != TK_OP_ADD)       continue;
        if (n->op_type != TK_OP_LAYERNORM) continue;
        if (a->outputs[0] != n->inputs[0]) continue;  /* must be the same tensor */

        /* Re-use ADD node as FUSED_ADD_NORM with 4 inputs */
        a->op_type     = TK_OP_FUSED_ADD_NORM;
        a->input_count = 4;

        struct tk_tensor** merged = arena_alloc(ctx->meta_arena, 4 * sizeof(struct tk_tensor*));
        a->ws_cursor_before = n->ws_cursor_before;
        merged[0] = a->inputs[0];  /* residual / running hidden */
        merged[1] = a->inputs[1];  /* sublayer output           */
        merged[2] = n->inputs[1];  /* gamma                     */
        merged[3] = n->inputs[2];  /* beta                      */
        a->inputs = merged;

        /* Output is whatever the LayerNorm was writing to */
        a->outputs[0] = n->outputs[0];

        n->skip = 1;
        ++fused;
    }

    if (fused) {
        printf("[Optimize] Fused %d ADD+LAYERNORM pair(s) into FUSED_ADD_NORM\n", fused);
    }
    return 0;
}

/*
 * GEMM + ADD + GELU fusion.
 *
 * Pattern: consecutive nodes
 *	 GEMM (src1, src2, -> out)
 *	 ADD  (out, bias -> out) 
 *	 GELU (out -> out)
 *
 * Only fires when ADD.outputs[0] == LAYERNORM.inputs[0] (same tensor pointer).
 */
static int tk_rt_fuse_gemm_bias_gelu(struct tk_rt_ctx* ctx, struct tk_rt_graph* g) {
    int fused = 0;

    for (int i = 0; i + 1 < g->node_count; i++) {
		struct tk_rt_node* gemm = &g->nodes[i];
        struct tk_rt_node* a = &g->nodes[i + 1];
        struct tk_rt_node* gelu = &g->nodes[i + 2];

        if (gemm->skip || a->skip || gelu->skip)	continue;
        if (gemm->op_type != TK_OP_GEMM)	continue;
        if (a->op_type != TK_OP_ADD)	 continue;
        if (gelu->op_type != TK_OP_GELU)	continue;
        if ((gemm->outputs[0] != a->inputs[0]) || 
			(a->outputs[0] != gelu->inputs[0]))	continue;  /* must be the same tensor */

        /* Re-use GEMM node as FUSED_GEMM_ADD_GELU with 4 inputs */
        gemm->op_type     = TK_OP_FUSED_GEMM_ADD_GELU;
        gemm->input_count = 3;

        struct tk_tensor** merged = arena_alloc(ctx->meta_arena, 3 * sizeof(struct tk_tensor*));
        gemm->ws_cursor_before = gelu->ws_cursor_before;
        a->ws_cursor_before = gelu->ws_cursor_before;
        merged[0] = gemm->inputs[0];  /* residual / running hidden */
        merged[1] = gemm->inputs[1];  /* sublayer output           */
        merged[2] = a->inputs[1];  /* gamma                     */
        gemm->inputs = merged;

        /* Output is whatever the LayerNorm was writing to */
        gemm->outputs[0] = gelu->outputs[0];

        a->skip = 1;
        gelu->skip = 1;
		++fused;
    }

    if (fused) {
        printf("[Optimize] Fused %d GEMM+ADD+GELU triplet(s) into FUSED_GEMM_ADD_GELU\n", fused);
    }
    return 0;
}

int tk_rt_graph_optimize(struct tk_rt_ctx* ctx) {
	tk_rt_fuse_add_norm(ctx, ctx->static_graph);
	return tk_rt_fuse_gemm_bias_gelu(ctx, ctx->static_graph);
}
