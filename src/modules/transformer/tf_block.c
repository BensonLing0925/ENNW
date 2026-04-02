#include "tf_block.h"
#include "../fc/fc.h"
#include "../runtime/rt_context.h"
#include "../runtime/workspaces/rt_workspaces.h"
#include "../../ops/tensor.h"
#include "../../ops/tensor_ops.h"

static struct tk_tensor* attention_forward(struct tk_rt_ctx* ctx,
                                           struct TransformerBlock* tf_block,
                                           struct tk_tensor* input) {


    /*  calculate size from TransformerBlock's config
     *  QKV weights buffer size: [seq_length, hidden_dim]
     */

    int seq_length = tf_block->config->seq_length;
    int hidden_dim = tf_block->config->hidden_dim;
    int n_heads = tf_block->config->n_heads;
    int head_dim = hidden_dim / n_heads;
    double scale_factor = 1.0 / sqrt((double)head_dim);

    // temporary and intermediate data (pre-allocate)
    struct tk_tensor* q_buf = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){seq_length, hidden_dim}, 2)
    struct tk_tensor* k_buf = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){seq_length, hidden_dim}, 2)
    struct tk_tensor* v_buf = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){seq_length, hidden_dim}, 2)
    struct tk_tensor* score_buf = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){n_heads, seq_length, seq_length}, 3);
    struct tk_tensor* atten_out = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){seq_length, hidden_dim}, 2);
    struct tk_tensor* kT = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){hidden_dim, seq_length}, 2);
    struct tk_tensor* kT_split = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){n_heads, head_dim, seq_length}, 3);

    // view change tensors
    struct tk_tensor* q_split = tk_tensor_view(ctx->meta_arena, q_buf, (int[]){seq_length, n_heads, head_dim}, 3);
    struct tk_tensor* k_split = tk_tensor_view(ctx->meta_arena, k_buf, (int[]){seq_length, n_heads, head_dim}, 3);
    struct tk_tensor* v_split = tk_tensor_view(ctx->meta_arena, v_buf, (int[]){seq_length, n_heads, head_dim}, 3);
    struct tk_tensor* x_split = tk_tensor_view(ctx->meta_arena, x, (int[]){seq_length, n_heads, head_dim}, 3);

    // point to contiguous buffer first, we will change to point to the real data later
    // the point is to create these meta data before actual run
    struct tk_tensor* qi = tk_tensor_view(ctx->meta_arena, q_buf, (int[]){seq_length, head_dim}, 2);
    struct tk_tensor* ki = tk_tensor_view(ctx->meta_arena, k_buf, (int[]){head_dim, seq_length}, 2);
    struct tk_tensor* vi = tk_tensor_view(ctx->meta_arena, v_buf, (int[]){seq_length, head_dim}, 2);
    struct tk_tensor* si = tk_tensor_view(ctx->meta_arena, score_buf, (int[]){seq_length, seq_length}, 2);
    struct tk_tensor* outi = tk_tensor_view(ctx->meta_arena, atten_buf, (int[]){seq_length, head_dim}, 2);

    // qkv permute to move n_head to first dimension (pre-allocated)
    struct tk_tensor* q_permute = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){n_heads, seq_length, head_dim}, 3);
    struct tk_tensor* k_permute = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){n_heads, seq_length, head_dim}, 3);
    struct tk_tensor* v_permute = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){n_heads, seq_length, head_dim}, 3);
    
    // pre-loaded data
    struct tk_tensor* q_weights = tf_block->q_weights;
    struct tk_tensor* k_weights = tf_block->k_weights;
    struct tk_tensor* v_weights = tf_block->v_weights;

    int err = 0;
    // 1. projections
    // x = [seq_length, hidden_dim]
    // weights = [hidden_dim, hidden_dim]
    // buf = [seq_length, hidden_dim]
    err = tk_ops_gemm(input, q_weights, q_buf); 
    err = tk_ops_gemm(input, k_weights, k_buf); 
    err = tk_ops_gemm(input, v_weights, v_buf); 

    // 2. core attention mechanism
    // x = [seq_length, hidden_dim] to [seq_length, n_head, hidden_dim] (x_split)

    // transpose(permute) to [n_head, seq_length, hidden_dim] (kT_split)
    // but we need to do physical data moving for contiguous speed up
    if (ctx->rt_type != RT_DRYRUN) {
        tk_tensor_data_reorder(q_split, q_permute);
        tk_tensor_data_reorder(k_split, k_permute);
        tk_tensor_data_reorder(v_split, v_permute);
        tk_tensor_data_reorder(k_split, kT_split);
    }

    // pick each head of Q and K and matmul
    // Qi = [seq_length, head_dim], Ki^T = [head_dim, seq_length]
    // QiKi^T = [seq_length, seq_length]
    // Vi = [seq_length, head_dim]
    // score_buf pre-allocated = [n_head, seq_length, seq_length]
    // score_bufi = [seq_length, seq_length]
    // atten_out = [seq_length, hidden_dim]
    for ( int i = 0 ; i < tf_block->config.n_head ; ++i ) {
        qi->data = q_permute->data + (i * seq_length * head_dim); 
        ki->data = kT_split->data + (i * head_dim * seq_length); 
        vi->data = v_permute->data + (i * seq_length * head_dim); 
        si->data = score_buf->data + (i * seq_length * seq_length); 
        // out = [n_head, seq_length, head_dim]
        // out_i = [seq_length, head_dim]
        out_i = atten_out->data + (i * seq_length * head_dim); 

        // finally doing qi * k^T
        err = tk_ops_gemm(qi, ki, si);
        err = tk_ops_scale(si, scale_factor);
        err = tk_ops_softmax(si, si);

        // softmax(si) * Vi
        err = tk_ops_gemm(si, vi, out_i);
    }

    return atten_out;
}

struct tk_tensor* tf_ffn_forward(struct tk_rt_ctx* ctx, struct TransformerBlock* tf_block, struct tk_tensor* input) {

    int seq_length = tf_block->config->seq_length;
    int hidden_dim = tf_block->config->hidden_dim;
    // 1. Up-projection: [L, D] -> [L, 4D]
    struct tk_tensor* inter = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){seq_length, 4 * hidden_dim}, 2);
    tk_ops_gemm(input, tf_block->ffn_up_weights, inter);

    // 2. gelu
    tk_ops_gelu(inter, inter); 

    // 3. Down-projection: [L, 4D] -> [L, D]
    struct tk_tensor* out = tk_ws_tensor_alloc(ctx, TK_F64, (int[]){seq_length, hidden_dim}, 2);
    tk_ops_gemm(inter, tf_block->ffn_down_weights, out);

    return out;
}

int tf_block_forward(struct tk_rt_ctx* ctx, struct TransformerBlock* tf_block, struct tk_tensor* input) {


    struct tk_tensor* x = input;

    {
        TK_WS_BEGIN(ctx);

        struct tk_tensor* ln1_out = tk_ws_tensor_alloc(ctx, TK_F64, x->shape, x->ndims);
        tk_ops_layernorm(input, tf_block->ln1_gamma, tf_block->ln1_beta, ln1_out);

        struct tk_tensor* attn_out = attention_forward(ctx, tf_block, ln1_out);
        tk_ops_add(x, attn_out, x);

        TK_WS_END(ctx);
    }

    {
        TK_WS_BEGIN(ctx);

        struct tk_tensor* ln2_out = tk_ws_tensor_alloc(ctx, TK_F64, x->shape, x->ndims);
        tk_ops_layernorm(x, tf_block->ln2_gamma, tf_block->ln2_beta, ln2_out);

        struct tk_tensor* ffn_out = tf_ffn_forward(ctx, tf_block, ln2_out);

        tk_ops_add(x, ffn_out, x);

        TK_WS_END(ctx);
    }

    return 0;
}
