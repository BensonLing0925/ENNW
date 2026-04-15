#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tf_block.h"
#include "../../ops/tensor.h"
#include "../../ops/tensor_ops.h"
#include "../../runtime/rt_context.h"
#include "../../runtime/workspaces/rt_workspaces.h"

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static void tf_rand_init(struct tk_tensor* t, double scale) {
    double* data = (double*)t->data;
    uint64_t n = shape_size_calc(t->shape, t->ndims);
    for (uint64_t i = 0; i < n; ++i)
        data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
}

static void tf_fill_ones(struct tk_tensor* t) {
    double* data = (double*)t->data;
    uint64_t n = shape_size_calc(t->shape, t->ndims);
    for (uint64_t i = 0; i < n; ++i) data[i] = 1.0;
}

/* ------------------------------------------------------------------ */
/* create / alloc                                                      */
/* ------------------------------------------------------------------ */

struct TransformerBlock* tf_block_create(struct tk_rt_ctx* ctx) {
    return arena_alloc(ctx->meta_arena, sizeof(struct TransformerBlock));
}

void tf_block_alloc(struct tk_rt_ctx* ctx,
                    struct TransformerBlock* tf,
                    int seq_length,
                    int hidden_dim,
                    int n_heads) {

    int head_dim  = hidden_dim / n_heads;
    int inter_dim = 4 * hidden_dim;

    tf->config.seq_length = seq_length;
    tf->config.hidden_dim = hidden_dim;
    tf->config.n_heads    = n_heads;
    tf->config.head_dim   = head_dim;
    tf->config.inter_dim  = inter_dim;

    struct arena* a = ctx->data_arena;
    const double scale = 0.02;

    /* QKV weights: [hidden_dim, hidden_dim] */
    int wqkv[2] = { hidden_dim, hidden_dim };
    tf->q_weights = tk_tensor_alloc(a, TK_F64, wqkv, 2);
    tf->k_weights = tk_tensor_alloc(a, TK_F64, wqkv, 2);
    tf->v_weights = tk_tensor_alloc(a, TK_F64, wqkv, 2);
    tf_rand_init(tf->q_weights, scale);
    tf_rand_init(tf->k_weights, scale);
    tf_rand_init(tf->v_weights, scale);

    /* QKV biases (unused in forward, kept for completeness): [hidden_dim] */
    int wbias[1] = { hidden_dim };
    tf->q_bias = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tf->k_bias = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tf->v_bias = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tk_tensor_fill_zero(tf->q_bias);
    tk_tensor_fill_zero(tf->k_bias);
    tk_tensor_fill_zero(tf->v_bias);

    /* FFN: up [hidden_dim, inter_dim], down [inter_dim, hidden_dim] */
    int wup[2]   = { hidden_dim, inter_dim };
    int wdown[2] = { inter_dim,  hidden_dim };
    tf->ffn_up_weights   = tk_tensor_alloc(a, TK_F64, wup,   2);
    tf->ffn_down_weights = tk_tensor_alloc(a, TK_F64, wdown, 2);
    tf_rand_init(tf->ffn_up_weights,   scale);
    tf_rand_init(tf->ffn_down_weights, scale);

    /* LayerNorm params: gamma = 1, beta = 0, shape [hidden_dim] */
    tf->ln1_gamma = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tf->ln1_beta  = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tf->ln2_gamma = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tf->ln2_beta  = tk_tensor_alloc(a, TK_F64, wbias, 1);
    tf_fill_ones(tf->ln1_gamma);
    tk_tensor_fill_zero(tf->ln1_beta);
    tf_fill_ones(tf->ln2_gamma);
    tk_tensor_fill_zero(tf->ln2_beta);
}

/* ------------------------------------------------------------------ */
/* attention forward                                                   */
/* ------------------------------------------------------------------ */

/*
 * Multi-head self-attention.
 *   input shape: [seq, hidden]
 *   returns atten_out with same shape (allocated from workspace)
 *
 * Strategy: per-head loop with explicit gather/scatter to avoid
 * non-contiguous tensor issues.  No tk_tensor_view used.
 */
static struct tk_tensor* attention_forward(struct tk_rt_ctx* ctx,
                                           struct TransformerBlock* tf,
                                           struct tk_tensor* input) {

    int seq    = tf->config.seq_length;
    int hidden = tf->config.hidden_dim;
    int heads  = tf->config.n_heads;
    int hdim   = tf->config.head_dim;   /* hidden / heads */
    double scale = 1.0 / sqrt((double)hdim);

    /* ---- workspace allocations ---- */
    int qkv_shape[2]   = { seq, hidden };
    int score_shape[2] = { seq, seq };
    int head_shape[2]  = { seq, hdim };
    int headT_shape[2] = { hdim, seq };
    int out_shape[2]   = { seq, hidden };

    struct tk_tensor* q_buf    = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, qkv_shape,   2);
    struct tk_tensor* k_buf    = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, qkv_shape,   2);
    struct tk_tensor* v_buf    = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, qkv_shape,   2);
    struct tk_tensor* score    = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, score_shape, 2);
    struct tk_tensor* qh       = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, head_shape,  2);
    struct tk_tensor* kh       = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, head_shape,  2);
    struct tk_tensor* kh_T     = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, headT_shape, 2);
    struct tk_tensor* vh       = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, head_shape,  2);
    struct tk_tensor* out_h    = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, head_shape,  2);
    struct tk_tensor* atten_out= tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, out_shape,   2);

    /* In dry-run mode we only need workspace sizing — skip computation */
    if (ctx->rt_type == RT_DRYRUN) return atten_out;

    /* ---- QKV projections: [seq, hidden] x [hidden, hidden] -> [seq, hidden] ---- */
    tk_ops_gemm(input, tf->q_weights, q_buf);
    tk_ops_gemm(input, tf->k_weights, k_buf);
    tk_ops_gemm(input, tf->v_weights, v_buf);

    double* q_data    = (double*)q_buf->data;
    double* k_data    = (double*)k_buf->data;
    double* v_data    = (double*)v_buf->data;
    double* qh_data   = (double*)qh->data;
    double* kh_data   = (double*)kh->data;
    double* khT_data  = (double*)kh_T->data;
    double* vh_data   = (double*)vh->data;
    double* out_data  = (double*)atten_out->data;

    for (int h = 0; h < heads; ++h) {
        /* Gather head h's slice (stride = hidden in the full projection) */
        for (int s = 0; s < seq; ++s) {
            for (int j = 0; j < hdim; ++j) {
                qh_data[s * hdim + j] = q_data[s * hidden + h * hdim + j];
                kh_data[s * hdim + j] = k_data[s * hidden + h * hdim + j];
                vh_data[s * hdim + j] = v_data[s * hidden + h * hdim + j];
            }
        }

        /* Transpose K: [seq, hdim] -> [hdim, seq] for score = Q * K^T */
        for (int s = 0; s < seq; ++s)
            for (int j = 0; j < hdim; ++j)
                khT_data[j * seq + s] = kh_data[s * hdim + j];

        /* score = qh [seq, hdim] x kh_T [hdim, seq] -> [seq, seq] */
        tk_ops_gemm(qh, kh_T, score);
        tk_ops_scale(score, scale);
        tk_ops_softmax(score, score);

        /* out_h = score [seq, seq] x vh [seq, hdim] -> [seq, hdim] */
        tk_ops_gemm(score, vh, out_h);

        /* Scatter back into atten_out */
        double* oh = (double*)out_h->data;
        for (int s = 0; s < seq; ++s)
            for (int j = 0; j < hdim; ++j)
                out_data[s * hidden + h * hdim + j] = oh[s * hdim + j];
    }

    return atten_out;
}

/* ------------------------------------------------------------------ */
/* FFN forward                                                         */
/* ------------------------------------------------------------------ */

static struct tk_tensor* tf_ffn_forward(struct tk_rt_ctx* ctx,
                                         struct TransformerBlock* tf,
                                         struct tk_tensor* input) {

    int seq    = tf->config.seq_length;
    int hidden = tf->config.hidden_dim;
    int inter  = tf->config.inter_dim;

    /* Up-projection: [seq, hidden] x [hidden, inter] -> [seq, inter] */
    int inter_shape[2] = { seq, inter };
    struct tk_tensor* inter_buf = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, inter_shape, 2);

    /* Down-projection: [seq, inter] x [inter, hidden] -> [seq, hidden] */
    int out_shape[2] = { seq, hidden };
    struct tk_tensor* out = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, out_shape, 2);

    if (ctx->rt_type != RT_DRYRUN) {
        tk_ops_gemm(input, tf->ffn_up_weights, inter_buf);
        tk_ops_gelu(inter_buf, inter_buf);
        tk_ops_gemm(inter_buf, tf->ffn_down_weights, out);
    }

    return out;
}

/* ------------------------------------------------------------------ */
/* block forward                                                       */
/* ------------------------------------------------------------------ */

/*
 * Pre-norm transformer block.
 * Modifies `input` in-place (residual connections write back to input->data).
 * input shape: [seq_length, hidden_dim]
 */
int tf_block_forward(struct tk_rt_ctx* ctx,
                     struct TransformerBlock* tf,
                     struct tk_tensor* input) {

    int seq    = tf->config.seq_length;
    int hidden = tf->config.hidden_dim;
    int ln_shape[2] = { seq, hidden };

    /* ---- Attention sub-layer ---- */
    {
        size_t ws_save = ctx->ws->cur_offset;

        struct tk_tensor* ln1_out = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, ln_shape, 2);
        if (ctx->rt_type != RT_DRYRUN)
            tk_ops_layernorm(input, tf->ln1_gamma, tf->ln1_beta, ln1_out);

        struct tk_tensor* attn_out = attention_forward(ctx, tf,
                                        ctx->rt_type == RT_DRYRUN ? input : ln1_out);

        if (ctx->rt_type != RT_DRYRUN)
            tk_ops_add(input, attn_out, input);

        ctx->ws->cur_offset = ws_save;  /* release attention temporaries */
    }

    /* ---- FFN sub-layer ---- */
    {
        size_t ws_save = ctx->ws->cur_offset;

        struct tk_tensor* ln2_out = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, ln_shape, 2);
        if (ctx->rt_type != RT_DRYRUN)
            tk_ops_layernorm(input, tf->ln2_gamma, tf->ln2_beta, ln2_out);

        struct tk_tensor* ffn_out = tf_ffn_forward(ctx, tf,
                                        ctx->rt_type == RT_DRYRUN ? input : ln2_out);

        if (ctx->rt_type != RT_DRYRUN)
            tk_ops_add(input, ffn_out, input);

        ctx->ws->cur_offset = ws_save;  /* release FFN temporaries */
    }

    return 0;
}
