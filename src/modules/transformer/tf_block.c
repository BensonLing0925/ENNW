#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tf_block.h"
#include "../../ops/tk_ops.h"
#include "../../ops/tensor.h"
#include "../../ops/tensor_ops.h"
#include "../../runtime/rt_context.h"
#include "../../runtime/workspaces/rt_workspaces.h"

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static void tk_tf_rand_init(struct tk_tensor* t, double scale) {
    uint64_t n = shape_size_calc(t->shape, t->ndims);
    TK_DISPATCH_TYPES(t->dtype, "tk_tf_rand_init", {
        scalar_t* data = (scalar_t*)t->data;
        for (uint64_t i = 0; i < n; ++i)
            data[i] = (scalar_t)(((double)rand() / RAND_MAX * 2.0 - 1.0) * scale);
    });
}

static void tk_tf_fill_ones(struct tk_tensor* t) {
    uint64_t n = shape_size_calc(t->shape, t->ndims);
    TK_DISPATCH_TYPES(t->dtype, "tk_tf_fill_ones", {
        scalar_t* data = (scalar_t*)t->data;
        for (uint64_t i = 0; i < n; ++i) data[i] = (scalar_t)1;
    });
}

/* ------------------------------------------------------------------ */
/* create / alloc                                                      */
/* ------------------------------------------------------------------ */

struct TransformerBlock* tk_tf_block_create(struct tk_rt_ctx* ctx) {
    return arena_alloc(ctx->meta_arena, sizeof(struct TransformerBlock));
}

// config should be setup and inferred before calling alloc
void tk_tf_block_alloc(struct tk_rt_ctx* ctx,
                       struct TransformerBlock* tf) {

    struct tk_tf_block_config* config = &tf->config;
    int hidden_dim = config->hidden_dim;
    int n_heads    = config->n_heads;
    int inter_dim  = (config->inter_dim == 0) ? 4 * hidden_dim : config->inter_dim;
    enum tk_dtype dtype  = (config->dtype == TK_NONE) ? TK_F64 : config->dtype;
    /* When int8 mode is requested, linear weights are stored as TK_I8.
     * Biases and LayerNorm parameters always stay in the compute dtype. */
    enum tk_dtype wdtype = ctx->use_int8 ? TK_I8 : dtype;
    (void)n_heads;

    struct arena* a = ctx->data_arena;
    const double scale = 0.02;

    /* QKV weights: [hidden_dim, hidden_dim] */
    int wqkv[2] = { hidden_dim, hidden_dim };
    tk_tensor_alloc(a, wdtype, wqkv, 2, &tf->q_weights);
    tk_tensor_alloc(a, wdtype, wqkv, 2, &tf->k_weights);
    tk_tensor_alloc(a, wdtype, wqkv, 2, &tf->v_weights);
    if (!ctx->use_int8) {
        tk_tf_rand_init(tf->q_weights, scale);
        tk_tf_rand_init(tf->k_weights, scale);
        tk_tf_rand_init(tf->v_weights, scale);
    } else {
        tk_tensor_fill_zero(tf->q_weights);
        tk_tensor_fill_zero(tf->k_weights);
        tk_tensor_fill_zero(tf->v_weights);
        tf->q_scale = tf->k_scale = tf->v_scale = 1.0f;
    }

    /* QKV biases: always compute dtype */
    int wbias[1] = { hidden_dim };
    if (config->use_qkv_bias) {
        tk_tensor_alloc(a, dtype, wbias, 1, &tf->q_bias);
        tk_tensor_alloc(a, dtype, wbias, 1, &tf->k_bias);
        tk_tensor_alloc(a, dtype, wbias, 1, &tf->v_bias);
        tk_tensor_fill_zero(tf->q_bias);
        tk_tensor_fill_zero(tf->k_bias);
        tk_tensor_fill_zero(tf->v_bias);
    }

    /* output projection */
    if (config->use_o_proj) {
        tk_tensor_alloc(a, wdtype, wqkv, 2, &tf->o_proj_weights);
        if (!ctx->use_int8) tk_tf_rand_init(tf->o_proj_weights, scale);
        else { tk_tensor_fill_zero(tf->o_proj_weights); tf->o_proj_scale = 1.0f; }
        if (config->use_o_proj_bias)
            tk_tensor_alloc(a, dtype, wbias, 1, &tf->o_proj_bias);
    }

    /* FFN weights: wdtype; biases: dtype */
    int wup[2]   = { hidden_dim, inter_dim };
    int wdown[2] = { inter_dim,  hidden_dim };
    tk_tensor_alloc(a, wdtype, wup,   2, &tf->ffn_up_weights);
    tk_tensor_alloc(a, wdtype, wdown, 2, &tf->ffn_down_weights);
    if (!ctx->use_int8) {
        tk_tf_rand_init(tf->ffn_up_weights,   scale);
        tk_tf_rand_init(tf->ffn_down_weights, scale);
    } else {
        tk_tensor_fill_zero(tf->ffn_up_weights);
        tk_tensor_fill_zero(tf->ffn_down_weights);
        tf->ffn_up_scale = tf->ffn_down_scale = 1.0f;
    }

    if (config->use_ffn_bias) {
        int ffn_up_wbias[1]   = { inter_dim };
        int ffn_down_wbias[1] = { hidden_dim };
        tk_tensor_alloc(a, dtype, ffn_up_wbias,   1, &tf->ffn_up_bias);
        tk_tensor_alloc(a, dtype, ffn_down_wbias, 1, &tf->ffn_down_bias);
    }

    /* LayerNorm params: always compute dtype */
    tk_tensor_alloc(a, dtype, wbias, 1, &tf->ln1_gamma);
    tk_tensor_alloc(a, dtype, wbias, 1, &tf->ln1_beta);
    tk_tensor_alloc(a, dtype, wbias, 1, &tf->ln2_gamma);
    tk_tensor_alloc(a, dtype, wbias, 1, &tf->ln2_beta);
    tk_tf_fill_ones(tf->ln1_gamma);
    tk_tensor_fill_zero(tf->ln1_beta);
    tk_tf_fill_ones(tf->ln2_gamma);
    tk_tensor_fill_zero(tf->ln2_beta);

    /* Activation calibration scales: 0 = dynamic (default) */
    tf->attn_in_act_scale     = 0.0f;
    tf->o_proj_in_act_scale   = 0.0f;
    tf->ffn_up_in_act_scale   = 0.0f;
    tf->ffn_down_in_act_scale = 0.0f;
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
int tk_tf_attention_forward(struct tk_rt_ctx* ctx,
                             struct TransformerBlock* tf,
                             struct tk_tensor* input,
                             struct tk_tensor** attn_out_ptr) {

    int seq    = tf->config.seq_length;
    int hidden = tf->config.hidden_dim;
    int heads  = tf->config.n_heads;
    int hdim   = tf->config.hidden_dim / tf->config.n_heads;
    double scale = 1.0 / sqrt((double)hdim);
    int dryrun = (ctx->rt_type == RT_DRYRUN);
    enum tk_dtype dtype = input->dtype;

    /* ---- workspace allocations ---- */
    int qkv_shape[2]   = { seq, hidden };
    int score_shape[2] = { seq, seq };
    int head_shape[2]  = { seq, hdim };
    int headT_shape[2] = { hdim, seq };
    int out_shape[2]   = { seq, hidden };

    struct tk_tensor* q_buf    = NULL;
    struct tk_tensor* k_buf    = NULL;
    struct tk_tensor* v_buf    = NULL;
    struct tk_tensor* score    = NULL;
    struct tk_tensor* qh       = NULL;
    struct tk_tensor* kh       = NULL;
    struct tk_tensor* kh_T     = NULL;
    struct tk_tensor* vh       = NULL;
    struct tk_tensor* out_h    = NULL;
    struct tk_tensor* atten_out= NULL;
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, qkv_shape,   2, &q_buf));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, qkv_shape,   2, &k_buf));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, qkv_shape,   2, &v_buf));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, score_shape, 2, &score));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, head_shape,  2, &qh));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, head_shape,  2, &kh));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, headT_shape, 2, &kh_T));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, head_shape,  2, &vh));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, head_shape,  2, &out_h));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, out_shape,   2, &atten_out));

    /* int8 mode: pre-allocate activation buffer so dryrun measures peak correctly */
    int use_i8 = tk_check_weight_is_i8(tf->q_weights);
    struct tk_tensor* input_i8 = NULL;
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, 
                            use_i8 ? TK_I8 : TK_F32, 
                            qkv_shape, 2, &input_i8));

    /* In dry-run mode we only need workspace sizing — skip computation.
     * IMPORTANT: allocate the same tensors that inference would allocate so
     * the recorded output pointer matches the inference output pointer.
     * When o_proj is present, inference returns proj_out (not atten_out), so
     * we must also allocate proj_out/ao_quant here and return proj_out. */
    if (dryrun) {
        if (tf->o_proj_weights) {
            struct tk_tensor* proj_out = NULL;
            RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, out_shape, 2, &proj_out));
            struct tk_tensor* ao_quant = NULL;
            enum tk_dtype target_dtype = (tf->o_proj_weights->dtype == TK_I8) ? TK_I8 : TK_F32;
            RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, target_dtype, out_shape, 2, &ao_quant));
            (void)ao_quant;
            *attn_out_ptr = proj_out;
        } else {
            *attn_out_ptr = atten_out;
        }
        return 0;
    }

    /* ---- QKV projections: [seq, hidden] x [hidden, hidden] -> [seq, hidden] ---- */
    // if use_i8, input_i8 is int8
    // if not use_i8, input_i8 is actually fp32
    // using input_i8 to unite function calls
    ctx->ops->quantize(ctx, input, input_i8, tf->attn_in_act_scale);
    ctx->ops->gemm(ctx, input_i8, tf->q_weights, q_buf);
    if (tf->q_bias) ctx->ops->add(ctx, q_buf, tf->q_bias, q_buf);
    ctx->ops->gemm(ctx, input_i8, tf->k_weights, k_buf);
    if (tf->k_bias) ctx->ops->add(ctx, k_buf, tf->k_bias, k_buf);
    ctx->ops->gemm(ctx, input_i8, tf->v_weights, v_buf);
    if (tf->v_bias) ctx->ops->add(ctx, v_buf, tf->v_bias, v_buf);

    TK_DISPATCH_TYPES(dtype, "tk_tf_attention_forward", {
        scalar_t* q_data   = (scalar_t*)q_buf->data;
        scalar_t* k_data   = (scalar_t*)k_buf->data;
        scalar_t* v_data   = (scalar_t*)v_buf->data;
        scalar_t* qh_data  = (scalar_t*)qh->data;
        scalar_t* kh_data  = (scalar_t*)kh->data;
        scalar_t* khT_data = (scalar_t*)kh_T->data;
        scalar_t* vh_data  = (scalar_t*)vh->data;
        scalar_t* out_data = (scalar_t*)atten_out->data;

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
            ctx->ops->gemm(ctx, qh, kh_T, score);
            // ctx->ops->scale(ctx, score, scale);
            tk_ops_scale(score, scale);

			if (tf->config.use_causal) {
				tk_tensor_causal_mask(score);
			}

            tk_ops_softmax(score, score);

            /* out_h = score [seq, seq] x vh [seq, hdim] -> [seq, hdim] */
            ctx->ops->gemm(ctx, score, vh, out_h);

            /* Scatter back into atten_out */
            scalar_t* oh = (scalar_t*)out_h->data;
            for (int s = 0; s < seq; ++s)
                for (int j = 0; j < hdim; ++j)
                    out_data[s * hidden + h * hdim + j] = oh[s * hdim + j];
        }
    });

    if (tf->o_proj_weights) {
        /* o_proj cannot write in-place into atten_out (GEMM aliasing bug). */
        struct tk_tensor* proj_out = NULL;
        RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, out_shape, 2, &proj_out));

        struct tk_tensor* ao_quant = NULL;
        enum tk_dtype target_dtype = (tf->o_proj_weights->dtype == TK_I8) ? TK_I8 : TK_F32;
        RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, target_dtype, out_shape, 2, &ao_quant));

        RT_CHECK(ctx->ops->quantize(ctx, atten_out, ao_quant, tf->o_proj_in_act_scale));
        RT_CHECK(ctx->ops->gemm(ctx, ao_quant, tf->o_proj_weights, proj_out));

        if (tf->o_proj_bias)
            RT_CHECK(ctx->ops->add(ctx, proj_out, tf->o_proj_bias, proj_out));

        *attn_out_ptr = proj_out;
        return 0;
    }

    *attn_out_ptr = atten_out;
    return 0;
}

/* ------------------------------------------------------------------ */
/* FFN forward                                                         */
/* ------------------------------------------------------------------ */

int tk_tf_ffn_forward(struct tk_rt_ctx* ctx,
                      struct TransformerBlock* tf,
                      struct tk_tensor* input,
                      struct tk_tensor** ffn_out_ptr) {

    int seq    = tf->config.seq_length;
    int hidden = tf->config.hidden_dim;
    int inter  = tf->config.inter_dim;

    enum tk_dtype dtype = input->dtype;

    /* Up-projection: [seq, hidden] x [hidden, inter] -> [seq, inter] */
    int inter_shape[2] = { seq, inter };
    struct tk_tensor* inter_buf = NULL;
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, inter_shape, 2, &inter_buf));

    /* Down-projection: [seq, inter] x [inter, hidden] -> [seq, hidden] */
    int out_shape[2] = { seq, hidden };
    struct tk_tensor* out = NULL;
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, out_shape, 2, &out));

    enum tk_dtype act_dtype = tk_check_weight_is_i8(tf->ffn_up_weights) ? TK_I8 : TK_F32;

    struct tk_tensor* inp_i8 = NULL;
    int inp_shape[2] = { seq, hidden };
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, act_dtype, inp_shape, 2, &inp_i8));

    struct tk_tensor* inter_i8 = NULL;
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, act_dtype, inter_shape, 2, &inter_i8));

    if (ctx->rt_type == RT_DRYRUN) {
        *ffn_out_ptr = out;
		// return 0;
    }

    // --- Up-projection (hidden -> inter) ---
    RT_CHECK(ctx->ops->quantize(ctx, input, inp_i8, tf->ffn_up_in_act_scale));
    RT_CHECK(ctx->ops->gemm(ctx, inp_i8, tf->ffn_up_weights, inter_buf));

    if (tf->config.use_ffn_bias)
        RT_CHECK(ctx->ops->add(ctx, inter_buf, tf->ffn_up_bias, inter_buf));

    // GELU 運算（輸出維持 F32）
    RT_CHECK(ctx->ops->gelu(ctx, &tf->config.ops_config, NULL, inter_buf, inter_buf));

    // --- Down-projection (inter -> hidden) ---
    // 再次量化：將 GELU 後的結果轉回 I8 以供下一層矩陣乘法使用
    RT_CHECK(ctx->ops->quantize(ctx, inter_buf, inter_i8, tf->ffn_down_in_act_scale));
    RT_CHECK(ctx->ops->gemm(ctx, inter_i8, tf->ffn_down_weights, out));

    if (tf->config.use_ffn_bias)
        RT_CHECK(ctx->ops->add(ctx, out, tf->ffn_down_bias, out));

    *ffn_out_ptr = out;
    return 0;
}

/* ------------------------------------------------------------------ */
/* block forward                                                       */
/* ------------------------------------------------------------------ */

/*
 * Pre-norm transformer block.
 * Modifies `input` in-place (residual connections write back to input->data).
 * input shape: [seq_length, hidden_dim]
 */
int tk_tf_block_forward(struct tk_rt_ctx* ctx,
                     struct TransformerBlock* tf,
                     struct tk_tensor* input) {

    int seq    = tf->config.seq_length;
    int hidden = tf->config.hidden_dim;
    int ln_shape[2] = { seq, hidden };
    enum tk_dtype dtype = input->dtype;

    if (tf->config.pre_norm) {
        /* ---- Pre-norm (GPT-style): LN -> sublayer -> residual ---- */

        struct tk_tensor* ln1_out = NULL;
        RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, ln_shape, 2, &ln1_out));
        ctx->ops->layernorm(ctx, input, tf->ln1_gamma, tf->ln1_beta, ln1_out);

        struct tk_tensor* attn_out = NULL;
        RT_CHECK(ctx->ops->attention(ctx, tf, ln1_out, &attn_out));
        ctx->ops->add(ctx, input, attn_out, input);

        struct tk_tensor* ln2_out = NULL;
        RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, ln_shape, 2, &ln2_out));
        ctx->ops->layernorm(ctx, input, tf->ln2_gamma, tf->ln2_beta, ln2_out);

        struct tk_tensor* ffn_out = NULL;
        // RT_CHECK(ctx->ops->ffn(ctx, tf, ln2_out, &ffn_out));
        RT_CHECK(tk_tf_ffn_forward(ctx, tf, ln2_out, &ffn_out));
        ctx->ops->add(ctx, input, ffn_out, input);

    } else {
        /* ---- Post-norm (BERT/DistilBERT-style): sublayer -> residual -> LN ---- */

        struct tk_tensor* attn_out = NULL;
        RT_CHECK(ctx->ops->attention(ctx, tf, input, &attn_out));
        RT_CHECK(ctx->ops->add(ctx, input, attn_out, input));
        ctx->ops->layernorm(ctx, input, tf->ln1_gamma, tf->ln1_beta, input);

        struct tk_tensor* ffn_out = NULL;
        // RT_CHECK(ctx->ops->ffn(ctx, tf, input, &ffn_out));
        RT_CHECK(tk_tf_ffn_forward(ctx, tf, input, &ffn_out));
        RT_CHECK(ctx->ops->add(ctx, input, ffn_out, input));
        ctx->ops->layernorm(ctx, input, tf->ln2_gamma, tf->ln2_beta, input);
    }
    return 0;
}
