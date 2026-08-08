#include "gpt2.h"
#include "rt_graph.h"
#include "tk_profiler.h"

/* ------------------------------------------------------------------ */
/* internal helpers                                                    */
/* ------------------------------------------------------------------ */

static void gpt2_block_config(struct tk_gpt2_block* block,
                                    struct tk_gpt2_config config,
                                    enum tk_dtype dtype) {
    int inter_dim = (config.inter_dim == 0) ? config.hidden_dim * 4
                                            : config.inter_dim;
    block->base->config = (struct tk_tf_block_config){
		.max_seq_len	 = config.max_seq_len,
        .seq_length      = config.seq_length,
        .hidden_dim      = config.hidden_dim,
        .n_heads         = config.n_heads,
        .inter_dim       = inter_dim,
        .use_qkv_bias    = config.use_qkv_bias,
        .use_o_proj      = config.use_o_proj,
        .use_o_proj_bias = config.use_o_proj_bias,
        .use_ffn_bias    = config.use_ffn_bias,
        .pre_norm        = config.use_pre_norm,  /* GPT-2 uses pre-norm */
        .use_causal      = config.use_causal, 
        .dtype           = dtype,

		/* Hardcoded ops_config for now */
		.ops_config = (struct tk_ops_config) {
			.gelu_variant = TK_OPS_GELU_TANH,
			.gelu_fn = &tk_ops_gelu_tanh,
			.gelu_fn_raw = &_tk_ops_gelu_tanh,
		},
    };
}

static void gpt2_emb_config(struct tk_emb_block* emb,
                                  struct tk_gpt2_config config) {
    emb->config = (struct tk_emb_config) {
        .vocab_size  = config.vocab_size,
        .hidden_dim  = config.hidden_dim,
        .max_seq_len = config.max_seq_len,
		.use_ln = 0,
    };
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

int tk_gpt2_config(struct tk_rt_ctx* ctx,
                         struct tk_gpt2_config config,
                         struct tk_gpt2** gpt2_out) {

    struct tk_gpt2* db = arena_alloc(ctx->meta_arena,
                                           sizeof(struct tk_gpt2));
    db->num_layers = config.num_layers;

    db->emb = tk_gpt2_emb_create(ctx);
    gpt2_emb_config(db->emb, config);

    db->blocks = arena_alloc(ctx->meta_arena,
                             sizeof(struct tk_gpt2_block*) * config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        db->blocks[i] = tk_gpt2_block_create(ctx);
        gpt2_block_config(db->blocks[i], config, ctx->compute_dtype);
    }

    *gpt2_out = db;
    return 0;
}

int tk_gpt2_alloc(struct tk_rt_ctx* ctx,
                        struct tk_gpt2_config config,
                        struct tk_gpt2* gpt2) {
    tk_gpt2_emb_alloc(ctx, gpt2->emb);
    for (int i = 0; i < config.num_layers; ++i) {
        tk_gpt2_block_alloc(ctx, gpt2->blocks[i]);
    }

	/* gpt2 specific final ln */
	int ln_shape[1] = { config.hidden_dim };
	enum tk_dtype dtype = TK_F32;
	tk_tensor_alloc(ctx->data_arena, dtype, ln_shape, 1, &gpt2->final_ln_gamma);
	tk_tensor_alloc(ctx->data_arena, dtype, ln_shape, 1, &gpt2->final_ln_beta);

	TK_DISPATCH_TYPES(dtype, "gpt2_final_ln_alloc", {
		scalar_t* gamma_data = (scalar_t*)gpt2->final_ln_gamma->data;
		for (int i = 0; i < config.hidden_dim; ++i) gamma_data[i] = (scalar_t)1;
	});
	tk_tensor_fill_zero(gpt2->final_ln_beta);

	/* since the information we need is after emb_forward, here we set the flag to 0 */
	gpt2->lm_head_ready = 0;


    return 0;
}

int tk_gpt2_forward(struct tk_rt_ctx* ctx,
                    struct tk_gpt2* gpt2,
                    struct tk_tensor* input_ids,
                    struct tk_tensor** output_ptr,
					enum tk_gpt2_forward_mode mode) {

    /* Embedding: token IDs -> [seq, hidden] (workspace tensor at offset 0) */
    struct tk_tensor* hidden = NULL;
	int pos_offset = (mode == TK_GPT2_DECODE) ? ctx->kv_cur_len : 0;

    // RT_CHECK(tk_gpt2_emb_forward(ctx, gpt2->emb, input_ids, pos_offset, &hidden));
    
    uint64_t t_emb_forward_start = tk_get_now_ns();
    RT_CHECK(tk_emb_forward(ctx, gpt2->emb, input_ids, pos_offset, &hidden));
    uint64_t t_emb_forward = tk_get_now_ns() - t_emb_forward_start;
    if (mode == TK_GPT2_PREFILL)
        printf("prefilling embedding forward: %.3f ms\n", t_emb_forward / 1e6);

    if (!gpt2->lm_head_ready) {
		/* weight tying */ 
        uint64_t t_lm_transpose_start = tk_get_now_ns();
		RT_CHECK(tk_tensor_transpose(ctx->meta_arena, gpt2->emb->word_emb->weights, 0, 1, &gpt2->lm_head_weight));
        uint64_t t_lm_transpose = tk_get_now_ns() - t_lm_transpose_start;
        printf("lm head allocation: %.3f ms\n", t_lm_transpose / 1e6);
		gpt2->lm_head_ready = 1;
    }

	struct tk_tensor* logits = NULL;

	tk_attn_fn attn_fn = NULL;


	switch(mode) {
		case TK_GPT2_PREFILL:
			attn_fn = ctx->ops->attention; 
			break;
		case TK_GPT2_DECODE:
			attn_fn = tk_tf_attention_forward_decode;
			break;
		default:
			printf("GPT2 forward mode not set, default to PREFILL mode\n");
			attn_fn = ctx->ops->attention; 
			break;
	}


    /* After tk_rt_prepare (graph_ready=1) the static graph holds the optimised
     * transformer-block schedule.  Use graph exec to pick up any fusions. */
    if (mode == TK_GPT2_PREFILL && ctx->graph_ready && ctx->rt_type == RT_INFERENCE) {
        uint64_t t_graph_exec_start = tk_get_now_ns();
        RT_CHECK(tk_rt_graph_exec(ctx));
        uint64_t t_graph_exec = tk_get_now_ns() - t_graph_exec_start;
        printf("prefill graph execute: %.3f ms\n", t_graph_exec / 1e6);
		logits = ctx->static_graph->last_node->outputs[0];
    } else {
        /* Dry-run and no-graph-optimise paths: run blocks normally. */
        for (int i = 0; i < gpt2->num_layers; ++i) {
            RT_CHECK(tk_tf_block_forward(ctx, gpt2->blocks[i]->base, hidden, attn_fn));
		}

		/* gpt2's final layernorm */
		enum tk_dtype dtype = TK_F32;
		struct tk_tensor* ln_out = NULL;
		RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, hidden->shape, hidden->ndims, &ln_out));
		ctx->ops->layernorm(ctx, hidden, gpt2->final_ln_gamma, gpt2->final_ln_beta, ln_out);
		
		/* LM head  */
		int logits_shape[2] = { ln_out->shape[0], gpt2->emb->config.vocab_size };
		RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, dtype, logits_shape, 2, &logits));
		ctx->ops->gemm(ctx, ln_out, gpt2->lm_head_weight, logits);
    }

    *output_ptr = logits;
    return 0;
}

int argmax_f32(const float* data, int n) {
    int best_idx = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int tk_gpt2_generate(struct tk_rt_ctx* ctx, struct tk_gpt2* gpt2,
                     struct tk_tensor* prompt_ids, int max_new_tokens, int eos_token_id,
                     int32_t** out_tokens, int* out_count) {

    /* ---- allocate KV cache for every layer, reset generation state ---- */
    for (int i = 0; i < gpt2->num_layers; ++i) {
        tk_tf_kv_cache_alloc(ctx, gpt2->blocks[i]->base);
	}
    ctx->kv_cur_len = 0;

    int prompt_len = prompt_ids->shape[0];
    int32_t* tokens = arena_alloc(ctx->meta_arena, sizeof(int32_t) * (prompt_len + max_new_tokens));
    memcpy(tokens, prompt_ids->data, sizeof(int32_t) * prompt_len);
    int total_count = prompt_len;

    /* ---- Prefill: process the whole prompt at once ---- */
    ctx->ws->cur_offset = 0;
    struct tk_tensor* logits = NULL;

    uint64_t prefill_tstart = tk_get_now_ns();
    RT_CHECK(tk_gpt2_forward(ctx, gpt2, prompt_ids, &logits, TK_GPT2_PREFILL));
    uint64_t prefill_tend = tk_get_now_ns();
    uint64_t t_prefill = (prefill_tend - prefill_tstart);

    /* take the last row of logits [prompt_len, vocab] -> [vocab], argmax */
    int vocab_size = gpt2->emb->config.vocab_size;
    float* last_row = (float*)logits->data + (size_t)(prompt_len - 1) * vocab_size;
    int next_token = argmax_f32(last_row, vocab_size);
    tokens[total_count++] = next_token;

    uint64_t t_decode_total = 0;
    int decode_steps = 0;

    /* ---- Decode loop: one token at a time ---- */
    for (int step = 0; step < max_new_tokens - 1; ++step) {
        if (next_token == eos_token_id) break;

        int in_shape[1] = { 1 };
        struct tk_tensor* step_input = NULL;
        RT_CHECK(tk_tensor_alloc(ctx->meta_arena, TK_I32, in_shape, 1, &step_input));
        *(int32_t*)step_input->data = next_token;

        ctx->ws->cur_offset = 0;

        uint64_t decode_tstart = tk_get_now_ns();
        RT_CHECK(tk_gpt2_forward(ctx, gpt2, step_input, &logits, TK_GPT2_DECODE));
        uint64_t decode_tend = tk_get_now_ns();

        t_decode_total += (decode_tend - decode_tstart);
        decode_steps++;

        /* logits here is [1, vocab] — only one row */
        next_token = argmax_f32((float*)logits->data, vocab_size);
        tokens[total_count++] = next_token;

        ctx->kv_cur_len++;   /* advance cache position after all layers processed this token */
    }

    printf("\n=== Timing ===\n");
    printf("Prefill (%d tokens) : %.3f ms\n", prompt_len, t_prefill / 1e6);
    printf("Decode total        : %.3f ms (%d steps)\n", t_decode_total / 1e6, decode_steps);
    printf("Decode per token    : %.3f ms\n", (double)t_decode_total / decode_steps / 1e6);
    printf("Total               : %.3f ms\n", (t_prefill + t_decode_total) / 1e6);

    *out_tokens = tokens;
    *out_count  = total_count;
    return 0;
}
