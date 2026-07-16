#include "distilbert.h"
#include "rt_graph.h"

/* ------------------------------------------------------------------ */
/* internal helpers                                                    */
/* ------------------------------------------------------------------ */

static void distilbert_block_config(struct tk_distilbert_block* block,
                                    struct tk_distilbert_config config,
                                    enum tk_dtype dtype) {
    int inter_dim = (config.inter_dim == 0) ? config.hidden_dim * 4
                                            : config.inter_dim;
    block->base->config = (struct tk_tf_block_config){
        .seq_length      = config.seq_length,
        .hidden_dim      = config.hidden_dim,
        .n_heads         = config.n_heads,
        .inter_dim       = inter_dim,
        .use_qkv_bias    = config.use_qkv_bias,
        .use_o_proj      = config.use_o_proj,
        .use_o_proj_bias = config.use_o_proj_bias,
        .use_ffn_bias    = config.use_ffn_bias,
        .pre_norm        = 0,  /* DistilBERT uses post-norm */
        .dtype           = dtype,

		/* Hardcoded ops_config for now */
		.ops_config = (struct tk_ops_config) {
			.gelu_variant = TK_OPS_GELU_ERF,
			.gelu_fn = &tk_ops_gelu_erf,
			.gelu_fn_raw = &_tk_ops_gelu_erf,
		},
    };
}

static void distilbert_emb_config(struct tk_distilbert_emb* emb,
                                  struct tk_distilbert_config config) {
    emb->config = (struct tk_distilbert_emb_config){
        .vocab_size  = config.vocab_size,
        .hidden_dim  = config.hidden_dim,
        .max_seq_len = config.max_seq_len,
    };
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

int tk_distilbert_config(struct tk_rt_ctx* ctx,
                         struct tk_distilbert_config config,
                         struct tk_distilbert** distilbert_out) {

    struct tk_distilbert* db = arena_alloc(ctx->meta_arena,
                                           sizeof(struct tk_distilbert));
    db->num_layers = config.num_layers;

    db->emb = tk_distilbert_emb_create(ctx);
    distilbert_emb_config(db->emb, config);

    db->blocks = arena_alloc(ctx->meta_arena,
                             sizeof(struct tk_distilbert_block*) * config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        db->blocks[i] = tk_distilbert_block_create(ctx);
        distilbert_block_config(db->blocks[i], config, ctx->compute_dtype);
    }

    *distilbert_out = db;
    return 0;
}

int tk_distilbert_alloc(struct tk_rt_ctx* ctx,
                        struct tk_distilbert_config config,
                        struct tk_distilbert* distilbert) {
    tk_distilbert_emb_alloc(ctx, distilbert->emb);
    for (int i = 0; i < config.num_layers; ++i) {
        tk_distilbert_block_alloc(ctx, distilbert->blocks[i]);
    }
    return 0;
}

int tk_distilbert_forward(struct tk_rt_ctx* ctx,
                          struct tk_distilbert* distilbert,
                          struct tk_tensor* input_ids,
                          struct tk_tensor** output_ptr) {

    /* Embedding: token IDs -> [seq, hidden] (workspace tensor at offset 0) */
    struct tk_tensor* hidden = NULL;
    RT_CHECK(tk_distilbert_emb_forward(ctx, distilbert->emb, input_ids, &hidden));

    /* After tk_rt_prepare (graph_ready=1) the static graph holds the optimised
     * transformer-block schedule.  Use graph exec to pick up any fusions. */
    if (ctx->graph_ready && ctx->rt_type == RT_INFERENCE) {
        RT_CHECK(tk_rt_graph_exec(ctx));
    } else {
        /* Dry-run and no-graph-optimise paths: run blocks normally. */
        for (int i = 0; i < distilbert->num_layers; ++i)
            RT_CHECK(tk_tf_block_forward(ctx, distilbert->blocks[i]->base, hidden));
    }

    *output_ptr = hidden;
    return 0;
}
