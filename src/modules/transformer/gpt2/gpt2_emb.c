#include "tensor_ops.h"
#include "tk_ops.h"
#include "rt_context.h"
#include "gpt2_emb.h"

struct tk_emb_block* tk_gpt2_emb_create(struct tk_rt_ctx* ctx) {
    return arena_alloc(ctx->meta_arena, sizeof(struct tk_emb_block));
}

// word_emb weights: [vocab_size, hidden_dim]
// pos_emb  weights: [max_seq_len, hidden_dim]
void tk_gpt2_emb_alloc(struct tk_rt_ctx* ctx,
                       struct tk_emb_block* emb) {

    int vocab_size   = emb->config.vocab_size;
    int hidden_dim   = emb->config.hidden_dim;
    int max_seq_len  = emb->config.max_seq_len;

    struct arena* meta = ctx->meta_arena;
    struct arena* data = ctx->data_arena;

    emb->word_emb = arena_alloc(meta, sizeof(struct tk_embedding));
    emb->pos_emb  = arena_alloc(meta, sizeof(struct tk_embedding));

    int word_emb_shape[2] = { vocab_size, hidden_dim };
    int pos_emb_shape[2]  = { max_seq_len, hidden_dim };
    int ln_shape[1]       = { hidden_dim };
	int out_buf_shape[2] = { max_seq_len, hidden_dim };

    enum tk_dtype dtype = ctx->compute_dtype;

    tk_tensor_alloc(data, dtype, word_emb_shape, 2, &emb->word_emb->weights);
    tk_tensor_alloc(data, dtype, pos_emb_shape,  2, &emb->pos_emb->weights);
    tk_tensor_alloc(data, dtype, ln_shape,        1, &emb->ln_gamma);
    tk_tensor_alloc(data, dtype, ln_shape,        1, &emb->ln_beta);
	tk_tensor_alloc(data, dtype, out_buf_shape, 2, &emb->out_buf);

	float emb_scale = 0.02f;
	tk_tensor_rand_init(emb->word_emb->weights, emb_scale);
	tk_tensor_rand_init(emb->pos_emb->weights, emb_scale);

    // ln_gamma = 1, ln_beta = 0
    uint64_t n = (uint64_t)hidden_dim;
    TK_DISPATCH_TYPES(dtype, "gpt2_emb_alloc", {
        scalar_t* gamma_data = (scalar_t*)emb->ln_gamma->data;
        for (uint64_t i = 0; i < n; ++i) gamma_data[i] = (scalar_t)1;
    });
    tk_tensor_fill_zero(emb->ln_beta);
}

/*
// input: token IDs, shape [seq_len] (TK_I32)
// output: [seq_len, hidden_dim] (TK_F64)
int tk_gpt2_emb_forward(struct tk_rt_ctx* ctx,
                        struct tk_gpt2_emb* emb,
                        struct tk_tensor* input,
						int pos_offset,
                        struct tk_tensor** output_ptr) {

	tk_emb_forward(ctx, emb, input, pos_offset, output_ptr)
    return 0;
}
*/
