#include "embedding.h"
#include "tensor_ops.h"
#include "tk_ops.h"
#include "rt_context.h"

// input: token IDs, shape [seq_len] (TK_I32)
// output: [seq_len, hidden_dim] (TK_F64)
int tk_emb_forward(struct tk_rt_ctx* ctx, struct tk_emb_block* emb,
                    struct tk_tensor* input, int pos_offset,
                    struct tk_tensor** output_ptr) {

	/*
	printf("[emb_forward] out_buf=%p, out_buf->shape[0]=%d, ln_gamma=%p, ln_gamma->shape[0]=%d\n",
       (void*)emb->out_buf, emb->out_buf ? emb->out_buf->shape[0] : -1,
       (void*)emb->ln_gamma, emb->ln_gamma ? emb->ln_gamma->shape[0] : -1);
	*/

    int seq          = input->shape[0];
    int hidden_dim   = emb->word_emb->weights->shape[1];

    int* out_shape = arena_alloc(ctx->meta_arena, sizeof(int) * 2);
    out_shape[0] = seq;
    out_shape[1] = hidden_dim;

    struct tk_tensor* emb_out = NULL;
    RT_CHECK(tk_tensor_view(ctx->meta_arena, emb->out_buf, out_shape, 2, &emb_out));

    if (ctx->rt_type == RT_DRYRUN) {
        *output_ptr = emb_out;
        return 0;
    }

    RT_CHECK(tk_ops_embedding_lookup(input, emb->word_emb->weights, emb_out));

    struct tk_tensor* pos_view = arena_alloc(ctx->meta_arena, sizeof(struct tk_tensor));
    pos_view->dtype   = emb->pos_emb->weights->dtype;
    pos_view->ndims   = 2;
    pos_view->shape   = out_shape;
    pos_view->strides = emb->pos_emb->weights->strides;
    pos_view->data    = (uint8_t*)emb->pos_emb->weights->data
                       + (size_t)pos_offset * hidden_dim * tk_get_dtype_size(pos_view->dtype);

    RT_CHECK(ctx->ops->add(ctx, emb_out, pos_view, emb_out));
	if (emb->config.use_ln) {
		RT_CHECK(ctx->ops->layernorm(ctx, emb_out, emb->ln_gamma, emb->ln_beta, emb_out));
	}

    *output_ptr = emb_out;
    return 0;
}


