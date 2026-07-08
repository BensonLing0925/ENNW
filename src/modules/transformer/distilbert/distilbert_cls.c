#include "distilbert_cls.h"
#include "../../../ops/tensor.h"
#include "../../../ops/tensor_ops.h"
#include "../../../runtime/rt_context.h"
#include "../../../runtime/workspaces/rt_workspaces.h"
#include "../../../error/rt_error.h"

struct tk_sst2_cls_head* tk_sst2_cls_create(struct tk_rt_ctx* ctx) {
    return arena_alloc(ctx->meta_arena, sizeof(struct tk_sst2_cls_head));
}

void tk_sst2_cls_alloc(struct tk_rt_ctx* ctx,
                       struct tk_sst2_cls_head* head,
                       int hidden_dim) {
    enum tk_dtype dtype = ctx->compute_dtype;
    struct arena* data  = ctx->data_arena;

    int hw[2]   = { hidden_dim, hidden_dim };
    int h2[2]   = { hidden_dim, 2 };
    int hv[1]   = { hidden_dim };
    int two[1]  = { 2 };

    tk_tensor_alloc(data, dtype, hw, 2, &head->pre_cls_w);
    tk_tensor_alloc(data, dtype, hv, 1, &head->pre_cls_b);
    tk_tensor_alloc(data, dtype, h2, 2, &head->cls_w);
    tk_tensor_alloc(data, dtype, two, 1, &head->cls_b);

    head->hidden_dim = hidden_dim;
}

int tk_sst2_cls_forward(struct tk_rt_ctx* ctx,
                        struct tk_sst2_cls_head* head,
                        struct tk_tensor* hidden,
                        struct tk_tensor** logits_out) {
    int H = head->hidden_dim;

    /* Extract [CLS] token: first row of hidden [seq, H] -> view [1, H] */
    int cls_shape[2] = { 1, H };
    struct tk_tensor* cls_in = NULL;
    RT_CHECK(tk_tensor_view(ctx->meta_arena, hidden, cls_shape, 2, &cls_in));

    /* Allocate workspace tensors */
    int h_shape[2] = { 1, H };
    int o_shape[2] = { 1, 2 };

    struct tk_tensor* pre_out = NULL;
    struct tk_tensor* logits  = NULL;
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                hidden->dtype, h_shape, 2, &pre_out));
    RT_CHECK(tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                hidden->dtype, o_shape, 2, &logits));

    if (ctx->rt_type == RT_DRYRUN) {
        *logits_out = logits;
        return 0;
    }

    /* pre_classifier: [1, H] x [H, H] -> [1, H] */
    /* bias [H] broadcasts over [1, H] via the 1D fast path in tk_ops_add */
    RT_CHECK(tk_ops_gemm(cls_in, head->pre_cls_w, pre_out));
    RT_CHECK(tk_ops_add(pre_out, head->pre_cls_b, pre_out));

    /* ReLU */
    tk_tensor_relu(pre_out);

    /* classifier: [1, H] x [H, 2] -> [1, 2] */

    RT_CHECK(tk_ops_gemm(pre_out, head->cls_w, logits));
    RT_CHECK(tk_ops_add(logits, head->cls_b, logits));

    /* Softmax */
    RT_CHECK(tk_ops_softmax(logits, logits));

    *logits_out = logits;
    return 0;
}
