#ifndef TK_DISTILBERT_CLS_H
#define TK_DISTILBERT_CLS_H

#include "tensor.h"
#include "rt_context.h"

/*
 * SST-2 classification head for DistilBertForSequenceClassification.
 * Architecture: Linear(H,H) -> ReLU -> Linear(H,2) -> Softmax
 */
struct tk_sst2_cls_head {
    struct tk_tensor* pre_cls_w;  /* [hidden_dim, hidden_dim] */
    struct tk_tensor* pre_cls_b;  /* [hidden_dim] */
    struct tk_tensor* cls_w;      /* [hidden_dim, 2] */
    struct tk_tensor* cls_b;      /* [2] */
    int hidden_dim;
};

struct tk_sst2_cls_head* tk_sst2_cls_create(struct tk_rt_ctx* ctx);

void tk_sst2_cls_alloc(struct tk_rt_ctx* ctx,
                       struct tk_sst2_cls_head* head,
                       int hidden_dim);

/*
 * Forward pass.
 * hidden : [seq_len, hidden_dim] — full encoder output
 * logits_out : set to a [2] tensor with softmax probabilities
 */
int tk_sst2_cls_forward(struct tk_rt_ctx* ctx,
                        struct tk_sst2_cls_head* head,
                        struct tk_tensor* hidden,
                        struct tk_tensor** logits_out);

#endif
