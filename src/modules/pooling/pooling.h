#ifndef POOLING_H
#define POOLING_H

#include "../runtime/rt_context.h"
#include "../../ops/tensor_ops.h"

typedef enum {
	MAX_POOL,
	AVG_POOL
} PoolingType;		

#define TK_PL_RECT(t, sth, stw, szh, szw, ph, pw) (struct tk_pooling_config){ \
    .pType = t,        \
    .stride_h = sth,   \
    .stride_w = stw,   \
    .kernel_h = szh,   \
    .kernel_w = szw,   \
    .padding_h = ph,   \
    .padding_w = pw    \
}

#define TK_PL_SQR(t, st, sz, p) TK_PL_RECT(t, st, st, sz, sz, p, p)

struct tk_pooling {
    struct tk_pooling_params* params;
	PoolingType pType;
    uint32_t stride_h;
    uint32_t stride_w;

    uint32_t kernel_h;
    uint32_t kernel_w;

    uint32_t padding_h;
    uint32_t padding_w;

    /* dynamic, passed down by previous layer's output */
    int input_h;
    int input_w;
    int input_c;

    /* both dynamic and inferred */
    int padded_h;
    int padded_w;

    int pooled_h;
    int pooled_w;
};

struct tk_pooling_config {
    int padding_h;
    int padding_w;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    PoolingType pType;
};

void tk_pooling_init(struct tk_pooling* pooling, struct tk_pooling_config cfg);
struct tk_pooling* tk_pooling_create(struct tk_rt_ctx* ctx);
void tk_pooling_setup(struct tk_pooling* pooling, struct tk_tensor* input);
struct tk_tensor* tk_pooling_forward(struct tk_rt_ctx* ctx, struct tk_pooling* pooling, struct tk_tensor* input);

#endif
