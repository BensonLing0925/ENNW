#ifndef TKOPS_H
#define TKOPS_H

#include "tensor.h"

// to solve circular dependency
struct tk_pooling_params {
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int padding_h,    padding_w;
    int input_c, input_h, input_w;
    int pooled_w, pooled_h;
    enum tk_pool_mode { TK_POOL_MAX, TK_POOL_AVG } pType;
};

int tk_ops_add(struct tk_tensor* src1, struct tk_tensor* src2,
               struct tk_tensor* dest);
int tk_ops_gemm(struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest);
int tk_ops_layernorm(struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest);
void tk_ops_gelu(struct tk_tensor* src, struct tk_tensor* dest);
int tk_ops_softmax(struct tk_tensor* src, struct tk_tensor* dest);
int tk_ops_scale(struct tk_tensor* tensor, double scale);
int tk_ops_convolute(struct tk_tensor* pic, struct tk_tensor* filter, struct tk_tensor* out);
int tk_ops_onehot(struct tk_tensor* labels, struct tk_tensor* out);
int tk_ops_pooling(struct tk_pooling_params* params, struct tk_tensor* src, struct tk_tensor* dest);
#endif
