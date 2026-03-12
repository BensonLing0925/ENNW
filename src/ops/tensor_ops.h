#ifndef TKOPS_H
#define TKOPS_H

#include "tensor.h"

int tk_ops_add(struct tk_tensor* src1, struct tk_tensor* src2,
               struct tk_tensor* dest);

int tk_ops_gemm(struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest);

int tk_ops_layernorm(struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest);

void tk_ops_gelu(struct tk_tensor* src, struct tk_tensor* dest);

int tk_ops_softmax(struct tk_tensor* src, struct tk_tensor* dest);

#endif
