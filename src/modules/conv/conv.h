#ifndef CONV_H
#define CONV_H

#include <stdio.h>
#include <stdlib.h>
#include "../mem/arena.h"
#include "../../ops/tensor.h"
#include "../../ops/tensor_ops.h"
#include "../../structDef.h"

typedef struct tk_conv2d {
    /* requisite */
    int input_c;
    int input_h;
    int input_w;

	int num_filter;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int has_bias;
    enum tk_dtype dtype;

    struct tk_tensor* biases;
    // Double2D* filters;
    struct tk_tensor* filters;

    /* inferred, does not store in weights file */
    int filtered_h;  // output size
    int filtered_w;  // output size
    int padded_h;
    int padded_w;

} tk_conv2d;

#define TK_CONV_RECT(f, kh, kw, sh, sw, ph, pw) (struct tk_conv2d_config){ \
    .num_filter = f, \
    .kernel_h = kh, .kernel_w = kw, \
    .stride_h = sh, .stride_w = sw, \
    .padding_h = ph, .padding_w = pw \
}

#define TK_CONV_SQR(f, k, s, p) TK_CONV_RECT(f, k, k, s, s, p, p)
    

struct tk_conv2d_config {
    int num_filter;

    int kernel_h;
    int kernel_w;

    int stride_h; 
    int stride_w;
                       
    int padding_h;
    int padding_w;
};


struct tk_conv2d* tk_conv2D_create(struct tk_rt_ctx* ctx);
void tk_conv2d_init(struct tk_conv2d* conv, struct tk_conv2d_config config);
void tk_conv2d_setup(struct tk_conv2d* conv, struct tk_tensor* input);
int tk_conv2d_load_weights(struct tk_conv2d* conv, FILE* fp);
void tk_conv2d_alloc(struct tk_rt_ctx* ctx, struct tk_conv2d* conv);
struct tk_tensor* tk_conv_forward(struct tk_rt_ctx* ctx, struct tk_conv2d* conv, struct Dataset* dataset);
#endif
