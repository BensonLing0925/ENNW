#include "tk_ops.h"
#include "../ops/tensor_ops.h"
#include "../error/rt_error.h"
#include "../profiler/tk_profiler.h"
#include "../runtime/rt_context.h"
#include "../modules/transformer/tf_block.h"

int exec_add(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest) {
    RT_CHECK(tk_ops_add(src1, src2, dest));
    return 0;
}

int exec_gemm(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest) {
    RT_CHECK(tk_ops_gemm(src1, src2, dest));
    return 0;
}

int exec_layernorm(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest) {
    RT_CHECK(tk_ops_layernorm(src, gamma, beta, dest));
    return 0;
}

int exec_gelu(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* dest) {
    RT_CHECK(tk_ops_gelu(src, dest));
    return 0;
}

int exec_quantize(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* dst, float calib) {
    (void)ctx; (void)calib;
    /* f32 pass-through: alias data pointer so downstream gemm reads real activations */
    dst->data  = src->data;
    dst->scale = 1.0f;
    return 0;
}

int exec_attention(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
                   struct tk_tensor* input, struct tk_tensor** out) {

    RT_CHECK(tk_tf_attention_forward(ctx, tf, input, out));

    return 0;
}

int exec_ffn(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
             struct tk_tensor* input, struct tk_tensor** out) {
    RT_CHECK(tk_tf_ffn_forward(ctx, tf, input, out));
    return 0;
}

const struct tk_ops_vtable tk_exec_vtable = {
    .add       = exec_add,
    .gemm      = exec_gemm,
    .layernorm = exec_layernorm,
    .gelu      = exec_gelu,
    .quantize  = exec_quantize,
    .attention = exec_attention,
    .ffn       = exec_ffn
};
