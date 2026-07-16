#include "tk_ops.h"
#include "tensor_ops.h"
#include "tensor_quantize.h"
#include "../error/rt_error.h"
#include "tk_ops_exec.h"

struct TransformerBlock;
extern int exec_add(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest);
extern int exec_layernorm(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest);
extern int exec_gelu(struct tk_rt_ctx* ctx, struct tk_ops_config* oc, struct tk_tensor* src, struct tk_tensor* dest);
extern int exec_attention(struct tk_rt_ctx* ctx, struct TransformerBlock* tf, struct tk_tensor* input, struct tk_tensor** out);
extern int exec_ffn(struct tk_rt_ctx* ctx, struct TransformerBlock* tf, struct tk_tensor* input, struct tk_tensor** out);

int exec_i8f32_gemm(struct tk_rt_ctx* ctx, struct tk_tensor* src1, 
                    struct tk_tensor* src2, struct tk_tensor* dest) {
    RT_CHECK(tk_ops_gemm_i8f32(src1, src2, dest));
    return 0;
}

int exec_i8_quantize(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* dst, float calib) {
    float final_scale = tk_tf_quantize(src, dst, calib);
    dst->scale = final_scale; 
    return 0;
}

const struct tk_ops_vtable tk_exec_i8_vtable = {
    .add       = exec_add,
    .gemm      = exec_i8f32_gemm,
    .layernorm = exec_layernorm,
    .gelu      = exec_gelu,
    .quantize  = exec_i8_quantize,
    .attention = exec_attention,
    .ffn       = exec_ffn
};
