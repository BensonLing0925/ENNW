#ifndef TK_PARAMS_H
#define TK_PARAMS_H

#define MAX_FUSED_OPS 4

#include "../../ops/tensor_ops_config.h"

enum rt_op_type {
    TK_OP_GEMM,
    TK_OP_ADD,
    TK_OP_GELU,
    TK_OP_LAYERNORM,
    TK_OP_QUANTIZE,
    TK_OP_ATTENTION,
    TK_OP_FFN,
    TK_OP_FUSED_ADD_NORM,  /* fused residual-add + layer-norm (post-norm only) */
	TK_OP_FUSED_GEMM_ADD_GELU,
};

struct tk_rt_gemm_params {
    int trans_b;
    float alpha;
};

struct tk_rt_layernorm_params {
    float epsilon;
};

struct tk_rt_gelu_params {
	enum tk_ops_gelu_variant gelu_variant;
	int (*gelu_fn) (struct tk_tensor* src, struct tk_tensor* dest);
	int (*gelu_fn_raw) (void* src_data, void* dest_data,
                     	enum tk_dtype, size_t size);
};

struct tk_rt_add_params {
    int is_bias_add;
    float alpha;
};

struct TransformerBlock;

struct tk_rt_attention_params {
    struct TransformerBlock* tf;
};

struct tk_rt_ffn_params {
    struct TransformerBlock* tf;
};

struct tk_rt_quantize_params {
    float calib_scale;
};

union tk_rt_ops_params {
	struct tk_rt_gemm_params        gemm;
	struct tk_rt_add_params         add;
	struct tk_rt_gelu_params        gelu;
	struct tk_rt_layernorm_params   layernorm;
	struct tk_rt_attention_params   attention;
	struct tk_rt_ffn_params         ffn;
	struct tk_rt_quantize_params    quantize;
};

struct tk_rt_fused_ops_entry {
	enum rt_op_type op_type;
	union tk_rt_ops_params params;
};

struct tk_rt_fused_params {
	int count;
	struct tk_rt_fused_ops_entry ops[MAX_FUSED_OPS];
};

#endif
