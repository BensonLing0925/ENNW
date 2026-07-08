#ifndef TK_PARAMS_H
#define TK_PARAMS_H

struct tk_rt_gemm_params {
    int trans_b;
    float alpha;
};

struct tk_rt_layernorm_params {
    float epsilon;
};

struct tk_rt_gelu_params {
    int use_approx;
    int inplace;
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

#endif
