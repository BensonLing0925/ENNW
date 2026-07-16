#ifndef TK_OPS_VT_H
#define TK_OPS_VT_H

#include "tensor.h"
#include "../runtime/graph/rt_graph.h"

struct tk_rt_ctx;
struct TransformerBlock;

struct tk_ops_vtable {
    int (*add)(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest);
    int (*gemm)(struct tk_rt_ctx* ctx, struct tk_tensor* src1, struct tk_tensor* src2, struct tk_tensor* dest);
    int (*layernorm)(struct tk_rt_ctx* ctx, struct tk_tensor* src, struct tk_tensor* gamma, struct tk_tensor* beta, struct tk_tensor* dest);
    int (*gelu)(struct tk_rt_ctx* ctx, struct tk_ops_config* oc,
			   struct tk_rt_node* node, struct tk_tensor* src, struct tk_tensor* dest);
    int (*quantize)(struct tk_rt_ctx* ctx,
                    struct tk_tensor* src,
                    struct tk_tensor* dst,
                    float calib_scale);
    int (*attention)(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
                     struct tk_tensor* input, struct tk_tensor** out);
    int (*ffn)(struct tk_rt_ctx* ctx, struct TransformerBlock* tf,
               struct tk_tensor* input, struct tk_tensor** out);
};

// let runtime sees these two table
extern const struct tk_ops_vtable tk_record_vtable;
extern const struct tk_ops_vtable tk_exec_vtable;
extern const struct tk_ops_vtable tk_exec_i8_vtable;

#endif
