#include "pooling.h"
#include "../../runtime/rt_context.h"

struct tk_pooling* tk_pooling_create(struct tk_rt_ctx* ctx) {
    struct tk_pooling* pooling = arena_alloc(ctx->meta_arena, sizeof(struct tk_pooling));
    pooling->params = arena_alloc(ctx->meta_arena, sizeof(struct tk_pooling_params));
    return pooling;
}

void tk_pooling_init(struct tk_pooling* pooling, struct tk_pooling_config cfg) {
    pooling->params->padding_h = pooling->padding_h = cfg.padding_h; 
    pooling->params->padding_w = pooling->padding_w = cfg.padding_w; 
    pooling->params->kernel_h = pooling->kernel_h = cfg.kernel_h;
    pooling->params->kernel_w = pooling->kernel_w = cfg.kernel_w;
    pooling->params->stride_h = pooling->stride_h = cfg.stride_h;
    pooling->params->stride_w = pooling->stride_w = cfg.stride_w;
    pooling->params->pType = pooling->pType = cfg.pType;
}

void tk_pooling_setup(struct tk_pooling* pooling, struct tk_tensor* input) {
    int input_c = input->shape[input->ndims-3];
    int input_h = input->shape[input->ndims-2];
    int input_w = input->shape[input->ndims-1];

    pooling->params->input_c = pooling->input_c = input_c;
    pooling->params->input_h = pooling->input_h = input_h;
    pooling->params->input_w = pooling->input_w = input_w;

    int padding_h = pooling->padding_h;
    int padding_w = pooling->padding_w;
    pooling->padded_h = input_h + 2 * padding_h;
    pooling->padded_w = input_w + 2 * padding_w;

    int pooled_h = ((input_h - pooling->kernel_h + 2 * pooling->padding_h) / pooling->stride_h) + 1;
    int pooled_w = ((input_w - pooling->kernel_w + 2 * pooling->padding_w) / pooling->stride_w) + 1;
    pooling->params->pooled_h = pooling->pooled_h = pooled_h;
    pooling->params->pooled_w = pooling->pooled_w = pooled_w;
}

struct tk_tensor* tk_pooling_forward(struct tk_rt_ctx* ctx, struct tk_pooling* pooling, struct tk_tensor* input) {
    tk_pooling_setup(pooling, input);

    int padded_shape[input->ndims];
    memcpy(padded_shape, input->shape,  sizeof(int) * input->ndims);
    padded_shape[input->ndims-2] = pooling->padded_h;
    padded_shape[input->ndims-1] = pooling->padded_w;

    struct tk_tensor* padded_input = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, input->dtype, padded_shape, input->ndims);
    struct tk_tensor* output = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, input->dtype, (int[]){pooling->input_c, pooling->pooled_h, pooling->pooled_w}, 3);
    if (ctx->rt_type != RT_DRYRUN) {
        tk_tensor_padding(input, padded_input,
                          pooling->padding_h, pooling->padding_w);
        tk_ops_pooling(pooling->params, padded_input, output);
    }
    return output;
}

