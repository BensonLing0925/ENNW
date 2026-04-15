#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <inttypes.h>
#include "../../ops/tensor_ops.h"
#include "fc.h"

struct Network* tk_fc_create(struct tk_rt_ctx* ctx) {
    struct Network* network = arena_alloc(ctx->meta_arena, sizeof(struct Network));
    return network;
}

struct LinearConfig* tk_ln_cfg_create(int num_configs, struct arena* a) {
    struct LinearConfig* config = arena_alloc(a, sizeof(struct LinearConfig) * num_configs);
    return config;
}

struct LinearConfigList* tk_ln_cfgls_create(int num_linears, struct arena* a) {
    struct LinearConfigList* context = arena_alloc(a, sizeof(struct LinearConfigList));
    context->configs = tk_ln_cfg_create(num_linears, a);
    context->num_configs = num_linears;
    return context;
}

/* Initialize weights/bias/delta after they have been allocated */
int tk_ln_weights_init(struct Linear* linear) {
    struct tk_tensor* weights = linear->weights;
    struct tk_tensor* bias    = linear->bias;

    double* weight_data = (double*)weights->data;
    double* bias_data   = (double*)bias->data;

    /* Flat random init (Xavier-style small values) */
    uint64_t weight_total = shape_size_calc(weights->shape, weights->ndims);
    for (uint64_t i = 0; i < weight_total; ++i)
        weight_data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.01;

    int bias_size = bias->shape[0];
    for (int i = 0; i < bias_size; ++i)
        bias_data[i] = 0.0;

    if (linear->is_training && linear->delta) {
        double* delta_data = (double*)linear->delta->data;
        uint64_t delta_total = shape_size_calc(linear->delta->shape, linear->delta->ndims);
        for (uint64_t i = 0; i < delta_total; ++i)
            delta_data[i] = 0.0;
    }
    return 0;
}

Linear* create_Linear(struct tk_rt_ctx* ctx, struct LinearConfig* config, struct arena* a) {
    struct Linear* linear = arena_alloc(ctx->meta_arena, sizeof(struct Linear));
    linear->num_neurons  = config->out_dim;
    linear->input_dim    = config->in_dim;
    linear->has_bias     = 1;
    linear->is_training  = config->is_training;

    enum tk_dtype dtype = config->dtype;

    /* weights: [in_dim, out_dim]  =>  GEMM: [N, in] x [in, out] -> [N, out] */
    int weight_shape[2] = { config->in_dim, config->out_dim };
    linear->weights = tk_tensor_alloc(a, dtype, weight_shape, 2);

    int bias_shape[1] = { config->out_dim };
    linear->bias = tk_tensor_alloc(a, dtype, bias_shape, 1);

    /* placeholder outputs tensor — overwritten every forward pass */
    int out_shape[2] = { 1, config->out_dim };
    linear->outputs = tk_tensor_alloc(a, dtype, out_shape, 2);

    linear->delta = NULL;
    if (config->is_training) {
        linear->delta = tk_tensor_alloc(a, dtype, weight_shape, 2);
    }

    tk_ln_weights_init(linear);
    return linear;
}

Network* create_Network(struct tk_rt_ctx* ctx, struct LinearConfigList* cfgls) {
    int num_linears = cfgls->num_configs;
    Network* network = (Network*)arena_alloc(ctx->meta_arena, sizeof(Network));
    network->linear_count = num_linears;
    network->linears = (Linear*)arena_alloc(ctx->meta_arena, num_linears * sizeof(Linear));

    for (int i = 0; i < cfgls->num_configs; ++i)
        network->linears[i] = *create_Linear(ctx, &cfgls->configs[i], ctx->data_arena);

    return network;
}

/*
 * inputs:  [N, in_features]
 * labels:  [N, num_classes]  one-hot encoded
 * loss:    accumulates mean cross-entropy (output param)
 * returns: number of correct predictions in this batch
 */
int fc_forward(struct tk_rt_ctx* ctx,
               Network* network,
               struct tk_tensor* inputs,
               struct tk_tensor* labels,
               double* loss)
{
    int N = inputs->shape[0];
    struct tk_tensor* current = inputs;

    for (int i = 0; i < network->linear_count; ++i) {
        Linear* linear = &(network->linears[i]);

        /* weights: [in_dim, out_dim]  =>  current [N, in] x weights [in, out] -> dest [N, out] */
        int out_features = linear->weights->shape[1];
        int dest_shape[] = {N, out_features};
        struct tk_tensor* dest = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, dest_shape, 2);

        int err = tk_ops_gemm(current, linear->weights, dest);
        (void)err;

        /* Add bias: dest[n][j] += bias[j] */
        if (linear->has_bias && linear->bias) {
            double* bias_data = (double*)linear->bias->data;
            double* dest_data = (double*)dest->data;
            for (int n = 0; n < N; ++n)
                for (int j = 0; j < out_features; ++j)
                    dest_data[n * out_features + j] += bias_data[j];
        }

        /* ReLU on all layers except the last */
        if (i < network->linear_count - 1)
            tk_tensor_relu(dest);

        linear->outputs = dest;
        current = dest;
    }

    /* current: [N, num_classes] logits */
    int num_classes = current->shape[1];
    double batch_loss = 0.0;
    int correct = 0;

    for (int n = 0; n < N; ++n) {
        /* Zero-copy row view into sample n */
        double* row_ptr = (double*)current->data + (n * num_classes);
        int row_shape[] = { num_classes };
        struct tk_tensor row_view = {
            .data  = row_ptr,
            .shape = row_shape,
            .ndims = 1,
            .dtype = TK_F64,
        };

        softMax(&row_view);

        /* Find true class from one-hot label */
        int true_class = -1;
        double* label_row = (double*)labels->data + (n * num_classes);
        for (int c = 0; c < num_classes; ++c) {
            if (label_row[c] == 1.0) { true_class = c; break; }
        }

        batch_loss += crossEntropyLoss(true_class, &row_view);

        int pred = findMax((size_t)num_classes, (double*)row_view.data);
        if (pred == true_class) correct++;
    }

    *loss += batch_loss / N;
    return correct;
}
