#ifndef FC_H
#define FC_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <inttypes.h>
#include "../../nn_utils/nn_utils.h" 
#include "../../ops/tensor.h"
#include "../../runtime/rt_context.h"

#define TK_LN_CFG_SET(_in, _out, _bias, _is_training, _dtype) (struct LinearConfig){                           \
    .in_dim = _in,               \
    .out_dim = _out,             \
    .has_bias = _bias,           \
    .is_training = _is_training, \
    .dtype = _dtype              \
}

typedef struct Linear {

    int input_dim;
	int num_neurons;    // weight's row
	int num_weight;     // weight's col

    // both training and inference
    struct tk_tensor* weights;
    int has_bias;
    struct tk_tensor* bias;

    // training
    int is_training;
    struct tk_tensor* delta;

    struct tk_tensor* outputs;

} Linear;		

struct LinearConfigList {
    int num_configs;    // == num_linears
    struct LinearConfig* configs;
};

struct LinearConfig {
    int in_dim;
    int out_dim;
    int is_training;
    int has_bias;
    enum tk_dtype dtype;
};

typedef struct Network {
	int linear_count;
	Linear* linears; // to point to individual Layer
    uint32_t network_type;
    uint32_t input_size;
    int is_learning;
} Network;		

struct Network* tk_fc_create(struct tk_rt_ctx* ctx);
struct LinearConfig* tk_ln_cfg_create(int num_configs, struct arena* a);
struct LinearConfigList* tk_ln_cfgls_create(int num_linears, struct arena* a);
int tk_ln_weights_init(struct Linear* linear);
Linear* create_Linear(struct tk_rt_ctx* ctx, struct LinearConfig* config, struct arena* a);
Network* create_Network(struct tk_rt_ctx* ctx, struct LinearConfigList* cfgls);
int fc_forward(struct tk_rt_ctx* ctx,
               Network* network,
               struct tk_tensor* inputs,
               struct tk_tensor* labels,
               double* loss);

#endif
