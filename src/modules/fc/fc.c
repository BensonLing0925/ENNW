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


int tk_ln_cfg_dim_set(struct LinearConfigList* context,
                          int* linears_num_neurons,
                          uint64_t input_size) {

    struct LinearConfig* configs = context->configs;
    for ( int i = 0 ; i < context->num_configs ; ++i ) {
        if (i == 0) {
            configs[i].in_dim = input_size;
        }
        else {
            configs[i].in_dim = configs[i-1].out_dim;
        }
        configs[i].out_dim = linears_num_neurons[i];
    }
    return 0;

}

int tk_ln_cfg_dtype_set(struct LinearConfigList* context, enum tk_dtype* dtypes) {
    
    struct LinearConfig* configs = context->configs;
    for ( int i = 0 ; i < context->num_configs ; ++i ) {
        configs[i].dtype = dtypes[i];
    }
    return 0;

}

// for training
int tk_ln_weights_init(struct Linear* linear) {

    struct tk_tensor* weights = linear->weights; 
    struct tk_tensor* bias = linear->bias;
    struct tk_tensor* delta = linear->delta;

    double* weight_data = (double*) weights->data;
    double* bias_data = (double*) bias->data;
    double* delta_data = (double*) delta->data;

    // potential for opm optimization
    for ( int row = 0 ; row < weights->shape[0] ; ++row ) {
        for ( int col = 0 ; col < weights->shape[1] ; ++col ) {
            weight_data[row * weights->shape[0] + col] =  ((double) rand() * 2 / RAND_MAX + (-1)) * 0.01;
            bias_data[row * bias->shape[0] + col] = 1;
        }
    }

    if (linear->is_training) {
        for ( int row = 0 ; row < weights->shape[0] ; ++row ) {
            for ( int col = 0 ; col < weights->shape[1] ; ++col ) {
                delta_data[row * delta->shape[0] + col] = 0.0;
            }
        }
    }
    return 0;
}

Linear* create_Linear(struct tk_rt_ctx* ctx, struct LinearConfig* config, struct arena* a) {

    struct Linear* linear = arena_alloc(ctx->meta_arena, sizeof(struct Linear));
	linear->num_neurons = config->out_dim;
    enum tk_dtype dtype = config->dtype;

    int shape[2] = { config->out_dim, config->in_dim };
    struct tk_tensor* weights = tk_tensor_alloc(a, dtype, shape, 2);
    struct tk_tensor* outputs = tk_tensor_alloc(a, dtype, shape, 2);
    struct tk_tensor* delta = NULL;

    if (config->is_training) {
        delta = tk_tensor_alloc(a, dtype, shape, 2);
    }

    tk_ln_weights_init(linear);

    linear->weights = weights;
    linear->outputs = outputs;
    linear->delta = delta;

    free(shape);

    linear->input_dim = config->in_dim;
    linear->has_bias = 1;     // default have bias
	return linear;
}		


// context is properly initialized
Network* create_Network(struct tk_rt_ctx* ctx, struct LinearConfigList* cfgls) {
    int num_linears = cfgls->num_configs;
	Network* network = (Network*)arena_alloc(ctx->meta_arena, sizeof(Network));
	network->linear_count = num_linears;
	network->linears = (Linear*)arena_alloc(ctx->meta_arena, num_linears * sizeof(Linear));

	for ( int i = 0 ; i < cfgls->num_configs ; ++i ) {
		// linearss[0] is the input of the neural network	
		network->linears[i] = *create_Linear(ctx, &cfgls->configs[i], ctx->data_arena);
	}		

	return network;
}		

// inputs:  [N, flatten_size]
// labels:  [N, num_classes] one-hot
// loss:    output param, accumulates mean cross-entropy
// returns: correct count in this batch
int fc_forward(struct tk_rt_ctx* ctx,
               Network* network,
               struct tk_tensor* inputs,
               struct tk_tensor* labels,
               double* loss)
{
    int N = inputs->shape[0];
    struct tk_tensor* current = inputs;  // 追蹤當前層的輸入

    for (int i = 0; i < network->linear_count; ++i) {
        Linear* linear = &(network->linears[i]);

        // weights: [in_features, out_features]
        // current: [N, in_features]
        // dest:    [N, out_features]
        int out_features = linear->weights->shape[1];
        int dest_shape[] = {N, out_features};
        struct tk_tensor* dest = tk_ws_tensor_alloc(ctx, TK_F64, dest_shape, 2);

        // current [N, in] x weights [in, out] -> dest [N, out]
        int err = tk_ops_gemm(current, linear->weights, dest);
        // TODO: handle err

        // ReLU on all layers except last
        if (i < network->linear_count - 1) {
            tk_tensor_relu(dest);
        }

        linear->outputs = dest;  // 每層都存，backward 會用到
        current = dest;
    }

    // current: [N, num_classes] logits
    int num_classes = current->shape[1];
    double batch_loss = 0.0;
    int correct = 0;

    for (int n = 0; n < N; ++n) {
        // zero-copy row view，指向第 n 個 sample
        struct tk_tensor row_view = {
            .data  = &current->data[n * num_classes],
            .shape = (int[]){num_classes},
            .ndims = 1,
            .dtype = TK_F64,
        };

        softMax(&row_view);

        // 從 one-hot 找 true class index
        int true_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            if (((double*)labels->data)[n * num_classes + c] == 1.0) {
                true_class = c;
                break;
            }
        }

        batch_loss += crossEntropyLoss(true_class, &row_view);

        int pred = findMax(num_classes, &row_view);
        if (pred == true_class)
            correct++;
    }

    *loss += batch_loss / N;  // mean loss over batch
    return correct;
}

/*
void backward_propagation(Network *network, double *expected, double learning_rate,
                          double* inputs, uint64_t input_size) {
    Linear* output_linear = &network->linears[network->linear_count - 1];

    for (int j = 0; j < output_linear->num_neurons; ++j) {
        Neuron* neuron = &output_linear->neurons[j];
        double output = output_linear->outputs[j];
        neuron->delta = (output - expected[j]);
    }

    for (int i = network->linear_count - 2; i >= 0; --i) {
        Linear *linear = &network->linears[i];
        Linear *next_linear = &network->linears[i + 1];

        for (int j = 0; j < linear->num_neurons; ++j) {
            double error = 0.0;
            for (int k = 0; k < next_linear->num_neurons; ++k) {
                error += next_linear->neurons[k].weights[j] * next_linear->neurons[k].delta;
            }
            linear->neurons[j].delta = error * reLU_diff(linear->neurons[j].output);
        }
    }

    for (int i = 0; i < network->linear_count; ++i) {
        Linear *linears = &network->linears[i];

        for (int j = 0; j < linears->num_neurons; ++j) {
            Neuron *neuron = &linears->neurons[j];

            if (i == 0) {
                if ((uint64_t)neuron->num_weight != input_size) {
                    printf("[ERROR] input_size mismatch in linears 0: %d vs %" PRIu64 "\n",
                           neuron->num_weight, (unsigned long long)input_size);
                    return;
                }
                for (int k = 0; k < neuron->num_weight; ++k) {
                    neuron->weights[k] -= learning_rate * neuron->delta * inputs[k];
                }
            } else {
                for (int k = 0; k < neuron->num_weight; ++k) {
                    neuron->weights[k] -= learning_rate * neuron->delta * network->linearss[i - 1].outputs[k];
                }
            }

            if (linears->has_bias) {
                neuron->bias -= learning_rate * neuron->delta;
            }
        }
    }
}
*/
