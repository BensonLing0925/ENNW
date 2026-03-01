#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <inttypes.h>
#include "fc.h"
#include "structDef.h"

void freeNetwork(Network* network) {
	for ( int i = 0; i < network->layer_count ; ++i ) {
		Layer* layer = &(network->layers[i]);
		for ( int j = 0 ; j < layer->num_neurons ; ++j ) {
			Neuron* neuron = &(layer->neurons[j]);
			free(neuron->weights);
		}
		free(layer->neurons);
		free(layer->outputs);
	}	
	free(network->layers);
	free(network);
}		

Neuron* create_Neuron(int weightCnt, struct arena* a) {
	Neuron* neuron = (Neuron*)arena_alloc(a, sizeof(Neuron));
	neuron->num_weight = weightCnt;
	neuron->weights = (double*)arena_alloc(a, weightCnt * sizeof(double));
	for ( int i = 0 ; i < weightCnt ; ++i ) {
		neuron->weights[i] = ((double) rand() * 2 / RAND_MAX + (-1)) * 0.01;   // -0.01 to 0.01
	}		
	neuron->bias = 1;
	neuron->output = 0.0;
	neuron->delta = 0.0;
	return neuron;
}		

Layer* create_Layer(int neuronCnt, int num_inputs_for_neurons, struct arena* a) {
	Layer* layer = (Layer*)arena_alloc(a, sizeof(Layer));
	layer->num_neurons = neuronCnt;
	layer->outputs = (double*)arena_alloc(a, neuronCnt * sizeof(double));
	layer->neurons = (Neuron*)arena_alloc(a, neuronCnt * sizeof(Neuron));
    layer->input_dim = num_inputs_for_neurons;
    layer->has_bias = 1;     // default have bias
	for ( int i = 0 ; i < neuronCnt ; ++i ) {
		layer->neurons[i] = *create_Neuron(num_inputs_for_neurons, a);		
	}		
	return layer;
}		

Network* create_Network(int* layers, uint64_t input_size, int num_layer, struct arena* a) {
	Network* network = (Network*)arena_alloc(a, sizeof(Network));
	network->layer_count = num_layer;
	network->layers = (Layer*)arena_alloc(a, num_layer * sizeof(Layer));

	for ( int i = 0 ; i < num_layer ; ++i ) {
		// layers[0] is the input of the neural network	
		int num_inputs = (i == 0) ? input_size : layers[i-1];	
		network->layers[i] = *create_Layer(layers[i], num_inputs, a);
	}		

	return network;
}		

int forward_propagation(Network* network, double* inputs, uint64_t input_size, int ansIdx, double* loss) {
	
	for ( int i = 0; i < network->layer_count ; ++i ) {
		Layer* layer = &(network->layers[i]);
		for ( int j = 0 ; j < layer->num_neurons ; ++j ) {
			Neuron* neuron = &(layer->neurons[j]);
			double sum = neuron->bias;
            if (i == 0) {
                for ( uint64_t input_i = 0 ; input_i < input_size ; ++input_i ) {
                    sum += neuron->weights[input_i] * inputs[input_i];
                }
            }
            else {
                for ( int k = 0 ; k < neuron->num_weight ; ++k ) {
                    sum += neuron->weights[k] * network->layers[i-1].outputs[k]; 
                }		
            }
			// neuron->output = reLU(sum);
			if ( i < network->layer_count - 1 ) {
				neuron->output = reLU(sum);
			}		
			else {
				neuron->output = sum;
			}
			layer->outputs[j] = neuron->output;
		}		
	}		
	Layer* output_layer = &(network->layers[network->layer_count-1]);
	softMax(output_layer->outputs, output_layer->num_neurons);
	*loss += crossEntropyLoss(ansIdx, output_layer->outputs);

	int answer = findMax(output_layer->num_neurons, output_layer->outputs);
	return answer;
}		

void backward_propagation(Network *network, double *expected, double learning_rate,
                          double* inputs, uint64_t input_size) {
    Layer* output_layer = &network->layers[network->layer_count - 1];

    /* output delta: softmax + cross entropy */
    for (int j = 0; j < output_layer->num_neurons; ++j) {
        Neuron* neuron = &output_layer->neurons[j];
        double output = output_layer->outputs[j];
        neuron->delta = (output - expected[j]);
    }

    /* hidden deltas */
    for (int i = network->layer_count - 2; i >= 0; --i) {
        Layer *layer = &network->layers[i];
        Layer *next_layer = &network->layers[i + 1];

        for (int j = 0; j < layer->num_neurons; ++j) {
            double error = 0.0;
            for (int k = 0; k < next_layer->num_neurons; ++k) {
                error += next_layer->neurons[k].weights[j] * next_layer->neurons[k].delta;
            }
            layer->neurons[j].delta = error * reLU_diff(layer->neurons[j].output);
        }
    }

    /* update */
    for (int i = 0; i < network->layer_count; ++i) {
        Layer *layer = &network->layers[i];

        for (int j = 0; j < layer->num_neurons; ++j) {
            Neuron *neuron = &layer->neurons[j];

            if (i == 0) {
                if ((uint64_t)neuron->num_weight != input_size) {
                    printf("[ERROR] input_size mismatch in layer 0: %d vs %" PRIu64 "\n",
                           neuron->num_weight, (unsigned long long)input_size);
                    return;
                }
                for (int k = 0; k < neuron->num_weight; ++k) {
                    neuron->weights[k] -= learning_rate * neuron->delta * inputs[k];
                }
            } else {
                for (int k = 0; k < neuron->num_weight; ++k) {
                    neuron->weights[k] -= learning_rate * neuron->delta * network->layers[i - 1].outputs[k];
                }
            }

            if (layer->has_bias) {
                neuron->bias -= learning_rate * neuron->delta;
            }
        }
    }
}
