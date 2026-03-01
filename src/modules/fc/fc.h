#ifndef FC_H
#define FC_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <inttypes.h>
#include "../../nn_utils/nn_utils.h" 
#include "structDef.h"

void freeNetwork(Network* network);
Neuron* create_Neuron(int weightCnt, struct arena* a);
Layer* create_Layer(int neuronCnt, int num_inputs_for_neurons, struct arena* a);
Network* create_Network(int* layers, uint64_t input_size, int num_layer, struct arena* a);
int forward_propagation(Network* network, double* inputs, uint64_t input_size, int ansIdx, double* loss);
void backward_propagation(Network *network, double *expected, double learning_rate,
                          double* inputs, uint64_t input_size);

#endif
