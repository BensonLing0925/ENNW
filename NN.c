#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "matrixOps.h" 
#include "loadPic.h"
#include "convolution.h"
#include "output.h"

#define MAX_ITER 100 
#define LEARNING_RATE 0.01
#define TRAIN 1
#define TEST 2

double** oneHot(DataSet* dataset, int outSize); 
void freeSamples(Sample* samples, int num_sample, size_t firstD);
void freeNetwork(Network* network);
Neuron* create_Neuron(int weightCnt);
Layer* create_Layer(int neuronCnt, int num_inputs_for_neurons); 
Network* create_Network(int* layers, int num_layer );
int forward_propagation(Network* network, double* inputs, int ansIdx, double* loss);
void backward_propagation(Network *network, double *expected, double learning_rate, double* inputs);
void train(Network* network, double** samples, double** answers, int num_sample, int max_iter);
void test(Network* network, double** testcases, double** answers, int num_sample);
void testCase2();
DataSet* readData();

int main() {
	DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
	
	loadImgFile("C:\\Users\\Benson Ling\\Desktop\\CNN\\MNIST\\train-images-idx3-ubyte\\train-images-idx3-ubyte",
			    dataset, -1);

	printf("num_sample: %d, rows: %d, cols: %d\n", dataset->num_sample, dataset->rows, dataset->cols);

	loadImgLabel("C:\\Users\\Benson Ling\\Desktop\\CNN\\MNIST\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte",
				 dataset);

	int* idx_ans = (int*)malloc(dataset->num_sample * sizeof(int));
	for ( int i = 0 ; i < dataset->num_sample ; ++i ) {
		idx_ans[i] = dataset->samples[i].answer;	
	}		
	double** answers = oneHot(dataset, 10);

	int num_filter = 10;
	int picSize = 28;
	int kernSize = 3;
	int fmapSize = picSize - kernSize + 1;
	int pSize = 2;
	double kernels[num_filter][3][3] = {
			// Horizontal edge detection
			{
				{-1, -1, -1},
				{0, 0, 0},
				{1, 1, 1}
			},
			// Vertical edge detection
			{
				{-1, 0, 1},
				{-1, 0, 1},
				{-1, 0, 1}
			},
			// Diagonal edge detection (135)
			{
				{-1, -1, 0},
				{-1, 0, 1},
				{0, 1, 1}
			},
			// Diagonal edge detection (45)
			{
				{0, -1, -1},
				{1, 0, -1},
				{1, 1, 0}
			},
			// Sharpen Filter
			{
				{0, -1, 0},
				{-1, 5, -1},
				{0, -1, 0}
			},
			// Sobel Horizontal 
			{
				{-1, -2, -1},
				{0, 0, 0},
				{1, 2, 1}
			},
			// Sobel Vertical 
			{
				{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}
			},
			// Horizontal line detection
			{
				{1, 1, 1},
				{0, 0, 0},
				{-1, -1, -1}
			},
			// Vertical line detection
			{
				{1, 0, -1},
				{1, 0, -1},
				{1, 0, -1}
			},
			// 45 line detection
			{
				{-2, -1, 0},
				{-1, 0, 1},
				{0, 1, 2}
			}
	};

	int pooledSize = fmapSize / pSize;
	int layers[] = {pooledSize * pooledSize * num_filter, 100, 50, 10};
	srand(time(NULL));
	Network* network = create_Network(layers, 4);

	int train_correct = 0;
	int batch_num = 500;
	int epoch = 0;
	double lossArr[MAX_ITER] = {0.0};
	clock_t start = clock();	
	for ( int iter = 0 ; iter < MAX_ITER ; ++iter ) {
		int currentPos = (batch_num * epoch)%dataset->num_sample;
		double loss = 0.0;
		for ( int n = currentPos ; n < (currentPos + batch_num ) ; ++n ) {
		//printf("#Sample: %d\n", n);
			double*** featureMaps = alloc3DArr(num_filter, fmapSize, fmapSize);
			double*** pooledMaps = alloc3DArr(num_filter, pooledSize, pooledSize);
			double** flat_pics = alloc2DArr(num_filter, pooledSize * pooledSize); 
			for ( int i = 0 ; i < num_filter ; ++i ) {
				featureMaps[i] = convolute( dataset->samples[n%(dataset->num_sample)].picture, dataset->rows, dataset->cols,
										kernels[i], kernSize, kernSize );
				pooledMaps[i] = pooling(featureMaps[i], fmapSize, pSize);
				flat_pics[i] = flatten(pooledMaps[i], pooledSize, pooledSize);
				reLU_pic(flat_pics, num_filter, pooledSize * pooledSize);
			}
			double* inputs = flatten(flat_pics, pooledSize * pooledSize, num_filter);
			int train_predict = forward_propagation(network, inputs, idx_ans[n%(dataset->num_sample)], &loss);
			if ( train_predict == idx_ans[n%(dataset->num_sample)] ) {
				train_correct++;
			}		
			backward_propagation(network, answers[n%(dataset->num_sample)], LEARNING_RATE, inputs);
			free3DArr(featureMaps, num_filter, fmapSize);
			free3DArr(pooledMaps, num_filter, pooledSize);
			free2DArr(flat_pics, num_filter);
			free(inputs);
		}
		loss /= batch_num;
		lossArr[epoch] = loss;
		epoch++;
		printf("#Iteration: %d\n", iter);
		printf("correct prediction: %d\n", train_correct);
		printf("wrong prediction: %d\n", batch_num - train_correct);
		printf("percentage:%lf%\n", ((double)train_correct/(double)(batch_num)) * 100);
		printf("batch loss:%lf%\n", loss);
		train_correct = 0;
	}
	double end = clock();	
	/*
	printf("final result:\n");
	printf("correct prediction: %d\n", train_correct);
	printf("wrong prediction: %d\n", dataset->num_sample - train_correct);
	printf("percentage:%lf%\n", ((double)train_correct/(double)(dataset->num_sample)) * 100);
	*/
	printf("time spent: %lf\n", ((double)end - (double)start)/CLOCKS_PER_SEC);

	int xArr[epoch] = {0};
	for ( int i = 0 ; i < epoch ; ++i ) {
		xArr[i] = i+1;
	}		

	free2DArr(answers, dataset->num_sample);
	freeSamples(dataset->samples, dataset->num_sample, dataset->rows);
	free(dataset);
	free(idx_ans);

	PlotInfo* plot = (PlotInfo*)malloc(sizeof(PlotInfo));
	initPlot("C:\\Users\\Benson Ling\\Desktop\\CNN\\MNIST\\plot1.gp",
			 "Epoch", 
			 "Loss",
			 "Training", plot); 

	plot->xdata.type = INT_TYPE;
	plot->xdata.dataSize = epoch;
	plot->ydata.dataSize = epoch;
	plot->ydata.type = DOUBLE_TYPE;
	plot->xdata.intData = xArr;
	plot->ydata.doubleData = lossArr;

	writeData(*plot);
	freePlt(plot);

	
	printf("=========================test=========================\n");

	DataSet* t_dataset = (DataSet*)malloc(sizeof(DataSet));
	
	loadImgFile("C:\\Users\\Benson Ling\\Desktop\\CNN\\MNIST\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte",
			    t_dataset, -1);

	printf("num_sample: %d, rows: %d, cols: %d\n", t_dataset->num_sample, t_dataset->rows, t_dataset->cols);

	loadImgLabel("C:\\Users\\Benson Ling\\Desktop\\CNN\\MNIST\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte",
				 t_dataset);

	int correct = 0;
	double** t_answers = oneHot(t_dataset, 10);
	int* t_idx_ans = (int*)malloc(t_dataset->num_sample * sizeof(int));
	for ( int i = 0 ; i < t_dataset->num_sample ; ++i ) {
		t_idx_ans[i] = t_dataset->samples[i].answer;	
	}		

	double t_loss = 0.0;
	for ( int n = 0 ; n < t_dataset->num_sample ; ++n ) {
		double*** featureMaps = alloc3DArr(num_filter, fmapSize, fmapSize);
		double*** pooledMaps = alloc3DArr(num_filter, pooledSize, pooledSize);
		double** flat_pics = alloc2DArr(num_filter, pooledSize * pooledSize); 
		for ( int i = 0 ; i < num_filter ; ++i ) {
			featureMaps[i] = convolute( t_dataset->samples[n].picture, t_dataset->rows, t_dataset->cols,
										kernels[i], kernSize, kernSize );
			pooledMaps[i] = pooling(featureMaps[i], fmapSize, pSize);
			flat_pics[i] = flatten(pooledMaps[i], pooledSize, pooledSize);
			reLU_pic(flat_pics, num_filter, pooledSize * pooledSize);
		}
		double* inputs = flatten(flat_pics, pooledSize * pooledSize, num_filter);
		int predict = forward_propagation(network, inputs, t_idx_ans[n], &t_loss);
		if ( predict == t_idx_ans[n] ) {
			correct++;
		}		
		backward_propagation(network, t_answers[n], LEARNING_RATE, inputs);
		free3DArr(featureMaps, num_filter, fmapSize);
		free3DArr(pooledMaps, num_filter, pooledSize);
		free2DArr(flat_pics, num_filter);
		free(inputs);
	}

	t_loss /= t_dataset->num_sample;
	printf("final result:\n");
	printf("correct prediction: %d\n", correct);
	printf("wrong prediction: %d\n", t_dataset->num_sample - correct);
	printf("percentage:%lf%\n", ((double)correct/(double)(t_dataset->num_sample)) * 100);
	printf("average loss:%lf%\n", t_loss);

	free(t_idx_ans);
	free2DArr(t_answers, t_dataset->num_sample);
	freeSamples(t_dataset->samples, t_dataset->num_sample, t_dataset->rows);
	free(t_dataset);
	freeNetwork(network);
}

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

Neuron* create_Neuron(int weightCnt) {
	Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
	neuron->num_weight = weightCnt;
	neuron->weights = (double*)malloc(weightCnt * sizeof(double));
	for ( int i = 0 ; i < weightCnt ; ++i ) {
		neuron->weights[i] = ((double) rand() * 2 / RAND_MAX + (-1)) * 0.01;   // -0.01 to 0.01
	}		
	neuron->bias = 1;
	neuron->output = 0.0;
	neuron->delta = 0.0;
	return neuron;
}		

Layer* create_Layer(int neuronCnt, int num_inputs_for_neurons) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->num_neurons = neuronCnt;
	layer->outputs = (double*)malloc(neuronCnt * sizeof(double));
	layer->neurons = (Neuron*)malloc(neuronCnt * sizeof(Neuron));
	for ( int i = 0 ; i < neuronCnt ; ++i ) {
		layer->neurons[i] = *create_Neuron(num_inputs_for_neurons);		
	}		
	return layer;
}		

Network* create_Network(int* layers, int num_layer ) {
	Network* network = (Network*)malloc(sizeof(Network));
	network->layer_count = num_layer;
	network->layers = (Layer*)malloc(num_layer * sizeof(Layer));

	for ( int i = 0 ; i < num_layer ; ++i ) {
		// layers[0] is the input of the neural network	
		int num_inputs = (i == 0) ? layers[i] : layers[i-1];	
		network->layers[i] = *create_Layer(layers[i], num_inputs);
	}		

	return network;
}		

int forward_propagation(Network* network, double* inputs, int ansIdx, double* loss) {
	
	for ( int i = 1; i < network->layer_count ; ++i ) {
		Layer* layer = &(network->layers[i]);
		for ( int j = 0 ; j < layer->num_neurons ; ++j ) {
			Neuron* neuron = &(layer->neurons[j]);
			double sum = neuron->bias;
			for ( int k = 0 ; k < neuron->num_weight ; ++k ) {
				sum += neuron->weights[k] * (i == 1 ? inputs[k] : network->layers[i-1].outputs[k]); 
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
	softMax(output_layer->outputs, 10);
	*loss += crossEntropyLoss(ansIdx, output_layer->outputs);

	/*
	printf("probability: \n");
	for ( int i = 0 ; i < output_layer->num_neurons ; ++i ) {
		printf("%d: %lf  ", i, output_layer->outputs[i]);
	}
	*/
	int answer = findMax(output_layer->num_neurons, output_layer->outputs);
	//printf("\nprediction: %d\n", answer);
	return answer;
}		

void backward_propagation(Network *network, double *expected, double learning_rate, double* inputs) {
    // Calculate delta for output layer

	Layer* output_layer = &network->layers[network->layer_count-1];
	for ( int j = 0 ; j < output_layer->num_neurons ; ++j ) {
		Neuron* neuron  = &output_layer->neurons[j];
		double output = output_layer->outputs[j]; 
		double error = (output - expected[j]);
		neuron->delta = error;
	}

    for (int i = network->layer_count - 2; i >= 1; i--) {
        Layer *layer = &network->layers[i];
        Layer *next_layer = &network->layers[i+1];
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron *neuron = &layer->neurons[j];
            double error = 0.0;
            for (int k = 0; k < next_layer->num_neurons; k++) {
                error += next_layer->neurons[k].weights[j] * next_layer->neurons[k].delta;
            }
            neuron->delta = error * reLU_diff(neuron->output);
        }
    }

    // Update weights and biases
    for (int i = 1; i < network->layer_count; i++) {
        Layer *layer = &network->layers[i];
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron *neuron = &layer->neurons[j];
            for (int k = 0; k < neuron->num_weight; k++) {
                double input = (i == 1) ? inputs[k] : network->layers[i-1].outputs[k];
                neuron->weights[k] -= learning_rate * neuron->delta * input;
            }
            neuron->bias -= learning_rate * neuron->delta;
        }
    }
}

void freeSamples(Sample* samples, int num_sample, size_t firstD) {
	for ( int i = 0 ; i < num_sample ; ++i ) {
		free2DArr(samples[i].picture, firstD);
	}		
	free(samples);
}		
	
DataSet* readData(int options) {
		
	FILE* fptr;
	if ( options == TRAIN ) {
		fptr = fopen("C:\\Users\\Benson Ling\\Desktop\\CNN\\dataset.txt", "r");
	}
	else if ( options == TEST ) {
		fptr = fopen("C:\\Users\\Benson Ling\\Desktop\\CNN\\testcase.txt", "r");
	}		
	int num_sample, firstD, secondD;
	int temp;
	fscanf(fptr, "%d\n%d%d", &num_sample, &firstD, &secondD);
	DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
	Sample* samples = (Sample*)malloc(num_sample * sizeof(Sample));	
	dataset->samples = samples;
	dataset->num_sample = num_sample;
	for ( int i = 0 ; i < num_sample ; ++i ) {
		fscanf(fptr, "%d", &samples[i].answer);
		samples[i].picture = alloc2DArr(firstD, secondD);
		for ( int j = 0 ; j < firstD ; ++j ) {
			for ( int k = 0 ; k < secondD ; ++k ) {	
				fscanf(fptr, "%d,", &temp);
				samples[i].picture[j][k] = (double)temp;
			}
		}		
	}
	//printOut(samples, num_sample);
	//freeSamples(samples, num_sample, firstD);
	fclose(fptr);
	return dataset;
}		

double** oneHot(DataSet* dataset, int outSize) {
	double** answers = alloc2DArr(dataset->num_sample, outSize);
	for ( int i = 0 ; i < dataset->num_sample ; ++i ) {
		answers[i][dataset->samples[i].answer] = 1.0;	
	}
	return answers;
}		
