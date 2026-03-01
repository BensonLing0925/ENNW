#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include "modules/fc/fc.h"
#include "modules/conv/conv.h"
#include "nn_utils/nn_utils.h"
#include "../config/config.h"
#include "../weightio/weightio.h"
#include "output.h"

#define TRAIN 1
#define TEST 2

double** oneHot(DataSet* dataset, int outSize); 
void freeSamples(Sample* samples, int num_sample, size_t firstD);
// void freeNetwork(Network* network);
// Neuron* create_Neuron(int weightCnt, struct arena* a);
// Layer* create_Layer(int neuronCnt, int num_inputs_for_neurons, struct arena* a); 
// Network* create_Network(int* layers, uint64_t input_size, int num_layer, struct arena* a );
// int forward_propagation(Network* network, double* inputs, uint64_t input_size, int ansIdx, double* loss);
// void backward_propagation(Network *network, double *expected, double learning_rate, double* inputs, uint64_t input_size);
void testCase2();
DataSet* readData();

int layer_meta_set() {


}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("usage: nn.exe \"path_to_json_config_file\"");
        return 0;
    }

	DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
    
    struct arena a;
    arena_init(&a);
	
    struct Config c;
    config_init(&c);
    if (load_json(argv[1], &a, &c) != 0)
        return -1;

    loadImgFile(c.imgPath, dataset, -1);
    loadImgLabel(c.imgLabelPath, dataset);

    if (c.seed == -1)
	    srand(time(NULL));
    else
        srand(c.seed);

	printf("num_sample: %d, rows: %d, cols: %d\n", dataset->num_sample, dataset->rows, dataset->cols);

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

	int pooledSize = fmapSize / pSize;
    uint64_t input_size = pooledSize * pooledSize * num_filter;

    // set network type
	int layers[] = {100, 50, 10};
	Network* network = create_Network(layers, input_size, 3, &a);
    network->input_size = input_size;
    network->network_type = FC_CHAIN;

    // declare entire architecture of model
    struct Model model;
    model.has_conv = 1;
    model.num_total_layers = 2;    // 1 conv + 1 fc (Network)
    model.init_input_c = 1;
    model.init_input_h = picSize;
    model.init_input_w = picSize;
    model.layers_meta = (struct LayerMeta*)arena_alloc(&a, sizeof(struct LayerMeta) * model.num_total_layers);    // create LayerMetas

    // set each layer type in model (Modulize in the future)
    model.layers_meta[0].layer_type = LAYER_CONV2D;
    model.layers_meta[0].layer_index = 0;
    model.layers_meta[0].dtype = LAYER_DTYPE_F64;

    model.layers_meta[1].layer_type = LAYER_FC; // network
    model.layers_meta[1].layer_index = 1;
    model.layers_meta[1].dtype = LAYER_DTYPE_F64;

    // layer index to track
    int cur_fc_layer_index = 0;

    // create a 2D convolution layer
    Conv2D* conv = (Conv2D*)malloc(sizeof(Conv2D));
    initConv2D(conv, 10, 3, 2, MAX_POOL);
    allocConv(conv, picSize, picSize);  // square for this case
    manualKernal(conv); // load filter (or kernel) into convolution layer

    // set LayerMetas
    for ( int i = 0 ; i < model.num_total_layers ; ++i ) {
        struct LayerMeta* current_layer_meta = &model.layers_meta[i];
        // provide layer information
        if (current_layer_meta->layer_type == LAYER_CONV2D) {
            model.layers_meta[i].u_layer.conv = conv;
        }
        if (current_layer_meta->layer_type == LAYER_FC) {
            model.layers_meta[i].u_layer.network = network;
        }
    }

	int train_correct = 0;
	int batch_num = 500;
	int epoch = 0;
    unsigned int max_iter = c.max_iter;
    double lr = c.lr;
	double lossArr[max_iter];

    /* runtime size context */
    int current_pic_h = picSize;
    int current_pic_w = picSize;
    int pooled_img_size = fmapSize / pSize;

	clock_t start = clock();	
	for ( uint32_t iter = 0 ; iter < max_iter ; ++iter ) {
		int currentPos = (batch_num * epoch)%dataset->num_sample;
		double loss = 0.0;

		for ( int n = currentPos ; n < (currentPos + batch_num ) ; ++n ) {
			double** flat_pics = alloc2DArr(num_filter, pooledSize * pooledSize); 
            /* runtime tensor */
            Double2D* filtered_pics = malloc(sizeof(Double2D) * conv->num_filter);
            Double2D* pooled_pics = malloc(sizeof(Double2D) * conv->num_filter);

			for ( int i = 0 ; i < conv->num_filter ; ++i ) {
				filtered_pics[i] = convolute( dataset->samples[n%(dataset->num_sample)].picture, current_pic_h, current_pic_w,
										conv->filters[i], conv->kernel_h, conv->kernel_w);
				pooled_pics[i] = pooling(filtered_pics[i], fmapSize, pSize);
				flat_pics[i] = flatten(pooled_pics[i], pooledSize, pooledSize);
				reLU_pic(flat_pics, num_filter, pooledSize * pooledSize);
			}
			double* inputs = flatten(flat_pics, pooledSize * pooledSize, num_filter);
            // send into FC layers
			int train_predict = forward_propagation(network, inputs, input_size, idx_ans[n%(dataset->num_sample)], &loss);
			if ( train_predict == idx_ans[n%(dataset->num_sample)] ) {
				train_correct++;
			}		
			backward_propagation(network, answers[n%(dataset->num_sample)], lr, inputs, input_size);
			free2DArr(flat_pics, num_filter);
            // freeConvPics(conv); // only free pooled pics and filtered pics
            free3DArr(filtered_pics, conv->num_filter, fmapSize);
            free3DArr(pooled_pics, conv->num_filter, pooledSize);
			free(inputs);
		}
		loss /= batch_num;
		lossArr[epoch] = loss;
		epoch++;
		printf("#Iteration: %d\n", iter);
		printf("correct prediction: %d\n", train_correct);
		printf("wrong prediction: %d\n", batch_num - train_correct);
		printf("percentage:%lf\n", ((double)train_correct/(double)(batch_num)) * 100);
		printf("batch loss:%lf\n", loss);
		train_correct = 0;
	}
	double end = clock();	
    // return 0;
	/*
	printf("final result:\n");
	printf("correct prediction: %d\n", train_correct);
	printf("wrong prediction: %d\n", dataset->num_sample - train_correct);
	printf("percentage:%lf%\n", ((double)train_correct/(double)(dataset->num_sample)) * 100);
	*/
	printf("time spent: %lf\n", ((double)end - (double)start)/CLOCKS_PER_SEC);

	int xArr[epoch];
	for ( int i = 0 ; i < epoch ; ++i ) {
		xArr[i] = i+1;
	}		

	free2DArr(answers, dataset->num_sample);
	freeSamples(dataset->samples, dataset->num_sample, dataset->rows);
	free(dataset);
	free(idx_ans);

    if (c.save_path)
        save_weight(c.save_path, &model);

    arena_destroy(&a);

    /*
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
    */
	
	printf("=========================test=========================\n");

	DataSet* t_dataset = (DataSet*)malloc(sizeof(DataSet));
    struct arena t_a;	
    arena_init(&t_a); 

    struct Model* t_model = model_load("C:\\Users\\Benson Ling\\Desktop\\CNN\\weights", &t_a);

    save_weight("C:\\Users\\Benson Ling\\Desktop\\CNN\\weight_files\\after_weight", t_model);

	loadImgFile("C:\\Users\\Benson Ling\\Desktop\\CNN\\src\\MNIST\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte",
			    t_dataset, -1);

	printf("num_sample: %d, rows: %d, cols: %d\n", t_dataset->num_sample, t_dataset->rows, t_dataset->cols);

	loadImgLabel("C:\\Users\\Benson Ling\\Desktop\\CNN\\src\\MNIST\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte",
				 t_dataset);

	int correct = 0;
	double** t_answers = oneHot(t_dataset, 10);
	int* t_idx_ans = (int*)malloc(t_dataset->num_sample * sizeof(int));
	for ( int i = 0 ; i < t_dataset->num_sample ; ++i ) {
		t_idx_ans[i] = t_dataset->samples[i].answer;	
	}		

    struct Conv2D* t_conv = t_model->layers_meta[0].u_layer.conv;
    struct Network* t_network = t_model->layers_meta[1].u_layer.network;

	double t_loss = 0.0;

    double** t_flat_pics = alloc2DArr(num_filter, pooledSize * pooledSize); 
    /* runtime tensor */
    Double2D* t_filtered_pics = malloc(sizeof(Double2D) * t_conv->num_filter);
    Double2D* t_pooled_pics = malloc(sizeof(Double2D) * t_conv->num_filter);

	for ( int n = 0 ; n < t_dataset->num_sample ; ++n ) {
        for ( int i = 0 ; i < t_conv->num_filter ; ++i ) {
            t_filtered_pics[i] = convolute( t_dataset->samples[n%(t_dataset->num_sample)].picture, current_pic_h, current_pic_w,
                                    t_conv->filters[i], t_conv->kernel_h, t_conv->kernel_w);
            t_pooled_pics[i] = pooling(t_filtered_pics[i], fmapSize, pSize);
            t_flat_pics[i] = flatten(t_pooled_pics[i], pooledSize, pooledSize);
            reLU_pic(t_flat_pics, num_filter, pooledSize * pooledSize);
        }
        double* inputs = flatten(t_flat_pics, pooledSize * pooledSize, num_filter);
        // send into FC layers
        int predict = forward_propagation(t_network, inputs, input_size, t_idx_ans[n%(t_dataset->num_sample)], &t_loss);
        if ( predict == t_idx_ans[n%(t_dataset->num_sample)] ) {
            correct++;
        }		
        free(inputs);
    }

    free2DArr(t_flat_pics, num_filter);
    free3DArr(t_filtered_pics, t_conv->num_filter, fmapSize);
    free3DArr(t_pooled_pics, t_conv->num_filter, pooledSize);
            /*
				t_conv->filtered_pics[i] = convolute( t_dataset->samples[n].picture, t_conv->input_rows, t_conv->input_cols, t_conv->filters[i], t_conv->filter_size, t_conv->filter_size);
				t_conv->pooled_pics[i] = pooling(t_conv->filtered_pics[i], fmapSize, pSize);
			flat_pics[i] = flatten(t_conv->pooled_pics[i], pooledSize, pooledSize);
			reLU_pic(flat_pics, num_filter, pooledSize * pooledSize);
		}
		double* inputs = flatten(flat_pics, pooledSize * pooledSize, num_filter);
		int predict = forward_propagation(network, inputs, input_size, t_idx_ans[n], &t_loss);
		if ( predict == t_idx_ans[n] ) {
			correct++;
		}		
		freeConv(t_conv);
		free2DArr(flat_pics, num_filter);
		free(inputs);
        */

	t_loss /= t_dataset->num_sample;
	printf("final result:\n");
	printf("correct prediction: %d\n", correct);
	printf("wrong prediction: %d\n", t_dataset->num_sample - correct);
	printf("percentage:%lf\n", ((double)correct/(double)(t_dataset->num_sample)) * 100);
	printf("average loss:%lf\n", t_loss);

	free(t_idx_ans);
	free2DArr(t_answers, t_dataset->num_sample);
	freeSamples(t_dataset->samples, t_dataset->num_sample, t_dataset->rows);
	free(t_dataset);

    arena_destroy(&t_a);
	// freeNetwork(network);
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
