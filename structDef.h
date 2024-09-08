#ifndef STRUCTDEF
#define STRUCTDEF

typedef struct {
	int num_weight;
	double* weights;
	double bias;
	double output;
	double delta;
} Neuron;	

typedef struct {
	int num_neurons;
	Neuron* neurons;
	double* outputs;
} Layer;		

typedef struct {
	int layer_count;
	Layer* layers;
} Network;		

typedef struct {
	int answer;
	double** picture;	
} Sample; 

typedef struct {
	Sample* samples;
	int num_sample;
	int rows;
	int cols;
} DataSet;	

typedef enum {
	INT_TYPE,
	FLOAT_TYPE,
	DOUBLE_TYPE
} DataType;

typedef struct {
	DataType type;
	union {
		int* intData;
		float* floatData;
		double* doubleData;
	};		
	size_t dataSize;
} DataPointer; 	

typedef struct {
	char* filePath;	
	char* xlabel;
	char* ylabel;
	char* title;
	DataPointer xdata;
	DataPointer ydata;
} PlotInfo;	

#endif
