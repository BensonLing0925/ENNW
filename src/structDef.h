#ifndef STRUCTDEF
#define STRUCTDEF

typedef double** Double2D;

#include <stdint.h>

#define MAX_CHILD 2  // bitwise Trie, only 0 and 1
#define MAX_HEIGHT 16  // JPEG at most height of 16, each height represent one bit
                       
typedef struct Neuron {
	int num_weight;
	double* weights;
	double bias;
	double output;
	double delta;
} Neuron;	

typedef struct Layer {
	int num_neurons;
    int input_dim;
    int has_bias;
	Neuron* neurons;
	double* outputs;
} Layer;		

typedef struct Network {
	int layer_count;
	Layer* layers; // to point to individual Layer
    uint32_t network_type;
    uint32_t input_size;
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

typedef enum {
	MAX_POOL,
	AVG_POOL
} PoolingType;	

typedef struct Conv2D {
    int in_channels;
	int num_filter;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int pooling_h;
    int pooling_w;
    int has_bias;
    double* biases;
	Double2D* filters;
	PoolingType pType;
} Conv2D;		

typedef struct LayerMeta {
    uint32_t layer_type;
    uint32_t layer_index;
    union {
        struct Conv2D* conv;
        // struct Layer* layer;
        struct Network* network;
    } u_layer;
    uint32_t dtype;
} LayerMeta;

typedef struct Model {
    int has_conv;
    int num_total_layers;
    int init_input_c;
    int init_input_h;
    int init_input_w;
    struct LayerMeta* layers_meta;
} Model;

typedef struct {
	char* filePath;	
	char* xlabel;
	char* ylabel;
	char* title;
	DataPointer xdata;
	DataPointer ydata;
} PlotInfo;	

typedef enum {
	LUMINANCE = 0,	
	CHROMINANCE = 1
} Component;

typedef enum {
	DC0 = 0,	
	DC1 = 1,
	AC0 = 2,
	AC1 = 3
} TableClass;

struct Node {
    struct Node* children[MAX_CHILD];
    char bitVal[MAX_HEIGHT];
    int val;
    int bitLen;
};    

typedef struct Node Node;

typedef struct HufTable {
    int bitLen;
    int val;
    char* bitStr;
} HufTable;    

typedef struct {
	Component component;	
	uint8_t precision;
	int dqt_len;
	union {
		uint8_t* table8;
		uint16_t* table16;
	};
} DQT_struct;

typedef struct {
	TableClass tClass;
	uint8_t codeLen[16];
	int dht_len;
	uint8_t* table8;
    Node* root;
} DHT_struct;	

typedef struct {
    uint8_t id;
    uint8_t vertFactor;
    uint8_t horzFactor;
    Component component;
} CompInfo;    

typedef struct {
    uint8_t precision;
    uint8_t num_component;
    CompInfo* compInfo;         // num_component * sizeof(compInfo)
} SOF_struct; 

typedef struct {
	int pic_width;
	int pic_height;
	uint8_t*** picture;
	int num_DQT;
	int num_DHT;
    SOF_struct SOF_info;
	DQT_struct* DQTs;
	DHT_struct* DHTs;
} Img;	
#endif
