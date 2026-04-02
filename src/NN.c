#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include "loadPic.h"
#include "modules/fc/fc.h"
#include "modules/conv/conv.h"
#include "modules/pooling/pooling.h"
#include "nn_utils/nn_utils.h"
#include "structDef.h"
#include "../config/config.h"
#include "../weightio/weightio.h"
#include "output.h"

#define TRAIN 1
#define TEST 2

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("usage: nn.exe \"path_to_json_config_file\"");
        return 0;
    }

    struct arena root_arena;
    arena_init(&root_arena);

    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&root_arena);
    ctx->rt_type = RT_DRYRUN;

	struct Dataset* dataset = tk_dataset_create(ctx);
    
    struct arena misc_arena;
    arena_init(&misc_arena);
	
    struct Config c;
    config_init(&c);
    if (load_json(argv[1], &misc_arena, &c) != 0)
        return -1;

    loadImgFile(ctx, dataset, c.imgPath, -1);
    loadImgLabel(ctx, dataset, c.imgLabelPath);

    if (c.seed == -1)
	    srand(time(NULL));
    else {
        srand(c.seed);
    }

	printf("num_sample: %d, rows: %d, cols: %d\n", dataset->num_samples, dataset->rows, dataset->cols);

    // model_load("C:\\Users\\Benson Ling\\Desktop\\CNN\\weights", &t_a)

	int num_filter = 10;
	int picSize = 28;
	int kernSize = 3;
	int fmapSize = picSize - kernSize + 1;
	int pSize = 2;

	int pooledSize = fmapSize / pSize;
    uint64_t input_size = pooledSize * pooledSize * num_filter;


    struct LinearConfig my_layers[] = {
        TK_LN_CFG_SET(input_size, 100, 1, 1, TK_F64),
        TK_LN_CFG_SET(100, 50, 1, 1, TK_F64),
        TK_LN_CFG_SET(50 , 10, 1, 1, TK_F64)
    };

    struct LinearConfigList my_model = {
        .num_configs = 3,
        .configs = my_layers
    };

    // create Network
	struct Network* network = create_Network(ctx, &my_model);
    network->input_size = input_size;
    network->network_type = FC_CHAIN;

    // declare entire architecture of model
    struct Model model;
    model.has_conv = 1;
    model.num_total_layers = 2;    // 1 conv + 1 fc (Network)
    model.init_input_c = 1;
    model.init_input_h = picSize;
    model.init_input_w = picSize;
    model.layers_meta = (struct LayerMeta*)arena_alloc(&root_arena, sizeof(struct LayerMeta) * model.num_total_layers);    // create LayerMetas

    // set each layer type in model (Modulize in the future)
    model.layers_meta[0].layer_type = LAYER_CONV2D;
    model.layers_meta[0].layer_index = 0;
    model.layers_meta[0].dtype = LAYER_DTYPE_F64;

    model.layers_meta[1].layer_type = LAYER_FC; // network
    model.layers_meta[1].layer_index = 1;
    model.layers_meta[1].dtype = LAYER_DTYPE_F64;

    ctx->model = &model;

    // create a 2D convolution layer
    // tk_conv2d* conv = (tk_conv2d*)malloc(sizeof(tk_conv2d));
    struct tk_conv2d* conv = tk_conv2D_create(ctx);
    tk_conv2d_init(conv, TK_CONV_SQR(10, 3, 1, 0));
    // dataset->samples will be struct tk_tensor
    tk_conv2d_setup(conv, dataset->samples);
    tk_conv2d_alloc(ctx, conv); // allocate space for weights
    // then model_load will call tk_conv2d_load_weights
    struct tk_pooling* pooling = tk_pooling_create(ctx);
    tk_pooling_init(pooling, TK_PL_SQR(MAX_POOL, 1, 2, 0));

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

	int correct_predict = 0;
    unsigned int max_epoch = c.max_iter;
    double loss = 0.0;

    int num_classes = 10;
    int onehot_shape[] = {dataset->num_samples, num_classes};
    struct tk_tensor* onehot = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_F64, onehot_shape, 2);
    int err = tk_ops_onehot(dataset->labels, onehot);
    if (err) {
        return err;
    }

	clock_t start = clock();	
	for ( uint32_t iter = 0 ; iter < max_epoch ; ++iter ) {
        struct tk_tensor* filtered_pics = tk_conv_forward(ctx, conv, dataset);
        // inference
        tk_pooling_setup(pooling, filtered_pics);
        struct tk_tensor* pooled_pics = tk_pooling_forward(ctx, pooling, filtered_pics);
        tk_tensor_relu(pooled_pics);

        // flatten before FC
        // 2. Flatten (將 3D [C, H, W] 轉為 2D [1, C*H*W])
        int N = dataset->num_samples;
        int flatten_size = conv->num_filter * pooling->pooled_h * pooling->pooled_w;
        int input_shape[] = {N, flatten_size};
        
        struct tk_tensor flattened_pics;
        flattened_pics.shape = input_shape;
        flattened_pics.ndims = 2;
        flattened_pics.data = pooled_pics->data; 
        flattened_pics.dtype = TK_F64;

        correct_predict = fc_forward(ctx, network, &flattened_pics, onehot, &loss);
	}
	double end = clock();	

    printf("Batched process\n");
    printf("correct prediction: %d\n", correct_predict);
    printf("wrong prediction: %d\n", dataset->num_samples - correct_predict);
    printf("total average loss:%lf\n", loss);
	printf("time spent: %lf\n", ((double)end - (double)start)/CLOCKS_PER_SEC);
    arena_destroy(ctx->data_arena);
    arena_destroy(ctx->meta_arena);
    arena_destroy(&root_arena);
    arena_destroy(&misc_arena);
    return 0;
}
