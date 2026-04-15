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
#include "modules/transformer/tf_block.h"
#include "nn_utils/nn_utils.h"
#include "structDef.h"
#include "../config/config.h"
#include "../weightio/weightio.h"

/* ------------------------------------------------------------------ */
/* Build model from config                                              */
/* Returns 0 on success; model->layers_meta must be pre-allocated.     */
/* ------------------------------------------------------------------ */

static int build_model(struct tk_rt_ctx* ctx,
                       const struct Config* c,
                       int picSize,
                       struct Model* model,
                       struct tk_conv2d** out_conv,
                       struct tk_pooling** out_pooling,
                       struct TransformerBlock** out_tf,
                       struct Network** out_network,
                       uint64_t* out_input_size) {

    int num_filter = c->num_filter;
    int kernSize   = c->kernel_size;
    int pSize      = c->pool_size;

    int fmapSize   = picSize - kernSize + 1;
    int pooledSize = fmapSize / pSize;

    if (pooledSize <= 0) {
        fprintf(stderr, "Invalid architecture: pooledSize=%d (picSize=%d, kernel=%d, pool=%d)\n",
                pooledSize, picSize, kernSize, pSize);
        return -1;
    }

    int tf_hidden = pooledSize * pooledSize;
    int tf_seq    = num_filter;
    int tf_heads  = c->tf_n_heads;

    if (tf_hidden % tf_heads != 0) {
        fprintf(stderr, "tf_n_heads=%d does not divide hidden_dim=%d\n", tf_heads, tf_hidden);
        return -1;
    }

    uint64_t input_size = (uint64_t)tf_seq * tf_hidden;
    *out_input_size = input_size;

    /* ---- FC layer sizes ---- */
    static const int default_fc[] = { 100, 50, 10 };
    const int* fc_sizes;
    int fc_count;
    if (c->fc_num_layers > 0 && c->fc_layers) {
        fc_sizes = c->fc_layers;
        fc_count = c->fc_num_layers;
    } else {
        fc_sizes = default_fc;
        fc_count = 3;
    }

    /* ---- FC network ---- */
    struct LinearConfig* lc_arr =
        arena_alloc(ctx->meta_arena, sizeof(struct LinearConfig) * fc_count);
    uint64_t prev_dim = input_size;
    for (int i = 0; i < fc_count; ++i) {
        lc_arr[i].in_dim      = (int)prev_dim;
        lc_arr[i].out_dim     = fc_sizes[i];
        lc_arr[i].has_bias    = 1;
        lc_arr[i].is_training = (c->mode == MODE_TRAIN) ? 1 : 0;
        lc_arr[i].dtype       = TK_F64;
        prev_dim = (uint64_t)fc_sizes[i];
    }
    struct LinearConfigList cfgls = { .num_configs = fc_count, .configs = lc_arr };
    struct Network* network = create_Network(ctx, &cfgls);
    network->input_size   = (uint32_t)input_size;
    network->network_type = FC_CHAIN;
    *out_network = network;

    /* ---- Model metadata ---- */
    /* Layers: 0=CONV2D, 1=TF, 2=FC */
    model->has_conv          = 1;
    model->num_total_layers  = 3;
    model->init_input_c      = 1;
    model->init_input_h      = picSize;
    model->init_input_w      = picSize;
    model->layers_meta = arena_alloc(ctx->meta_arena,
                                     sizeof(struct LayerMeta) * 3);

    model->layers_meta[0].layer_type  = LAYER_CONV2D;
    model->layers_meta[0].layer_index = 0;
    model->layers_meta[0].dtype       = LAYER_DTYPE_F64;

    model->layers_meta[1].layer_type  = LAYER_TF;
    model->layers_meta[1].layer_index = 1;
    model->layers_meta[1].dtype       = LAYER_DTYPE_F64;

    model->layers_meta[2].layer_type  = LAYER_FC;
    model->layers_meta[2].layer_index = 2;
    model->layers_meta[2].dtype       = LAYER_DTYPE_F64;

    /* ---- Conv layer ---- */
    struct tk_conv2d* conv = tk_conv2D_create(ctx);
    tk_conv2d_init(conv, TK_CONV_SQR(num_filter, kernSize, 1, 0));

    int sample_shape[3]   = { 1, picSize, picSize };
    int sample_strides[3] = { picSize * picSize, picSize, 1 };
    struct tk_tensor sample_meta = {
        .dtype   = TK_F64,
        .data    = NULL,
        .ndims   = 3,
        .shape   = sample_shape,
        .strides = sample_strides,
    };
    tk_conv2d_setup(conv, &sample_meta);
    tk_conv2d_alloc(ctx, conv);
    *out_conv = conv;

    /* ---- Pooling layer ---- */
    struct tk_pooling* pooling = tk_pooling_create(ctx);
    /* stride = pool_size for non-overlapping max-pool */
    tk_pooling_init(pooling, TK_PL_SQR(MAX_POOL, pSize, pSize, 0));
    *out_pooling = pooling;

    /* ---- Transformer block ---- */
    struct TransformerBlock* tf_block = tf_block_create(ctx);
    tf_block_alloc(ctx, tf_block, tf_seq, tf_hidden, tf_heads);
    *out_tf = tf_block;

    /* ---- Wire layer meta ---- */
    model->layers_meta[0].u_layer.conv     = conv;
    model->layers_meta[1].u_layer.tf_block = tf_block;
    model->layers_meta[2].u_layer.network  = network;

    printf("Architecture: conv(%d filters, %dx%d) -> pool(%dx%d) "
           "-> tf(seq=%d, hidden=%d, heads=%d) -> fc[",
           num_filter, kernSize, kernSize, pSize, pSize,
           tf_seq, tf_hidden, tf_heads);
    for (int i = 0; i < fc_count; ++i) {
        printf("%d", fc_sizes[i]);
        if (i < fc_count - 1) printf(",");
    }
    printf("]\n");
    printf("input_size = %" PRIu64 "\n", input_size);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-sample forward: conv -> pool -> relu -> tf                      */
/* Returns pointer to the flat output row in flat_buf for sample n.   */
/* ------------------------------------------------------------------ */

static void forward_sample(struct tk_rt_ctx* ctx,
                            struct tk_conv2d* conv,
                            struct tk_pooling* pooling,
                            struct TransformerBlock* tf_block,
                            const uint8_t* raw_sample,
                            int H, int W,
                            int tf_seq, int tf_hidden,
                            double* flat_buf_row,
                            uint64_t input_size) {
    size_t sample_ws = ctx->ws->cur_offset;

    /* U8 -> F64 [1, H, W] */
    int s_shape[3] = { 1, H, W };
    struct tk_tensor* s_f64 = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                                  TK_F64, s_shape, 3);
    double* sf = (double*)s_f64->data;
    for (int k = 0; k < H * W; ++k)
        sf[k] = raw_sample[k] / 255.0;

    /* Conv forward */
    struct tk_tensor* filtered = tk_conv_forward(ctx, conv,
        &(struct Dataset){ .samples = s_f64, .rows = H, .cols = W,
                           .num_samples = 1, .labels = NULL });

    /* Pooling */
    tk_pooling_setup(pooling, filtered);
    struct tk_tensor* pooled = tk_pooling_forward(ctx, pooling, filtered);

    /* ReLU in-place */
    tk_tensor_relu(pooled);

    /* Transformer view [tf_seq, tf_hidden] */
    int tf_shape[2]   = { tf_seq, tf_hidden };
    int tf_strides[2] = { tf_hidden, 1 };
    struct tk_tensor tf_input = {
        .dtype   = TK_F64,
        .data    = pooled->data,
        .ndims   = 2,
        .shape   = tf_shape,
        .strides = tf_strides,
    };
    tf_block_forward(ctx, tf_block, &tf_input);

    /* Copy flat output */
    double* src = (double*)pooled->data;
    for (uint64_t k = 0; k < input_size; ++k)
        flat_buf_row[k] = src[k];

    ctx->ws->cur_offset = sample_ws;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("usage: nn.exe \"path_to_json_config_file\"\n");
        return 0;
    }

    /* ---- Arenas ---- */
    struct arena root_arena;
    arena_init(&root_arena);

    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&root_arena);

    struct Dataset* dataset = tk_dataset_create(ctx);

    struct arena misc_arena;
    arena_init(&misc_arena);

    /* ---- Config ---- */
    struct Config c;
    config_init(&c);
    if (load_json(argv[1], &misc_arena, &c) != 0)
        return -1;

    if (c.mode == MODE_NONE) {
        fprintf(stderr, "Config error: 'mode' must be TRAIN or TEST\n");
        return -1;
    }

    if (c.mode == MODE_TEST && !c.weights_path) {
        fprintf(stderr, "Config error: TEST mode requires 'weights_path'\n");
        return -1;
    }

    /* ---- Load dataset ---- */
    loadImgFile(ctx, dataset, c.imgPath, -1);
    loadImgLabel(ctx, dataset, c.imgLabelPath);

    /* ---- Seed ---- */
    if (c.seed == -1)
        srand((unsigned int)time(NULL));
    else
        srand((unsigned int)c.seed);

    printf("num_sample: %d, rows: %d, cols: %d\n",
           dataset->num_samples, dataset->rows, dataset->cols);

    int picSize = dataset->rows;   /* assume square images */

    /* ---- Build model ---- */
    struct Model model;
    struct tk_conv2d*       conv    = NULL;
    struct tk_pooling*      pooling = NULL;
    struct TransformerBlock* tf_block = NULL;
    struct Network*         network = NULL;
    uint64_t input_size = 0;

    if (build_model(ctx, &c, picSize, &model,
                    &conv, &pooling, &tf_block, &network, &input_size) != 0)
        return -1;

    ctx->model = &model;

    int tf_seq    = c.num_filter;
    int tf_hidden = (int)input_size / tf_seq;

    /* ---- One-hot labels ---- */
    int num_classes = network->linears[network->linear_count - 1].num_neurons;
    int onehot_shape[] = { dataset->num_samples, num_classes };
    struct tk_tensor* onehot = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                                   TK_F64, onehot_shape, 2);
    if (tk_ops_onehot(dataset->labels, onehot) != 0)
        return -1;

    size_t ws_base = ctx->ws->cur_offset;

    int    N = dataset->num_samples;
    int    H = dataset->rows;
    int    W = dataset->cols;
    uint8_t* raw = (uint8_t*)dataset->samples->data;

    ctx->rt_type = RT_DRYRUN;
    int flat_shape[2] = { N, (int)input_size };
    struct tk_tensor* flat_buf = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                                         TK_F64, flat_shape, 2);
    for (int n = 0; n < 1; ++n) {
        forward_sample(ctx, conv, pooling, tf_block,
                       raw + (size_t)n * H * W,
                       H, W, tf_seq, tf_hidden,
                       (double*)flat_buf->data + (size_t)n * input_size,
                       input_size);
    }

    size_t weight_size = arena_get_total_used(ctx->data_arena);
    size_t peak_ws = ctx->ws->peak_offset;

    printf("------------------------------------\n");
    printf("ENNW Memory Report (Final Audit):\n");
    printf("  Persistent Weights: %.2f MB (%lu bytes)\n", 
            (double)weight_size / (1024.0 * 1024.0), (unsigned long)weight_size);
    printf("  Peak Workspace:     %.2f MB (%lu bytes)\n", 
            (double)peak_ws / (1024.0 * 1024.0), (unsigned long)peak_ws);
    printf("------------------------------------\n");

    /* ================================================================
     * TEST (inference) mode
     * ================================================================ */
    ctx->rt_type = RT_INFERENCE;
    if (c.mode == MODE_TEST) {
        printf("\n--- Inference mode ---\n");

        /* Load weights into the pre-built model */
        if (model_load(c.weights_path, ctx, &model) != 0) {
            fprintf(stderr, "Failed to load weights from '%s'\n", c.weights_path);
            return -1;
        }

        ctx->ws->cur_offset = ws_base;

        int flat_shape[2] = { N, (int)input_size };
        struct tk_tensor* flat_buf = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                                         TK_F64, flat_shape, 2);

        for (int n = 0; n < N; ++n) {
            forward_sample(ctx, conv, pooling, tf_block,
                           raw + (size_t)n * H * W,
                           H, W, tf_seq, tf_hidden,
                           (double*)flat_buf->data + (size_t)n * input_size,
                           input_size);
        }

        double loss = 0.0;
        int correct = fc_forward(ctx, network, flat_buf, onehot, &loss);

        printf("\n=== Inference Result ===\n");
        printf("correct : %d / %d  (%.2f%%)\n",
               correct, N, 100.0 * correct / N);
        printf("loss    : %.6f\n", loss);

        arena_destroy(ctx->data_arena);
        arena_destroy(ctx->meta_arena);
        arena_destroy(&root_arena);
        arena_destroy(&misc_arena);
        return 0;
    }

    /* ================================================================
     * TRAIN mode
     * ================================================================ */
    printf("\n--- Training mode (%u epochs) ---\n", c.max_iter);

    int    correct_predict = 0;
    double loss            = 0.0;
    unsigned int max_epoch = c.max_iter;

    clock_t start = clock();
    for (uint32_t iter = 0; iter < max_epoch; ++iter) {

        ctx->ws->cur_offset = ws_base;
        correct_predict = 0;
        loss            = 0.0;

        int flat_shape[2] = { N, (int)input_size };
        struct tk_tensor* flat_buf = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena,
                                                         TK_F64, flat_shape, 2);

        for (int n = 0; n < N; ++n) {
            forward_sample(ctx, conv, pooling, tf_block,
                           raw + (size_t)n * H * W,
                           H, W, tf_seq, tf_hidden,
                           (double*)flat_buf->data + (size_t)n * input_size,
                           input_size);
        }

        correct_predict = fc_forward(ctx, network, flat_buf, onehot, &loss);

        printf("[epoch %u] correct: %d / %d  loss: %.6f\n",
               iter + 1, correct_predict, N, loss);
    }
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("\n=== Final (epoch %u) ===\n", max_epoch);
    printf("correct prediction : %d\n", correct_predict);
    printf("wrong prediction   : %d\n", N - correct_predict);
    printf("total average loss : %f\n", loss);
    printf("time spent         : %.3f s\n", elapsed);

    /* Save weights if requested */
    if (c.save_path) {
        if (save_weight(c.save_path, &model) != 0)
            fprintf(stderr, "[WARNING] Failed to save weights to '%s'\n", c.save_path);
    }

    arena_destroy(ctx->data_arena);
    arena_destroy(ctx->meta_arena);
    arena_destroy(&root_arena);
    arena_destroy(&misc_arena);
    return 0;
}
