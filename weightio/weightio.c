#include "weightio.h"
#include "../mem/arena.h"
#include "../src/structDef.h"

int header_create(struct Binary_Header* bh, struct Model* model) {

    memcpy(bh->magic, "ENNW", 4);
    bh->ver = 1;
    bh->endian = 1;    // little endian
    bh->dtype = 2;                        // 1 => float32, 2 => float64(double)
    bh->model_type = 2;    // 1 => FN, 2 => CNN
    bh->layer_count = model->num_total_layers;
    if (model->has_conv) {
        bh->input_h = model->init_input_h;
        bh->input_w = model->init_input_w;
        bh->input_c = model->init_input_c;   // (gray scale for now)
    }
    else {
        bh->input_h = 0;
        bh->input_w = 0;
        bh->input_c = 0;
    }
    memset(bh->reserved, 0, sizeof(bh->reserved));
}

uint64_t fc_param_calculate(const struct Layer* layer) {
    uint64_t weights_sum = 0;
    if (!layer) {
        printf("[WARNING] Layer is NULL\n");
        return 0;
    }
    int num_neurons = layer->num_neurons;
    if (num_neurons < 0) {
        printf("[WARNING] num_neurons is a negative number: %d\n", num_neurons);
        return 0;
    }
    for ( int j = 0 ; j < num_neurons ; ++j ) {
        weights_sum += (uint64_t) layer->neurons[j].num_weight + 1;    // bias
    } 
    return weights_sum;
}

uint64_t network_fc_param_calculate(const struct Network* net) {
    if (!net) {
        printf("[WARNING] Network is NULL\n");
        return 0;
    }
    uint64_t total_weights = 0;
    for ( int i = 0 ; i < net->layer_count; ++i )
        total_weights += fc_param_calculate(&net->layers[i]);
    return total_weights;
}

uint64_t conv2d_param_calculate(const struct Conv2D* conv) {
    if (!conv) {
        printf("[WARNING] Conv2D is NULL\n");
        return 0;
    }
    uint64_t param_count = (uint64_t) (conv->num_filter) * (uint64_t) (conv->kernel_h) * (uint64_t) (conv->kernel_w) * (uint64_t) (conv->in_channels);
    return param_count;
}

uint64_t layer_param_calculate(struct LayerMeta* layer) {
    uint64_t num_layer_param = 0;
    switch (layer->layer_type) {
        case LAYER_FC:
            num_layer_param += network_fc_param_calculate(layer->u_layer.network); 
            break;
        case LAYER_CONV2D:
            num_layer_param += conv2d_param_calculate(layer->u_layer.conv);
            break;
        case LAYER_POOL:
            break;
        default:
            RT_FAIL(RT_EINVAL, "Unknown Layer Type: %u", layer->layer_type);
    }
    return num_layer_param;
}

// caller should properly allocate fc_meta
// fc_meta is an array
int net_layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* net_lm) {
    if (blm->layer_type != net_lm->layer_type) {
        RT_FAIL(RT_EINVAL, 
                "Incompatible meta conversion between %u and %u", 
                blm->layer_type, net_lm->layer_type);
    }
    struct Binary_Net_Layer_Meta* meta = &(blm->u_layer.net_layer_meta);
    struct Network* net = net_lm->u_layer.network;
    meta->network_type = FC_CHAIN;
    meta->fc_layer_count = net->layer_count;
    meta->input_size = net->input_size;

    return 0;
}

// fc_meta is an element
int fc_layer_meta_create(struct Layer* net_layer, struct Binary_FC_Layer_Meta* fc_meta) {
    fc_meta->num_neurons = net_layer->num_neurons;
    fc_meta->input_dim = net_layer->input_dim;
    fc_meta->has_bias = net_layer->has_bias;
    fc_meta->reserved = 0u;
    return 0;
}

int conv2d_layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* lm) {

    if (blm->layer_type != lm->layer_type) {
        RT_FAIL(RT_EINVAL, 
                "Incompatible meta conversion between %u and %u", 
                blm->layer_type, lm->layer_type);
    }
    struct Binary_Conv2D_Layer_Meta* meta = &(blm->u_layer.conv2d_layer_meta);
    struct Conv2D* lm_conv = lm->u_layer.conv;
    meta->num_filter = lm_conv->num_filter;
    meta->in_channels = lm_conv->in_channels;
    meta->kernel_h = lm_conv->kernel_h; //assume square for now
    meta->kernel_w = lm_conv->kernel_w;
    meta->stride_h = lm_conv->stride_h;
    meta->stride_w = lm_conv->stride_w;
    meta->padding_h = lm_conv->padding_h;
    meta->padding_w = lm_conv->padding_w;
    meta->has_bias = lm_conv->has_bias;
    if (lm_conv->pType == MAX_POOL)
        meta->pooling_type = 1;
    else if (lm_conv->pType == AVG_POOL)
        meta->pooling_type = 2;
    else
        meta->pooling_type = 0;
    // assume pooling square for now
    meta->pooling_h = lm_conv->pooling_h;
    meta->pooling_w = lm_conv->pooling_w;
    memset(meta->reserved, 0, sizeof(meta->reserved));
    return 0;
}

int pool_layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* lm) {
    if (blm->layer_type != lm->layer_type) {
        RT_FAIL(RT_EINVAL, 
                "Incompatible meta conversion between %u and %u", 
                blm->layer_type, lm->layer_type);
    }
    struct Binary_Pool_Layer_Meta* meta = &(blm->u_layer.pool_layer_meta);
    struct tk_pooling* lm_pl = lm->u_layer.pooling;
    meta->kernel_h = lm_pl->kernel_h; //assume square for now
    meta->kernel_w = lm_pl->kernel_w;
    meta->stride_h = lm_pl->stride_h;
    meta->stride_w = lm_pl->stride_w;
    meta->padding_h = lm_pl->padding_h;
    meta->padding_w = lm_pl->padding_w;
    if (lm_pl->pType == MAX_POOL)
        meta->pooling_type = 1;
    else if (lm_pl->pType == AVG_POOL)
        meta->pooling_type = 2;
    else
        meta->pooling_type = 0;
    memset(meta->reserved, 0, sizeof(meta->reserved));
    return 0;
}

int pool_meta_write(FILE* fptr,
                    const struct Binary_Pool_Layer_Meta* pm) {
    if (!fptr || !pm) return -1;
    if (write_u32(fptr, pm->stride_h)     < 0) goto write_fail;
    if (write_u32(fptr, pm->stride_w)     < 0) goto write_fail;
    if (write_u32(fptr, pm->kernel_h)       < 0) goto write_fail;
    if (write_u32(fptr, pm->kernel_w)       < 0) goto write_fail;
    if (write_u32(fptr, pm->padding_h)    < 0) goto write_fail;
    if (write_u32(fptr, pm->padding_w)    < 0) goto write_fail;
    if (write_u32(fptr, pm->pooling_type) < 0) goto write_fail;
    for (int i = 0; i < 4; i++)
        if (write_u32(fptr, pm->reserved[i]) < 0) goto write_fail;
    return 0;
write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int pool_meta_load(FILE* fptr, struct Binary_Pool_Layer_Meta* pm) {
    if (!fptr || !pm) return -1;
    if (read_u32(fptr, &pm->stride_h)     < 0) goto read_fail;
    if (read_u32(fptr, &pm->stride_w)     < 0) goto read_fail;
    if (read_u32(fptr, &pm->kernel_h)       < 0) goto read_fail;
    if (read_u32(fptr, &pm->kernel_w)       < 0) goto read_fail;
    if (read_u32(fptr, &pm->padding_h)    < 0) goto read_fail;
    if (read_u32(fptr, &pm->padding_w)    < 0) goto read_fail;
    if (read_u32(fptr, &pm->pooling_type) < 0) goto read_fail;
    fseek(fptr, 4 * sizeof(uint32_t), SEEK_CUR);  // reserved
    return 0;
read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Read fail");
}

int layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* lm) {
    blm->layer_type = lm->layer_type;
    blm->layer_index = lm->layer_index;
    blm->param_count = layer_param_calculate(lm);
    switch (lm->dtype) {
        case LAYER_DTYPE_F32:
            blm->payload_bytes = blm->param_count * sizeof(float);
            break;
        case LAYER_DTYPE_F64:
            blm->payload_bytes = blm->param_count * sizeof(double);
            break;
        default:
            RT_FAIL(RT_EINVAL, "Unknown layer meta data type: %u", lm->dtype);
    }

    switch (blm->layer_type) {
        case LAYER_FC:
            if (net_layer_meta_create(blm, lm) < 0) {
                return -1;
            } 
            break;
        case LAYER_CONV2D:
            if (conv2d_layer_meta_create(blm, lm) < 0) {
                return -1;
            }
            break;
        case LAYER_POOL:
            if (pool_layer_meta_create(blm, lm) < 0) {
                return -1;
            }
            break;
        default:
            RT_FAIL(RT_EINVAL, "Unknown binary layer type: %u", blm->layer_type);
    }

    return 0;
}

static int write_bytes(FILE *fp, const void *ptr, size_t n) {
    return fwrite(ptr, 1, n, fp) == n ? 0 : -1;
}

static int write_u32(FILE *fp, uint32_t v) {
    return write_bytes(fp, &v, sizeof(v));
}

static int write_u64(FILE *fp, uint64_t v) {
    return write_bytes(fp, &v, sizeof(v));
}

static int write_f64(FILE* fp, double v) {
    return write_bytes(fp, &v, sizeof(v));
}

static int read_bytes(FILE* fp, void* ptr, size_t n) {
    return (fread(ptr, 1, n, fp) == n) ? 0 : -1;
}

static int read_u32(FILE* fp, uint32_t* v) {
    return read_bytes(fp, v, sizeof(*v));
}

static int read_u64(FILE* fp, uint64_t* v) {
    return read_bytes(fp, v, sizeof(*v));
}

static int read_f64(FILE* fp, double* v) {
    return read_bytes(fp, v, sizeof(*v));
}

int header_write(FILE* fptr, const struct Binary_Header* bh) {
    if (!fptr || !bh) return -1;

    if (write_bytes(fptr, bh->magic, sizeof(bh->magic)) != 0) goto write_fail;

    if (write_u32(fptr, bh->ver)        < 0) goto write_fail;
    if (write_u32(fptr, bh->endian)     < 0) goto write_fail;
    if (write_u32(fptr, bh->dtype)      < 0) goto write_fail;
    if (write_u32(fptr, bh->model_type) < 0) goto write_fail;

    if (write_u32(fptr, bh->layer_count) < 0) goto write_fail;
    if (write_u32(fptr, bh->input_h)     < 0) goto write_fail;
    if (write_u32(fptr, bh->input_w)     < 0) goto write_fail;
    if (write_u32(fptr, bh->input_c)     < 0) goto write_fail;

    if (write_bytes(fptr, bh->reserved, sizeof(bh->reserved)) != 0) goto write_fail;

    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int fc_meta_write(FILE* fptr, const struct Binary_FC_Layer_Meta* fc_blm) {
    if (!fptr || !fc_blm) return -1;

    if (write_u32(fptr, fc_blm->num_neurons) < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->input_dim)   < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->has_bias)    < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->reserved)    < 0) goto write_fail;

    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int net_meta_write(FILE* fptr,
                   const struct Binary_Net_Layer_Meta* net_blm,
                   const struct Network* net)
{
    if (!fptr || !net_blm || !net) return -1;
    if (net_blm->fc_layer_count != net->layer_count) {
        RT_FAIL(RT_EINVAL, 
                "Layer count mismatch between %u and %u", 
                net_blm->fc_layer_count, net->layer_count);
    }

    if (write_u32(fptr, net_blm->network_type)   < 0) goto write_fail;
    if (write_u32(fptr, net_blm->fc_layer_count) < 0) goto write_fail;
    if (write_u32(fptr, net_blm->input_size) < 0) goto write_fail;
    if (write_bytes(fptr, net_blm->reserved, sizeof(net_blm->reserved)) < 0) goto write_fail;

    for (uint32_t i = 0; i < net_blm->fc_layer_count; ++i) {
        struct Binary_FC_Layer_Meta fc_meta;

        if (fc_layer_meta_create(&net->layers[i], &fc_meta) < 0) return -1;
        if (fc_meta_write(fptr, &fc_meta) < 0) return -1;
    }

    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int conv2d_meta_write(FILE *fptr, const struct Binary_Conv2D_Layer_Meta *m) {
    if (!fptr || !m)
        RT_FAIL(RT_EINVAL, "Assignment pointer is NULL");

    if (write_u32(fptr, m->num_filter)    < 0) goto write_fail;
    if (write_u32(fptr, m->in_channels)   < 0) goto write_fail;
    if (write_u32(fptr, m->kernel_h)      < 0) goto write_fail;
    if (write_u32(fptr, m->kernel_w)      < 0) goto write_fail;
    if (write_u32(fptr, m->stride_h)      < 0) goto write_fail;
    if (write_u32(fptr, m->stride_w)      < 0) goto write_fail;
    if (write_u32(fptr, m->padding_h)     < 0) goto write_fail;
    if (write_u32(fptr, m->padding_w)     < 0) goto write_fail;
    if (write_u32(fptr, m->has_bias)      < 0) goto write_fail;
    if (write_u32(fptr, m->pooling_type)  < 0) goto write_fail;
    if (write_u32(fptr, m->pooling_h)     < 0) goto write_fail;
    if (write_u32(fptr, m->pooling_w)     < 0) goto write_fail;

    for (int i = 0; i < 4; ++i) {
        if (write_u32(fptr, m->reserved[i]) < 0) goto write_fail;
    }

    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int layer_meta_write(FILE* fptr, 
                     const struct Binary_Layer_Meta* blm, 
                     struct Model* model,
                     int index) {
    if (!fptr || !blm) return -1;

    if (write_u32(fptr, blm->layer_type)    < 0) goto write_fail;
    if (write_u32(fptr, blm->layer_index)   < 0) goto write_fail;
    if (write_u64(fptr, blm->param_count)   < 0) goto write_fail;
    if (write_u64(fptr, blm->payload_bytes) < 0) goto write_fail;

    switch (blm->layer_type) {
        case LAYER_FC:
            if (net_meta_write(fptr, &blm->u_layer.net_layer_meta, model->layers_meta[index].u_layer.network) < 0) goto write_fail;
            break;

        case LAYER_CONV2D:
            if (conv2d_meta_write(fptr, &blm->u_layer.conv2d_layer_meta) < 0) goto write_fail;
            break;

        default:
            RT_FAIL(RT_EINVAL, "Unknown binary layer type: %u", blm->layer_type);
    }

    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int fc_payload_write(FILE* fptr, const struct Layer* layer) {
    for ( int i = 0 ; i < layer->num_neurons ; ++i ) {
        const struct Neuron* neuron = &layer->neurons[i];
        for ( int j = 0 ; j < neuron->num_weight ; ++j ) {
            if (write_f64(fptr, neuron->weights[j]) < 0) goto write_fail; 
        }
        if (layer->has_bias)
            if (write_f64(fptr, neuron->bias) < 0) goto write_fail;
    }
    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int conv2d_payload_write(FILE* fptr, const struct Conv2D* conv) {
    for ( int i = 0 ; i < conv->num_filter ; ++i ) {
        const Double2D current_filter = conv->filters[i];
        // assuming square for now
        int filter_height = conv->kernel_h;
        int filter_width = conv->kernel_w;
        for ( int filter_h = 0 ; filter_h < filter_height ; ++filter_h ) {
            for ( int filter_w = 0 ; filter_w < filter_width ; ++filter_w ) {
                if (write_f64(fptr, current_filter[filter_h][filter_w]) < 0) goto write_fail;
            }
        }
        if (conv->has_bias) {
            if (write_f64(fptr, conv->biases[i]) < 0) goto write_fail;
        }
    }
    return 0;
write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Write fail");
}

int payload_write(FILE* fptr, struct LayerMeta* lm) {
    long start = ftell(fptr);
    switch (lm->layer_type) {
        case LAYER_FC:
            for ( int i = 0 ; i < lm->u_layer.network->layer_count ; ++i ) {
                if (fc_payload_write(fptr, &lm->u_layer.network->layers[i]) < 0) return -1;
            }
            break;
        case LAYER_CONV2D:
            if (conv2d_payload_write(fptr, lm->u_layer.conv) != 0) return -1;
        default:
            RT_FAIL(RT_EINVAL, "Unknown layer type: %u", lm->layer_type);
    }
    long end = ftell(fptr);
    fprintf(stderr,
        "[DEBUG] payload size: wrote=%ld\n",
        end - start);
    return 0;
}

int save_weight(const char* path, struct Model* model) {
    FILE* fptr = fopen(path, "wb");
    if (!fptr) {
        RT_FAIL(RT_EIO, "fopen error");
    }

    struct Binary_Header bh;
    if (header_create(&bh, model) < 0) goto save_fail;
    if (header_write(fptr, &bh) < 0) goto save_fail;
    for ( int i = 0 ; i < model->num_total_layers ; ++i ) {
        struct Binary_Layer_Meta blm;
        memset(&blm, 0, sizeof(blm));
        if (layer_meta_create(&blm, &model->layers_meta[i]) < 0) goto save_fail;
        if (layer_meta_write(fptr, &blm, model, i) < 0) goto save_fail;
    }
    for ( int j = 0 ; j < model->num_total_layers ; ++j )
        if (payload_write(fptr, &model->layers_meta[j]) < 0) goto save_fail;

    fclose(fptr);
    return 0;

save_fail:
    return -1;
}

int header_check(FILE* fptr, struct Model* model) {

    char magic[4];
    if (read_bytes(fptr, magic, 4)   < 0) goto read_fail;
    if (memcmp(magic, "BNNW", 4) != 0) {
        RT_FAIL(RT_EINVAL, "Magic error");
    }

    // version (1 is the version number)
    uint32_t version;
    if (read_u32(fptr, &version)  < 0) goto read_fail;
    if (version != 1) {
        RT_FAIL(RT_EINVAL, "Version error");
    }

    uint32_t endian;
    if (read_u32(fptr, &endian)  < 0) goto read_fail;
    if (endian != 1) {
        // do endian conversion
        // return error for now
        RT_FAIL(RT_EINVAL, "Endian error");
    }

    uint32_t dtype;
    if (read_u32(fptr, &dtype)  < 0) goto read_fail;
    if (dtype != 2) {
        RT_FAIL(RT_EINVAL, "Header data type error");
    }

    uint32_t model_type;
    if (read_u32(fptr, &model_type)  < 0) goto read_fail;
    if (model_type >= 3) {
        RT_FAIL(RT_EINVAL, "Unknown model type");
    }

    uint32_t layer_count;
    if (read_u32(fptr, &layer_count)  < 0) goto read_fail;
    model->num_total_layers = layer_count;

    uint32_t input_h;
    if (read_u32(fptr, &input_h)  < 0) goto read_fail;
    model->init_input_h = input_h;
    uint32_t input_w;
    if (read_u32(fptr, &input_w)  < 0) goto read_fail;
    model->init_input_w = input_w;
    uint32_t input_c;
    if (read_u32(fptr, &input_c)  < 0) goto read_fail;
    model->init_input_c = input_c;

    return 0;

read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Read fail");
}

int layer_common_header_check(FILE* fptr, uint32_t* layer_type, struct PayloadEntry* payload_entry) {
    if (read_u32(fptr, layer_type) < 0)    goto read_fail;
    if (*layer_type >= 3) {
        RT_FAIL(RT_EINVAL, "Unknown layer type");
    }

    static uint32_t prev_layer_index = 0;
    uint32_t layer_index;
    if (read_u32(fptr, &layer_index) < 0)    goto read_fail;
    if (layer_index != 0 && (layer_index != prev_layer_index + 1)) {
        fprintf(stdout, "[WARNING] Incontiguous layer index at \"__func__\"");
    }

    uint64_t param_count;
    if (read_u64(fptr, &param_count) < 0)   goto read_fail;
    // check param_count is equal to layer's supposed param_count
    // printf("layer_common_header_check param_count: %I64u\n", param_count);

    uint64_t payload_bytes;
    if (read_u64(fptr, &payload_bytes) < 0) goto read_fail;
    payload_entry->payload_bytes = payload_bytes;
    // printf("layer_common_header_check payload_bytes: %I64u\n", param_count * sizeof(double));

    return 0;

read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Read fail");
}

int conv_meta_load(FILE* fptr, struct LayerMeta* layer, struct arena* a) {
    layer->layer_type = LAYER_CONV2D;

    uint32_t num_filter;
    if (read_u32(fptr, &num_filter) < 0)    goto read_fail;

    uint32_t in_channels;
    if (read_u32(fptr, &in_channels) < 0)    goto read_fail;

    uint32_t kernel_h;
    if (read_u32(fptr, &kernel_h) < 0)    goto read_fail;

    uint32_t kernel_w;
    if (read_u32(fptr, &kernel_w) < 0)    goto read_fail;

    uint32_t stride_h;
    if (read_u32(fptr, &stride_h) < 0)    goto read_fail;

    uint32_t stride_w;
    if (read_u32(fptr, &stride_w) < 0)    goto read_fail;

    uint32_t padding_h;
    if (read_u32(fptr, &padding_h) < 0)   goto read_fail;

    uint32_t padding_w;
    if (read_u32(fptr, &padding_w) < 0)    goto read_fail;

    uint32_t has_bias;
    if (read_u32(fptr, &has_bias) < 0)    goto read_fail;

    uint32_t pooling_type;
    if (read_u32(fptr, &pooling_type) < 0)    goto read_fail;

    uint32_t pooling_h;
    if (read_u32(fptr, &pooling_h) < 0)    goto read_fail;

    uint32_t pooling_w;
    if (read_u32(fptr, &pooling_w) < 0)    goto read_fail;

    layer->u_layer.conv = (struct Conv2D*)arena_alloc(a, sizeof(struct Conv2D));
    struct Conv2D* conv = layer->u_layer.conv;
    conv->num_filter = num_filter;
    conv->in_channels = in_channels;
    conv->kernel_h = kernel_h;
    conv->kernel_w = kernel_w;
    conv->stride_h = stride_h;
    conv->stride_w = stride_w;
    conv->padding_h = padding_h;
    conv->padding_w = padding_w;
    conv->has_bias = has_bias;
    if (pooling_type == 1) {
        conv->pType = MAX_POOL;
    }
    else if (pooling_type == 2) {
        conv->pType = AVG_POOL;
    }
    conv->pooling_h = pooling_h;
    conv->pooling_w = pooling_w;

    fseek(fptr, 4 * sizeof(uint32_t), SEEK_CUR);
    return 0;

read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Read fail");
}

int net_meta_peek(FILE* fptr, int* layers, uint32_t fc_layer_count) {

    uint64_t pos = ftell(fptr);

    for ( uint32_t i = 0 ; i < fc_layer_count ; ++i ) {
        uint32_t num_neurons;
        if (read_u32(fptr, &num_neurons) < 0) {
            fclose(fptr);
            RT_FAIL(RT_EIO, "Read fail");
        }
        fseek(fptr, 3 * sizeof(uint32_t), SEEK_CUR);
        layers[i] = num_neurons;
    }

    fseek(fptr, pos, SEEK_SET);
    return 0;
}

int fc_meta_load(FILE* fptr, struct Layer* layer) {

    uint32_t num_neurons;
    if (read_u32(fptr, &num_neurons) < 0)   goto read_fail;

    uint32_t input_dim;
    if (read_u32(fptr, &input_dim) < 0)    goto read_fail;

    uint32_t has_bias;
    if (read_u32(fptr, &has_bias) < 0)    goto read_fail;

    // reserved
    fseek(fptr, sizeof(uint32_t), SEEK_CUR);

    layer->num_neurons = num_neurons;
    layer->input_dim = input_dim;
    layer->has_bias = has_bias;

    return 0;

read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Read fail");
}

int net_meta_load(FILE* fptr, struct LayerMeta* lm, struct arena* a) {
    lm->layer_type = LAYER_FC;

    uint32_t network_type;
    if (read_u32(fptr, &network_type) < 0)    goto read_fail;

    uint32_t fc_layer_count;
    if (read_u32(fptr, &fc_layer_count) < 0)    goto read_fail;

    uint32_t input_size;
    if (read_u32(fptr, &input_size) < 0)    goto read_fail;

    fseek(fptr, 2 * sizeof(uint32_t), SEEK_CUR);

    int* layers = malloc(sizeof(int) * fc_layer_count);
    if (net_meta_peek(fptr, layers, fc_layer_count) < 0) goto read_fail;
    struct Network* network = create_Network(layers, input_size, fc_layer_count, a);
    network->network_type = network_type;
    network->layer_count = fc_layer_count;
    network->input_size = input_size;
    lm->u_layer.network = network;

    for ( uint32_t i = 0 ; i < fc_layer_count ; ++i ) {
        if (fc_meta_load(fptr, &network->layers[i]) < 0) goto read_fail;
    }

    free(layers);
    return 0;

read_fail:
    fclose(fptr);
    free(layers);
    RT_FAIL(RT_EIO, "Read fail");
}

int load_conv_weight(FILE* fptr, struct tk_conv2d* conv) {
    conv->filters = alloc3DArr(conv->num_filter, conv->kernel_h, conv->kernel_w);
    for ( int i = 0 ; i < conv->num_filter ; ++i ) {
        for ( int j = 0 ; j < conv->kernel_h ; ++j ) {
            for ( int k = 0 ; k < conv->kernel_w ; ++k ) {
                double weight;
                if (read_f64(fptr, &weight) < 0) {
                    fclose(fptr);
                    RT_FAIL(RT_EIO, "Read fail");
                }
                conv->filters[i][j][k] = weight;
            }
        }
    }
    return 0;
}

int load_net_weight(FILE* fptr, struct Network* network) {
    int layer_count = network->layer_count;
    for ( int i = 0 ; i < layer_count ; ++i ) {
        struct Layer* layer = &network->layers[i]; 
        uint32_t has_bias = layer->has_bias;
        for ( int j = 0 ; j < layer->num_neurons ; ++j ) {
            struct Neuron* neuron = &layer->neurons[j];
            for ( int k = 0 ; k < neuron->num_weight ; ++k ) {
                double weight;
                if (read_f64(fptr, &weight) < 0) goto read_fail;
                neuron->weights[k] = weight;
            }
            if (has_bias) {
                double bias;
                if (read_f64(fptr, &bias) < 0) goto read_fail;
                neuron->bias = bias;
            }
        }
    }
    return 0;

read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "Read fail");
}

int load_weight(FILE* fptr, struct Model* model, struct PayloadEntry* payload_entry) {

    if (!model) {
        RT_FAIL(RT_EINVAL, "Model is NULL");
    }

    for ( int i = 0 ; i < model->num_total_layers ; ++i ) {
        if (payload_entry[i].layer_type == LAYER_CONV2D) {
            if (load_conv_weight(fptr, model->layers_meta[i].u_layer.conv) < 0) return -1;
        }
        if (payload_entry[i].layer_type == LAYER_FC) {
            if (load_net_weight(fptr, model->layers_meta[i].u_layer.network) < 0) return -1;
        }
    }

    fclose(fptr);
    return 0;
}

int model_load(const char* path, struct arena* a, struct Model** model) {

    FILE* fptr = fopen(path, "rb");
    if (!fptr) {
        RT_FAIL(RT_EINVAL, "fopen error");
    }
    
    struct Model* model_rt = arena_alloc(a, sizeof(struct Model)); 
    *model = model_rt; 

    if (header_check(fptr, model)   < 0) goto read_fail;
    model->layers_meta = arena_alloc(a, sizeof(struct LayerMeta) * model->num_total_layers);

    // skip reserved space
    fseek(fptr, 8 * sizeof(uint32_t), SEEK_CUR);

    // temporary payload map
    struct PayloadEntry* payload_entry = malloc(sizeof(struct PayloadEntry) * model->num_total_layers);

    // skip param check for now
    for ( int i = 0 ; i < model->num_total_layers ; ++i ) {
        uint32_t layer_type;
        if (layer_common_header_check(fptr, &layer_type, &payload_entry[i]) < 0) goto read_fail;
        model->layers_meta[i].layer_index = i;
        model->layers_meta[i].dtype = LAYER_DTYPE_F64;
        if (layer_type == LAYER_CONV2D) {
            model->layers_meta[i].layer_type = LAYER_CONV2D;
            payload_entry[i].layer_type = LAYER_CONV2D;
            if (conv_meta_load(fptr, &model->layers_meta[i], a) < 0) goto read_fail;
        }
        if (layer_type == LAYER_FC) {
            model->layers_meta[i].layer_type = LAYER_FC;
            payload_entry[i].layer_type = LAYER_FC;
            if (net_meta_load(fptr, &model->layers_meta[i], a) < 0) goto read_fail;
        }
        if (layer_type == LAYER_POOL) {
            model->layers_meta[i].layer_type = LAYER_POOL;
            payload_entry[i].layer_type      = LAYER_POOL;
            payload_entry[i].payload_bytes   = 0;
            if (pool_meta_load(fptr, &model->layers_meta[i].u_layer.pool_layer_meta) < 0)
                goto read_fail;
}
    }

    long int pos = ftell(fptr);
    uint64_t offset = pos;
    for ( int i = 0 ; i < model->num_total_layers ; ++i ) {
        payload_entry[i].payload_offset = pos;
        pos += payload_entry[i].payload_bytes;
    }

    if (load_weight(fptr, model, payload_entry) < 0) goto read_fail;

    return 0;

read_fail:
    fclose(fptr);
    return -1;
}

