#include "../src/structDef.h"
#include "weightio.h"

int header_create(struct Binary_Header* bh, struct Model* model) {

    memcpy(bh->magic, "BNNW", 4);
    bh->ver = 1;
    bh->endian = 1;    // little endian
    bh->dtype = 2;                        // 1 => float32, 2 => float64(double)
    bh->model_type = 2;    // 1 => FN, 2 => CNN
    bh->layer_count = mode->num_total_layer;
    if (model.has_conv) {
        bh->input_h = model->conv->input_rows;
        bh->input_w = model->conv->input_cols;
        bh->input_c = model->conv->in_channels;   // (gray scale for now)
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
    for ( int i = 0 ; i < net->num_layers ; ++i )
        total_weights += fc_param_calculate(&net->layers[i]);
    return total_weights;
}

uint64_t conv2d_param_calculate(const struct Conv2D* conv) {
    if (!conv) {
        printf("[WARNING] Conv2D is NULL\n");
        return 0;
    }
    uint64_t param_count = (uint64_t) (conv->num_filter) * (uint64_t) (conv->filter_size) * (uint64_t) (conv->filter_size) * (uint64_t) (conv->in_channels);
    return param_count;
}

uint64_t layer_param_calculate(struct LayerMeta* layer) {
    uint64_t num_layer_param = 0;
    switch (layer->layer_type) {
        case LAYER_FC:
            num_layer_param += network_fc_param_calculate(layer->net); 
            break;
        case LAYER_CONV2D:
            num_layer_param += conv2d_param_calculate(layer->conv);
            break;
        default:
            printf("Unknown Layer Type at \"layer_param_calculate\"");
            break;
    }
    return num_layer_param;
}

int fc_layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* lm) {
    if (blm->layer_type != lm->layer_type) {
        printf("[ERROR] incompatible Meta conversion at \"fc_layer_meta_create\"");
        return -1;
    }
    struct Binary_FC_Layer_Meta* meta = &(blm->u_layer.fc_layer_meta);
    meta->num_neurons = lm->layer->num_neurons;
    meta->input_dim = lm->layer->input_dim;
    meta->has_bias = lm->layer->has_bias;
    meta->reserved = 0u;
    return 0;
}

int conv2d_layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* lm) {
    if (blm->layer_type != lm->layer_type) {
        printf("[ERROR] incompatible Meta conversion at \"conv2d_layer_meta_create\"");
        return -1;
    }
    struct Binary_FC_Layer_Meta* meta = &(blm->u_layer.conv2d_layer_meta);
    meta->num_filter = lm->num_filter;
    meta->in_channels = lm->in_channels;
    meta->kernel_h = lm->filter_size; //assume square for now
    meta->kernel_w = lm->filter_size;
    meta->stride_h = lm->stride_h;
    meta->stride_w = lm->stride_w;
    meta->padding_h = lm->padding_h;
    meta->padding_w = lm->padding_w;
    meta->has_bias = lm->has_bias;
    meta->pooling_type = lm->pType;
    if (lm->pType == enum PoolingType MAX_POOL)
        meta->pooling_type = 1;
    else if (lm->pType == enum PoolingType AVG_POOL)
        meta->pooling_type = 2;
    else
        meta->pooling_type = 0;
    // assume pooling square for now
    meta->pooling_h = lm->pooling_size;
    meta->pooling_w = lm->pooling_size;
    memset(meta->reserved, 0, sizeof(meta->reserved));
    return 0;
}

void layer_meta_create(struct Binary_Layer_Meta* blm, struct LayerMeta* lm) {
    blm->layer_type = lm->layer_type;
    blm->layer_index = lm->layer_index;
    blm->param_count = layer_param_calculate(lm);
    switch (lm->dtype) {
        case NETWORK_DTYPE_F32:
            blm->payload_bytes = blm->param_count * sizeof(float);
            break;
        case NETWORK_DTYPE_F64:
            blm->payload_bytes = blm->param_count * sizeof(double);
            break;
        default:
            printf("Unknown Layer Type at \"layer_meta_create\"");
            break;
    }

    switch (blm->layer_type) {
        case LAYER_FC:
            if (fc_layer_meta_create(blm, lm) != 0) {
                return;
            } 
            break;
        case LAYER_CONV2D:
            if (conv2d_layer_meta_create(blm, lm) != 0) {
                return;
            }
            break;
        case LAYER_POOL:
            break;
        default:
            printf("[ERROR] Unknown Binary Layer Type at \"layer_meta_create\"");
            break;

    }
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
    perror("[ERROR] write at \"header_write\"");
    fclose(fptr);
    return -1;
}

int fc_meta_write(FILE* fptr, const struct Binary_FC_Layer_Meta* fc_blm) {
    if (!fptr || !fc_blm) return -1;

    if (write_u32(fptr, fc_blm->num_neurons) < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->input_dim)   < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->has_bias)    < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->reserved)    < 0) goto write_fail;

    return 0;

write_fail:
    perror("[ERROR] write fail at \"fc_meta_write\"");
    fclose(fptr);
    return -1;
}

int conv2d_meta_write(FILE *fptr, const struct Binary_Conv2D_Layer_Meta *m) {
    if (!fptr || !m) return -1;

    if (write_u32(fptr, m->num_neurons)   < 0) goto write_fail;
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
    perror("[ERROR] write fail at \"conv_meta_write\"\n");
    return -1;
}

int layer_meta_write(FILE* fptr, const struct Binary_Layer_Meta* blm) {
    if (!fptr || !blm) return -1;

    if (write_u32(fptr, blm->layer_type)    < 0) goto write_fail;
    if (write_u32(fptr, blm->layer_index)   < 0) goto write_fail;
    if (write_u64(fptr, blm->param_count)   < 0) goto write_fail;
    if (write_u64(fptr, blm->payload_bytes) < 0) goto write_fail;

    switch (blm->layer_type) {
        case LAYER_FC:
            if (fc_meta_write(fptr, &blm->u_layer.fc_layer_meta) < 0) goto write_fail;
            break;

        case LAYER_CONV2D:
            if (conv2d_meta_write(fptr, &blm->u_layer.conv2d_layer_meta) < 0) goto write_fail;
            break;

        default:
            fprintf(stderr,
                    "[ERROR] unknown layer type (%u) at \"layer_meta_write\"\n",
                    (unsigned)blm->layer_type);
            goto write_fail;
    }

    return 0;

write_fail:
    perror("[ERROR] write fail at \"layer_meta_write\"");
    fclose(fptr);
    return -1;
}

int fc_payload_write(FILE* fptr, const struct Layer* layer) {
    for ( int i = 0 ; i < layer->num_neurons ; ++i ) {
        const struct Neuron* neuron = &layer->neurons[i];
        for ( int j = 0 ; j < neuron->num_weight ; ++j ) {
            if (write_f64(fptr, neuron->weights[j]) < 0) goto write_fail; 
            if (neuron->has_bias)
                if (write_f64(fptr, neuron->bias) < 0) goto write_fail;
        }
    }
    return 0;

write_fail:
    fclose(fptr);
    return -1;
}

int conv2d_payload_write(FILE* fptr, const struct Conv2D* conv) {
    for ( int i = 0 ; i < conv->num_filter ; ++i ) {
        const struct Double2D* current_filter = &conv->filters[i];
        // assuming square for now
        int filter_height = conv->filter_size;
        int filter_width = conv->filter_size;
        for ( int filter_h = 0 ; filter_h < filter_height ; ++filter_h ) {
            for ( int filter_w = 0 ; filter_w < filter_width ; ++filter_w ) {
                if (write_f64(fptr, current_filter[filter_h][filter_w]) < 0) goto write_fail;
                if (conv->has_bias) {
                    if (write_f64(fptr, conv->biases[i]) < 0) goto write_fail;
                }
            }
        }
    }
write_fail:
    fclose(fptr);
    return -1;
}

int payload_write(FILE* fptr, struct LayerMeta* lm) {
    switch (lm.layer_type) {
        case LAYER_FC:
            if (fc_payload_write(lm->u_layer.layer) != 0) goto write_fail;
            break;
        case LAYER_CONV2D:
            if (conv2d_payload_write(lm->u_layer.conv) != 0) goto write_fail;
        default:
            break;
    }

write_fail:
    fclose(fptr);
    return -1;
}

int save_weight(const char* path, struct Model* model) {
    FILE* fptr = fopen(path, "wb");
    if (!fptr) {
        perror("[ERROR] fopen at \"save_weight\"");
        return -1;
    }

    struct Binary_Header bh;
    if (header_create(&bh, model) < 0) goto save_fail;
    if (header_write(fptr, bh) < 0) goto save_fail;
    for ( int i = 0 ; i < model->num_total_layer ; ++i ) {
        struct Binary_Layer_Meta blm;
        if (layer_meta_create(&blm, model->layers_meta[i]) < 0) goto save_fail;
        if (layer_meta_write(fptr, blm) < 0) goto save_fail;
    }
    if (payload_write(fptr, model->layers_meta) < 0) goto save_fail;
    fclose(fptr);
}
