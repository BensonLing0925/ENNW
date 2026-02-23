#ifndef WEIGHTIO_H
#define WEIGHTIO_H

#include <stdint.h>

#define NETWORK_DTYPE_F32 1u
#define NETWORK_DTYPE_F64 2u

#define LAYER_FC 1u
#define LAYER_CONV2D 2u
#define LAYER_POOL 3u

struct Conv2D;
struct Network;

struct Binary_Header {
    char magic[4]; 
    uint32_t ver;
    uint32_t endian;    // little endian
    uint32_t dtype;                        // 1 => float32, 2 => float64(double)
    uint32_t model_type;    // 1 => FN, 2 => CNN
    uint32_t layer_count;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;   // (gray scale for now)
    uint32_t reserved[8]; // reserved
};

struct Binary_FC_Layer_Meta {
    uint32_t num_neurons;
    uint32_t input_dim; 
    uint32_t has_bias;
    uint32_t reserved;
};

struct Binary_Conv2D_Layer_Meta {
    uint32_t num_filter;
    uint32_t in_channels; 
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t padding_h;
    uint32_t padding_w;
    uint32_t has_bias;
    uint32_t pooling_type;
    uint32_t pooling_h;
    uint32_t pooling_w;
    uint32_t reserved[4];
};

struct Binary_Layer_Meta {
    uint32_t layer_type;
    uint32_t layer_index;
    uint64_t param_count;
    uint64_t payload_bytes;
    union {
        struct Binary_FC_Layer_Meta fc_layer_meta;
        struct Binary_Conv2D_Layer_Meta conv2d_layer_meta;
    } u_layer;
};

int save_weight(struct Network* net);
int load_weight(const char* weight_path);

#endif
