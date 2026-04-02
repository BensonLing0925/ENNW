#ifndef WEIGHTIO_H
#define WEIGHTIO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/structDef.h"
#include "../src/modules/conv/conv.h"
#include "../src/modules/fc/fc.h"

#define FIXED_HEADER_SIZE 68u

#define LAYER_DTYPE_F32 1u
#define LAYER_DTYPE_F64 2u

#define LAYER_FC 1u
#define LAYER_CONV2D 2u
#define LAYER_POOL 3u

#define FC_CHAIN 1u

#define WEIGHT_OK 0
#define WEIGHT_ERR_IO 1
#define WEIGHT_ERR_MAGIC 2
#define WEIGHT_ERR_VERSION 3
#define WEIGHT_ERR_DTYPE 4
#define WEIGHT_ERR_FORMAT 5
#define WEIGHT_ERR_OOM 6
#define WEIGHT_FAIL_READ 7
#define WEIGHT_UNKNOWN_LAYER_TYPE 8

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

struct Binary_Net_Layer_Meta {
    uint32_t network_type;
    uint32_t fc_layer_count;
    uint32_t input_size;
    uint32_t reserved[2];
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

struct Binary_Pool_Layer_Meta {
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t size_h;
    uint32_t size_w;
    uint32_t padding_h;
    uint32_t padding_w;
    uint32_t pooling_type;
    uint32_t reserved[4];
};

struct Binary_Layer_Meta {
    uint32_t layer_type;
    uint32_t layer_index;
    uint64_t param_count;
    uint64_t payload_bytes;
    union {
        struct Binary_Net_Layer_Meta net_layer_meta;
        struct Binary_Conv2D_Layer_Meta conv2d_layer_meta;
        struct Binary_Pool_Layer_Meta   pool_layer_meta;
    } u_layer;
};

typedef struct PayloadEntry {
    uint32_t layer_type;     // LAYER_CONV2D / LAYER_FC (network block)
    uint64_t payload_bytes;  // current block's payload length
    uint64_t payload_offset; // payload offset（absolute file offset）
} PayloadEntry;

int save_weight(const char* path, struct Model* model);
int model_load(const char* path, struct arena* a, struct Model* model);

#endif
