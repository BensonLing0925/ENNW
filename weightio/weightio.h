#ifndef WEIGHTIO_H
#define WEIGHTIO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/structDef.h"
#include "../src/modules/conv/conv.h"
#include "../src/modules/fc/fc.h"
#include "../src/runtime/rt_context.h"

#define FIXED_HEADER_SIZE 68u

#define LAYER_DTYPE_F32 1u
#define LAYER_DTYPE_F64 2u

#define LAYER_FC     1u
#define LAYER_CONV2D 2u
#define LAYER_POOL   3u
#define LAYER_TF     4u

#define FC_CHAIN 1u

#define WEIGHT_OK               0
#define WEIGHT_ERR_IO           1
#define WEIGHT_ERR_MAGIC        2
#define WEIGHT_ERR_VERSION      3
#define WEIGHT_ERR_DTYPE        4
#define WEIGHT_ERR_FORMAT       5
#define WEIGHT_ERR_OOM          6
#define WEIGHT_FAIL_READ        7
#define WEIGHT_UNKNOWN_LAYER_TYPE 8

struct Binary_Header {
    char     magic[4];
    uint32_t ver;
    uint32_t endian;      /* 1 = little-endian */
    uint32_t dtype;       /* 1 = float32, 2 = float64 */
    uint32_t model_type;  /* 1 = FC-only, 2 = CNN */
    uint32_t layer_count;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t input_c;
    uint32_t reserved[8];
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
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_h;
    uint32_t padding_w;
    uint32_t pooling_type;
    uint32_t reserved[4];
};

/* Transformer block meta — written after the common 24-byte header */
struct Binary_TF_Layer_Meta {
    uint32_t seq_length;
    uint32_t hidden_dim;
    uint32_t n_heads;
    uint32_t inter_dim;
    uint32_t reserved[4];
};

struct Binary_Layer_Meta {
    uint32_t layer_type;
    uint32_t layer_index;
    uint64_t param_count;
    uint64_t payload_bytes;
    union {
        struct Binary_Net_Layer_Meta    net_layer_meta;
        struct Binary_Conv2D_Layer_Meta conv2d_layer_meta;
        struct Binary_Pool_Layer_Meta   pool_layer_meta;
        struct Binary_TF_Layer_Meta     tf_layer_meta;
    } u_layer;
};

typedef struct PayloadEntry {
    uint32_t layer_type;
    uint64_t payload_bytes;
    uint64_t payload_offset;
} PayloadEntry;

/* Low-level I/O helpers (defined in weightio.c) */
int write_bytes(FILE* fp, const void* ptr, size_t n);
int write_u32(FILE* fp, uint32_t v);
int write_u64(FILE* fp, uint64_t v);
int write_f64(FILE* fp, double v);
int read_bytes(FILE* fp, void* ptr, size_t n);
int read_u32(FILE* fp, uint32_t* v);
int read_u64(FILE* fp, uint64_t* v);
int read_f64(FILE* fp, double* v);

int header_write(FILE* fptr, const struct Binary_Header* bh);
int pool_meta_write(FILE* fptr, const struct Binary_Pool_Layer_Meta* pm);
int pool_meta_load(FILE* fptr, struct Binary_Pool_Layer_Meta* pm);
int fc_meta_write(FILE* fptr, const struct Binary_FC_Layer_Meta* fc_blm);
int conv2d_meta_write(FILE* fptr, const struct Binary_Conv2D_Layer_Meta* m);

/*
 * Save all layer weights to a binary file.
 * model->layers_meta must be fully populated (conv, tf, network pointers set).
 */
int save_weight(const char* path, struct Model* model);

/*
 * Load weights from a binary file into a pre-built model.
 * The model must already have all layer structs allocated with the correct
 * shapes (same architecture as when the file was saved).
 * Only the weight/bias data tensors are filled in.
 */
int model_load(const char* path, struct tk_rt_ctx* ctx, struct Model* model);

#endif
