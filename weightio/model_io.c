/*
 * model_io.c — save_weight / model_load
 *
 * File layout:
 *   [Binary_Header]            68 bytes
 *   For each layer i:
 *     [uint32 layer_type]
 *     [uint32 layer_index]
 *     [uint64 param_count]
 *     [uint64 payload_bytes]
 *     [layer-specific meta]    (Binary_Conv2D_Layer_Meta / Binary_TF_Layer_Meta /
 *                               Binary_Net_Layer_Meta + N×Binary_FC_Layer_Meta)
 *   For each layer i:
 *     [raw double payload]     payload_bytes bytes
 *
 * Payload order per layer type:
 *   CONV2D : filters  [num_filter, in_c, kh, kw]
 *   TF     : q_weights, k_weights, v_weights,
 *            ffn_up_weights, ffn_down_weights,
 *            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta
 *   FC     : for each Linear: weights [in, out], bias [out]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "weightio.h"
#include "../src/error/rt_error.h"
#include "../src/ops/tensor.h"
#include "../src/modules/conv/conv.h"
#include "../src/modules/fc/fc.h"
#include "../src/modules/transformer/tf_block.h"
#include "../src/runtime/rt_context.h"
#include "../mem/arena.h"

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static int write_tensor_f64(FILE* fp, const struct tk_tensor* t) {
    uint64_t n = shape_size_calc(t->shape, t->ndims);
    const double* data = (const double*)t->data;
    for (uint64_t i = 0; i < n; ++i)
        if (write_f64(fp, data[i]) < 0) return -1;
    return 0;
}

static int read_tensor_f64(FILE* fp, struct tk_tensor* t) {
    uint64_t n = shape_size_calc(t->shape, t->ndims);
    double* data = (double*)t->data;
    for (uint64_t i = 0; i < n; ++i)
        if (read_f64(fp, &data[i]) < 0) return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* save_weight                                                          */
/* ------------------------------------------------------------------ */

int save_weight(const char* path, struct Model* model) {
    if (!path || !model) return -1;

    FILE* fptr = fopen(path, "wb");
    if (!fptr) {
        fprintf(stderr, "save_weight: cannot open '%s' for writing\n", path);
        return -1;
    }

    /* ---- File header ---- */
    struct Binary_Header bh;
    memcpy(bh.magic, "ENNW", 4);
    bh.ver        = 2;
    bh.endian     = 1;
    bh.dtype      = LAYER_DTYPE_F64;
    bh.model_type = 2;   /* CNN */
    bh.layer_count = (uint32_t)model->num_total_layers;
    bh.input_h    = (uint32_t)model->init_input_h;
    bh.input_w    = (uint32_t)model->init_input_w;
    bh.input_c    = (uint32_t)model->init_input_c;
    memset(bh.reserved, 0, sizeof(bh.reserved));
    if (header_write(fptr, &bh) < 0) goto fail;

    /* ---- Layer meta blocks ---- */
    for (int i = 0; i < model->num_total_layers; ++i) {
        struct LayerMeta* lm = &model->layers_meta[i];

        switch (lm->layer_type) {

        case LAYER_CONV2D: {
            struct tk_conv2d* conv = lm->u_layer.conv;
            uint64_t n = shape_size_calc(conv->filters->shape, conv->filters->ndims);
            if (write_u32(fptr, LAYER_CONV2D)        < 0) goto fail;
            if (write_u32(fptr, (uint32_t)i)         < 0) goto fail;
            if (write_u64(fptr, n)                   < 0) goto fail;
            if (write_u64(fptr, n * sizeof(double))  < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->num_filter)  < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->input_c)     < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->kernel_h)    < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->kernel_w)    < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->stride_h)    < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->stride_w)    < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->padding_h)   < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->padding_w)   < 0) goto fail;
            if (write_u32(fptr, (uint32_t)conv->has_bias)    < 0) goto fail;
            if (write_u32(fptr, 0) < 0) goto fail;   /* pooling_type = none */
            if (write_u32(fptr, 0) < 0) goto fail;   /* pooling_h */
            if (write_u32(fptr, 0) < 0) goto fail;   /* pooling_w */
            for (int r = 0; r < 4; ++r)
                if (write_u32(fptr, 0) < 0) goto fail;
            break;
        }

        case LAYER_TF: {
            struct TransformerBlock* tf = lm->u_layer.tf_block;
            int hidden  = tf->config.hidden_dim;
            int inter   = tf->config.inter_dim;
            int seq     = tf->config.seq_length;
            int n_heads = tf->config.n_heads;
            /* Q, K, V: [hidden, hidden] each; FFN up/down; 4× LN vecs */
            uint64_t n = (uint64_t)(hidden * hidden * 3 +
                                     hidden * inter  +
                                     inter  * hidden +
                                     hidden * 4);
            if (write_u32(fptr, LAYER_TF)            < 0) goto fail;
            if (write_u32(fptr, (uint32_t)i)         < 0) goto fail;
            if (write_u64(fptr, n)                   < 0) goto fail;
            if (write_u64(fptr, n * sizeof(double))  < 0) goto fail;
            if (write_u32(fptr, (uint32_t)seq)       < 0) goto fail;
            if (write_u32(fptr, (uint32_t)hidden)    < 0) goto fail;
            if (write_u32(fptr, (uint32_t)n_heads)   < 0) goto fail;
            if (write_u32(fptr, (uint32_t)inter)     < 0) goto fail;
            for (int r = 0; r < 4; ++r)
                if (write_u32(fptr, 0) < 0) goto fail;
            break;
        }

        case LAYER_FC: {
            struct Network* net = lm->u_layer.network;
            uint64_t n = 0;
            for (int j = 0; j < net->linear_count; ++j) {
                struct Linear* ln = &net->linears[j];
                n += shape_size_calc(ln->weights->shape, ln->weights->ndims);
                n += shape_size_calc(ln->bias->shape,    ln->bias->ndims);
            }
            if (write_u32(fptr, LAYER_FC)                          < 0) goto fail;
            if (write_u32(fptr, (uint32_t)i)                       < 0) goto fail;
            if (write_u64(fptr, n)                                  < 0) goto fail;
            if (write_u64(fptr, n * sizeof(double))                 < 0) goto fail;
            if (write_u32(fptr, (uint32_t)net->network_type)        < 0) goto fail;
            if (write_u32(fptr, (uint32_t)net->linear_count)        < 0) goto fail;
            if (write_u32(fptr, (uint32_t)net->input_size)          < 0) goto fail;
            if (write_u32(fptr, 0) < 0) goto fail;   /* reserved[0] */
            if (write_u32(fptr, 0) < 0) goto fail;   /* reserved[1] */
            for (int j = 0; j < net->linear_count; ++j) {
                struct Linear* ln = &net->linears[j];
                if (write_u32(fptr, (uint32_t)ln->num_neurons) < 0) goto fail;
                if (write_u32(fptr, (uint32_t)ln->input_dim)   < 0) goto fail;
                if (write_u32(fptr, (uint32_t)ln->has_bias)    < 0) goto fail;
                if (write_u32(fptr, 0)                          < 0) goto fail;
            }
            break;
        }

        default:
            fprintf(stderr, "save_weight: unknown layer type %u at index %d\n",
                    lm->layer_type, i);
            goto fail;
        }
    }

    /* ---- Payload blocks ---- */
    for (int i = 0; i < model->num_total_layers; ++i) {
        struct LayerMeta* lm = &model->layers_meta[i];
        switch (lm->layer_type) {

        case LAYER_CONV2D:
            if (write_tensor_f64(fptr, lm->u_layer.conv->filters) < 0) goto fail;
            break;

        case LAYER_TF: {
            struct TransformerBlock* tf = lm->u_layer.tf_block;
            if (write_tensor_f64(fptr, tf->q_weights)        < 0) goto fail;
            if (write_tensor_f64(fptr, tf->k_weights)        < 0) goto fail;
            if (write_tensor_f64(fptr, tf->v_weights)        < 0) goto fail;
            if (write_tensor_f64(fptr, tf->ffn_up_weights)   < 0) goto fail;
            if (write_tensor_f64(fptr, tf->ffn_down_weights) < 0) goto fail;
            if (write_tensor_f64(fptr, tf->ln1_gamma)        < 0) goto fail;
            if (write_tensor_f64(fptr, tf->ln1_beta)         < 0) goto fail;
            if (write_tensor_f64(fptr, tf->ln2_gamma)        < 0) goto fail;
            if (write_tensor_f64(fptr, tf->ln2_beta)         < 0) goto fail;
            break;
        }

        case LAYER_FC: {
            struct Network* net = lm->u_layer.network;
            for (int j = 0; j < net->linear_count; ++j) {
                struct Linear* ln = &net->linears[j];
                if (write_tensor_f64(fptr, ln->weights) < 0) goto fail;
                if (write_tensor_f64(fptr, ln->bias)    < 0) goto fail;
            }
            break;
        }
        }
    }

    fclose(fptr);
    printf("[weightio] Saved to '%s'\n", path);
    return 0;

fail:
    fclose(fptr);
    fprintf(stderr, "save_weight: write error\n");
    return -1;
}

/* ------------------------------------------------------------------ */
/* model_load                                                           */
/* ------------------------------------------------------------------ */

int model_load(const char* path, struct tk_rt_ctx* ctx, struct Model* model) {
    (void)ctx;  /* reserved for future arena-based allocation */

    if (!path || !model) return -1;

    FILE* fptr = fopen(path, "rb");
    if (!fptr) {
        fprintf(stderr, "model_load: cannot open '%s'\n", path);
        return -1;
    }

    /* ---- Validate header ---- */
    char magic[4];
    if (fread(magic, 1, 4, fptr) != 4 || memcmp(magic, "ENNW", 4) != 0) {
        fprintf(stderr, "model_load: bad magic bytes\n");
        fclose(fptr); return -1;
    }

    uint32_t ver, endian, dtype, model_type, layer_count;
    uint32_t ih, iw, ic;
    if (read_u32(fptr, &ver)        < 0) goto read_fail;
    if (read_u32(fptr, &endian)     < 0) goto read_fail;
    if (read_u32(fptr, &dtype)      < 0) goto read_fail;
    if (read_u32(fptr, &model_type) < 0) goto read_fail;
    if (read_u32(fptr, &layer_count)< 0) goto read_fail;
    if (read_u32(fptr, &ih)         < 0) goto read_fail;
    if (read_u32(fptr, &iw)         < 0) goto read_fail;
    if (read_u32(fptr, &ic)         < 0) goto read_fail;
    fseek(fptr, 8 * (long)sizeof(uint32_t), SEEK_CUR);  /* skip reserved */

    if ((int)layer_count != model->num_total_layers) {
        fprintf(stderr, "model_load: layer count mismatch (file=%u, model=%d)\n",
                layer_count, model->num_total_layers);
        fclose(fptr); return -1;
    }

    /* ---- Skip meta blocks, remember layer order ---- */
    uint32_t* order = malloc(sizeof(uint32_t) * layer_count);
    if (!order) { fclose(fptr); return -1; }

    for (uint32_t i = 0; i < layer_count; ++i) {
        uint32_t ltype, lidx;
        uint64_t param_count, payload_bytes;
        if (read_u32(fptr, &ltype)         < 0) goto read_fail_free;
        if (read_u32(fptr, &lidx)          < 0) goto read_fail_free;
        if (read_u64(fptr, &param_count)   < 0) goto read_fail_free;
        if (read_u64(fptr, &payload_bytes) < 0) goto read_fail_free;
        order[i] = ltype;

        switch (ltype) {
        case LAYER_CONV2D:
            fseek(fptr, (long)sizeof(struct Binary_Conv2D_Layer_Meta), SEEK_CUR);
            break;
        case LAYER_TF:
            fseek(fptr, (long)sizeof(struct Binary_TF_Layer_Meta), SEEK_CUR);
            break;
        case LAYER_FC: {
            uint32_t net_type, fc_count, input_sz, r0, r1;
            if (read_u32(fptr, &net_type)  < 0) goto read_fail_free;
            if (read_u32(fptr, &fc_count)  < 0) goto read_fail_free;
            if (read_u32(fptr, &input_sz)  < 0) goto read_fail_free;
            if (read_u32(fptr, &r0)        < 0) goto read_fail_free;
            if (read_u32(fptr, &r1)        < 0) goto read_fail_free;
            fseek(fptr, (long)(fc_count * sizeof(struct Binary_FC_Layer_Meta)), SEEK_CUR);
            break;
        }
        case LAYER_POOL:
            fseek(fptr, (long)sizeof(struct Binary_Pool_Layer_Meta), SEEK_CUR);
            break;
        default:
            fprintf(stderr, "model_load: unknown layer type %u\n", ltype);
            goto read_fail_free;
        }
    }

    /* ---- Load payloads in order ---- */
    for (uint32_t i = 0; i < layer_count; ++i) {
        struct LayerMeta* lm = &model->layers_meta[i];
        if (lm->layer_type != order[i]) {
            fprintf(stderr, "model_load: type mismatch at layer %u "
                    "(file type=%u, model type=%u)\n", i, order[i], lm->layer_type);
            goto read_fail_free;
        }

        switch (lm->layer_type) {

        case LAYER_CONV2D:
            if (read_tensor_f64(fptr, lm->u_layer.conv->filters) < 0)
                goto read_fail_free;
            break;

        case LAYER_TF: {
            struct TransformerBlock* tf = lm->u_layer.tf_block;
            if (read_tensor_f64(fptr, tf->q_weights)        < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->k_weights)        < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->v_weights)        < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->ffn_up_weights)   < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->ffn_down_weights) < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->ln1_gamma)        < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->ln1_beta)         < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->ln2_gamma)        < 0) goto read_fail_free;
            if (read_tensor_f64(fptr, tf->ln2_beta)         < 0) goto read_fail_free;
            break;
        }

        case LAYER_FC: {
            struct Network* net = lm->u_layer.network;
            for (int j = 0; j < net->linear_count; ++j) {
                struct Linear* ln = &net->linears[j];
                if (read_tensor_f64(fptr, ln->weights) < 0) goto read_fail_free;
                if (read_tensor_f64(fptr, ln->bias)    < 0) goto read_fail_free;
            }
            break;
        }

        case LAYER_POOL:
            /* pooling has no learnable weights */
            break;
        }
    }

    free(order);
    fclose(fptr);
    printf("[weightio] Loaded weights from '%s'\n", path);
    return 0;

read_fail_free:
    free(order);
read_fail:
    fclose(fptr);
    fprintf(stderr, "model_load: read error\n");
    return -1;
}
