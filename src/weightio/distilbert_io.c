/*
 * distilbert_io.c
 *
 * Loader for the DBERT binary weight format produced by
 * tools/export_distilbert_weights.py.
 *
 * All weight matrices are pre-transposed in the Python exporter so the
 * C loader does straight fread() into the existing tensor data buffers —
 * no per-element loops.
 *
 * If a bias tensor is NULL (the model was configured with use_*_bias = 0
 * but the file still contains that section), the bytes are skipped.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "distilbert_io.h"
#include "rt_error.h"
#include "tensor.h"
#include "distilbert_cls.h"
#include "tf_block.h"

/* ------------------------------------------------------------------ */
/* Low-level helpers                                                    */
/* ------------------------------------------------------------------ */

static int read_u32(FILE* fp, uint32_t* v) {
    return fread(v, sizeof(*v), 1, fp) == 1 ? 0 : -1;
}

/* Bulk-read n float32 values directly into a pre-allocated buffer. */
static int read_f32_bulk(FILE* fp, void* data, uint64_t n) {
    size_t bytes = (size_t)n * sizeof(float);
    return fread(data, 1, bytes, fp) == bytes ? 0 : -1;
}

/* Skip n float32 values in the file (tensor not used in this model). */
static int skip_f32(FILE* fp, uint64_t n) {
    return fseek(fp, (long)((size_t)n * sizeof(float)), SEEK_CUR) == 0 ? 0 : -1;
}

/*
 * Read n float32 values into tensor t->data, or skip them if t == NULL.
 * n must match the expected element count for that tensor slot.
 */
static int load_or_skip(FILE* fp, struct tk_tensor* t, uint64_t n) {
    if (t)
        return read_f32_bulk(fp, t->data, n);
    return skip_f32(fp, n);
}

/* ---- Version 3 (int8) helpers ---- */

/* Read n int8 values directly into a pre-allocated TK_I8 tensor buffer. */
static int read_i8_bulk(FILE* fp, void* data, uint64_t n) {
    return fread(data, sizeof(int8_t), n, fp) == n ? 0 : -1;
}

/* Skip a v3 weight entry: n int8 elements + weight_scale (f32) + act_scale (f32). */
static int skip_i8_weight(FILE* fp, uint64_t n) {
    long bytes = (long)(n * sizeof(int8_t) + 2 * sizeof(float));
    return fseek(fp, bytes, SEEK_CUR) == 0 ? 0 : -1;
}

/*
 * Version 3: read n int8 values, then weight_scale, then act_scale.
 * act_scale == 0.0 means dynamic quantization (not calibrated).
 * If t == NULL, the whole entry is skipped.
 */
static int load_or_skip_i8(FILE* fp, struct tk_tensor* t, uint64_t n,
                            float* weight_scale_out, float* act_scale_out) {
    if (t) {
        if (read_i8_bulk(fp, t->data, n) < 0) return -1;
        if (fread(weight_scale_out, sizeof(float), 1, fp) != 1) return -1;
        if (fread(act_scale_out,    sizeof(float), 1, fp) != 1) return -1;
        return 0;
    }
    return skip_i8_weight(fp, n);
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

int tk_distilbert_load_weights(const char* path,
                               const struct tk_distilbert_config* cfg,
                               struct tk_distilbert* model,
                               uint32_t* version_out) {
    if (!path || !cfg || !model)
        RT_FAIL(RT_EINVAL, "tk_distilbert_load_weights: NULL argument");

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[distilbert_io] cannot open '%s'\n", path);
        RT_FAIL(RT_EIO, "cannot open weight file");
    }

    /* ---- Read and validate header ---- */
    char magic[DBERT_MAGIC_LEN];
    if (fread(magic, 1, DBERT_MAGIC_LEN, fp) != DBERT_MAGIC_LEN ||
        memcmp(magic, DBERT_MAGIC, DBERT_MAGIC_LEN) != 0) {
        fprintf(stderr, "[distilbert_io] bad magic bytes in '%s'\n", path);
        fclose(fp); RT_FAIL(RT_EIO, "bad DBERT magic");
    }

    uint32_t version, num_layers, hidden_dim, inter_dim;
    uint32_t vocab_size, max_seq_len, n_heads, reserved;
    if (read_u32(fp, &version)    < 0 ||
        read_u32(fp, &num_layers) < 0 ||
        read_u32(fp, &hidden_dim) < 0 ||
        read_u32(fp, &inter_dim)  < 0 ||
        read_u32(fp, &vocab_size) < 0 ||
        read_u32(fp, &max_seq_len)< 0 ||
        read_u32(fp, &n_heads)    < 0 ||
        read_u32(fp, &reserved)   < 0) {
        fclose(fp); RT_FAIL(RT_EIO, "header read failed");
    }

    if (version != DBERT_VERSION &&
        version != DBERT_VERSION_SST2 &&
        version != DBERT_VERSION_INT8) {
        fprintf(stderr, "[distilbert_io] unsupported version %u\n", version);
        fclose(fp); RT_FAIL(RT_EINVAL, "unsupported DBERT version");
    }
    if (version_out) *version_out = version;

    /* Validate dimensions against the C model config */
    int ok = (int)num_layers  == cfg->num_layers  &&
             (int)hidden_dim  == cfg->hidden_dim   &&
             (int)vocab_size  == cfg->vocab_size   &&
             (int)max_seq_len == cfg->max_seq_len  &&
             (int)n_heads     == cfg->n_heads;

    int inter_cfg = (cfg->inter_dim == 0) ? cfg->hidden_dim * 4 : cfg->inter_dim;
    ok = ok && ((int)inter_dim == inter_cfg);

    if (!ok) {
        fprintf(stderr,
                "[distilbert_io] dimension mismatch:\n"
                "  file : layers=%u hidden=%u inter=%u vocab=%u max_seq=%u heads=%u\n"
                "  model: layers=%d hidden=%d inter=%d vocab=%d max_seq=%d heads=%d\n",
                num_layers, hidden_dim, inter_dim, vocab_size, max_seq_len, n_heads,
                cfg->num_layers, cfg->hidden_dim, inter_cfg,
                cfg->vocab_size, cfg->max_seq_len, cfg->n_heads);
        fclose(fp); RT_FAIL(RT_EINVAL, "DBERT dimension mismatch");
    }

    uint64_t H = hidden_dim;
    uint64_t I = inter_dim;
    uint64_t V = vocab_size;
    uint64_t S = max_seq_len;

    struct tk_distilbert_emb* emb = model->emb;

    /* ---- Embedding section ---- */
    if (read_f32_bulk(fp, emb->word_emb->weights->data, V * H) < 0) goto io_err;
    if (read_f32_bulk(fp, emb->pos_emb->weights->data,  S * H) < 0) goto io_err;
    if (read_f32_bulk(fp, emb->ln_gamma->data,          H)     < 0) goto io_err;
    if (read_f32_bulk(fp, emb->ln_beta->data,           H)     < 0) goto io_err;

    printf("[distilbert_io] embeddings loaded\n");

    /* ---- Transformer layers ---- */
    int is_v3 = (version == DBERT_VERSION_INT8);

    for (int i = 0; i < (int)num_layers; ++i) {
        struct TransformerBlock* tf = model->blocks[i]->base;

        if (is_v3) {
            /* int8 weight matrices: int8_data + weight_scale + act_scale
             * Biases and LayerNorm remain float32.
             * Q, K, V share the same input, so all three carry attn_in_act_scale. */
            float aq = 0.0f, ak = 0.0f, av = 0.0f;   /* Q/K/V act scales (same value) */
            float ao = 0.0f, afu = 0.0f, afd = 0.0f; /* o_proj, ffn_up, ffn_down */

            if (load_or_skip_i8(fp, tf->q_weights, H*H, &tf->q_scale, &aq)           < 0) goto io_err;
            if (load_or_skip(fp, tf->q_bias, H)                                       < 0) goto io_err;
            if (load_or_skip_i8(fp, tf->k_weights, H*H, &tf->k_scale, &ak)           < 0) goto io_err;
            if (load_or_skip(fp, tf->k_bias, H)                                       < 0) goto io_err;
            if (load_or_skip_i8(fp, tf->v_weights, H*H, &tf->v_scale, &av)           < 0) goto io_err;
            if (load_or_skip(fp, tf->v_bias, H)                                       < 0) goto io_err;
            if (load_or_skip_i8(fp, tf->o_proj_weights, H*H, &tf->o_proj_scale, &ao) < 0) goto io_err;
            if (load_or_skip(fp, tf->o_proj_bias, H)                                  < 0) goto io_err;
            if (load_or_skip(fp, tf->ln1_gamma, H)                                    < 0) goto io_err;
            if (load_or_skip(fp, tf->ln1_beta,  H)                                    < 0) goto io_err;
            if (load_or_skip_i8(fp, tf->ffn_up_weights,   H*I, &tf->ffn_up_scale,   &afu) < 0) goto io_err;
            if (load_or_skip(fp, tf->ffn_up_bias,   I)                                < 0) goto io_err;
            if (load_or_skip_i8(fp, tf->ffn_down_weights, I*H, &tf->ffn_down_scale,  &afd) < 0) goto io_err;
            if (load_or_skip(fp, tf->ffn_down_bias, H)                                < 0) goto io_err;
            if (load_or_skip(fp, tf->ln2_gamma, H)                                    < 0) goto io_err;
            if (load_or_skip(fp, tf->ln2_beta,  H)                                    < 0) goto io_err;

            /* Store calibration act scales (0.0 means dynamic) */
            tf->attn_in_act_scale     = aq;   /* Q, K, V all same input */
            tf->o_proj_in_act_scale   = ao;
            tf->ffn_up_in_act_scale   = afu;
            tf->ffn_down_in_act_scale = afd;
            (void)ak; (void)av;               /* redundant, same as aq */
        } else {
            /* float32 all weights (version 1 and 2) */
            if (load_or_skip(fp, tf->q_weights,    H * H) < 0) goto io_err;
            if (load_or_skip(fp, tf->q_bias,        H)    < 0) goto io_err;
            if (load_or_skip(fp, tf->k_weights,    H * H) < 0) goto io_err;
            if (load_or_skip(fp, tf->k_bias,        H)    < 0) goto io_err;
            if (load_or_skip(fp, tf->v_weights,    H * H) < 0) goto io_err;
            if (load_or_skip(fp, tf->v_bias,        H)    < 0) goto io_err;
            if (load_or_skip(fp, tf->o_proj_weights, H * H) < 0) goto io_err;
            if (load_or_skip(fp, tf->o_proj_bias,    H)     < 0) goto io_err;
            if (load_or_skip(fp, tf->ln1_gamma, H) < 0) goto io_err;
            if (load_or_skip(fp, tf->ln1_beta,  H) < 0) goto io_err;
            if (load_or_skip(fp, tf->ffn_up_weights,   H * I) < 0) goto io_err;
            if (load_or_skip(fp, tf->ffn_up_bias,       I)    < 0) goto io_err;
            if (load_or_skip(fp, tf->ffn_down_weights, I * H) < 0) goto io_err;
            if (load_or_skip(fp, tf->ffn_down_bias,     H)    < 0) goto io_err;
            if (load_or_skip(fp, tf->ln2_gamma, H) < 0) goto io_err;
            if (load_or_skip(fp, tf->ln2_beta,  H) < 0) goto io_err;
        }

        printf("[distilbert_io] layer %d loaded\n", i);
    }

    fclose(fp);
    printf("[distilbert_io] all weights loaded from '%s'\n", path);
    return 0;

io_err:
    fclose(fp);
    fprintf(stderr, "[distilbert_io] read error in '%s'\n", path);
    RT_FAIL(RT_EIO, "weight read failed");
}

/* ------------------------------------------------------------------ */
/* SST-2 classification head loader (VERSION 2 files only)            */
/* ------------------------------------------------------------------ */

/*
 * Seek past the base model weights (header + embeddings + transformer layers)
 * then read: pre_cls_w [H,H], pre_cls_b [H], cls_w [H,2], cls_b [2].
 */
int tk_distilbert_load_cls_weights(const char* path,
                                   const struct tk_distilbert_config* cfg,
                                   struct tk_sst2_cls_head* head) {
    if (!path || !cfg || !head)
        RT_FAIL(RT_EINVAL, "tk_distilbert_load_cls_weights: NULL argument");

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[distilbert_io] cannot open '%s'\n", path);
        RT_FAIL(RT_EIO, "cannot open weight file");
    }

    /* Re-read header to get version and dims */
    char magic[DBERT_MAGIC_LEN];
    if (fread(magic, 1, DBERT_MAGIC_LEN, fp) != DBERT_MAGIC_LEN ||
        memcmp(magic, DBERT_MAGIC, DBERT_MAGIC_LEN) != 0) {
        fclose(fp); RT_FAIL(RT_EIO, "bad DBERT magic");
    }

    uint32_t version, num_layers, hidden_dim, inter_dim;
    uint32_t vocab_size, max_seq_len, n_heads, reserved;
    if (read_u32(fp, &version)    < 0 ||
        read_u32(fp, &num_layers) < 0 ||
        read_u32(fp, &hidden_dim) < 0 ||
        read_u32(fp, &inter_dim)  < 0 ||
        read_u32(fp, &vocab_size) < 0 ||
        read_u32(fp, &max_seq_len)< 0 ||
        read_u32(fp, &n_heads)    < 0 ||
        read_u32(fp, &reserved)   < 0) {
        fclose(fp); RT_FAIL(RT_EIO, "header read failed");
    }

    if (version != DBERT_VERSION_SST2 && version != DBERT_VERSION_INT8) {
        fprintf(stderr, "[distilbert_io] file version %u does not contain SST-2 head\n",
                version);
        fclose(fp); RT_FAIL(RT_EINVAL, "no SST-2 head in this file");
    }

    uint64_t H = hidden_dim;
    uint64_t I = inter_dim;
    uint64_t V = vocab_size;
    uint64_t S = max_seq_len;
    uint64_t N = num_layers;

    /* Skip embeddings */
    if (skip_f32(fp, V * H) < 0) goto cls_io_err;  /* word_emb */
    if (skip_f32(fp, S * H) < 0) goto cls_io_err;  /* pos_emb  */
    if (skip_f32(fp, H)     < 0) goto cls_io_err;  /* ln_gamma */
    if (skip_f32(fp, H)     < 0) goto cls_io_err;  /* ln_beta  */

    /* Skip transformer layers: format depends on version */
    int cls_v3 = (version == DBERT_VERSION_INT8);
    for (uint64_t i = 0; i < N; ++i) {
        if (cls_v3) {
            /* int8 weights + float32 scale, float32 biases/LN */
            if (skip_i8_weight(fp, H*H) < 0) goto cls_io_err; /* q_w  */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* q_b  */
            if (skip_i8_weight(fp, H*H) < 0) goto cls_io_err; /* k_w  */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* k_b  */
            if (skip_i8_weight(fp, H*H) < 0) goto cls_io_err; /* v_w  */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* v_b  */
            if (skip_i8_weight(fp, H*H) < 0) goto cls_io_err; /* o_w  */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* o_b  */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* sa_ln_w */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* sa_ln_b */
            if (skip_i8_weight(fp, H*I) < 0) goto cls_io_err; /* ffn_up_w */
            if (skip_f32(fp, I)         < 0) goto cls_io_err; /* ffn_up_b */
            if (skip_i8_weight(fp, I*H) < 0) goto cls_io_err; /* ffn_dn_w */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* ffn_dn_b */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* out_ln_w */
            if (skip_f32(fp, H)         < 0) goto cls_io_err; /* out_ln_b */
        } else {
            if (skip_f32(fp, H * H) < 0) goto cls_io_err; /* q_w  */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* q_b  */
            if (skip_f32(fp, H * H) < 0) goto cls_io_err; /* k_w  */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* k_b  */
            if (skip_f32(fp, H * H) < 0) goto cls_io_err; /* v_w  */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* v_b  */
            if (skip_f32(fp, H * H) < 0) goto cls_io_err; /* o_w  */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* o_b  */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* sa_ln_w */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* sa_ln_b */
            if (skip_f32(fp, H * I) < 0) goto cls_io_err; /* ffn_up_w */
            if (skip_f32(fp, I)     < 0) goto cls_io_err; /* ffn_up_b */
            if (skip_f32(fp, I * H) < 0) goto cls_io_err; /* ffn_dn_w */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* ffn_dn_b */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* out_ln_w */
            if (skip_f32(fp, H)     < 0) goto cls_io_err; /* out_ln_b */
        }
    }

    /* Now at classification head section */
    if (read_f32_bulk(fp, head->pre_cls_w->data, H * H) < 0) goto cls_io_err;
    if (read_f32_bulk(fp, head->pre_cls_b->data, H)     < 0) goto cls_io_err;
    if (read_f32_bulk(fp, head->cls_w->data,     H * 2) < 0) goto cls_io_err;
    if (read_f32_bulk(fp, head->cls_b->data,     2)     < 0) goto cls_io_err;

    fclose(fp);
    printf("[distilbert_io] SST-2 classification head loaded\n");
    return 0;

cls_io_err:
    fclose(fp);
    fprintf(stderr, "[distilbert_io] read error while loading SST-2 head\n");
    RT_FAIL(RT_EIO, "cls weight read failed");
}
