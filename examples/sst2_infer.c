/*
 * sst2_infer.c
 *
 * SST-2 sentiment inference using DistilBertForSequenceClassification.
 *
 * Token IDs binary format (produced by tools/tokenize_sst2.py):
 *   int32_t  seq_length
 *   int32_t  token_ids[seq_length]
 *
 * Usage:
 *   sst2_infer --weights distilbert_sst2.bin --tokens tokens.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../mem/arena.h"
#include "../src/runtime/rt_context.h"
#include "../src/ops/tensor.h"
#include "../src/modules/transformer/distilbert/distilbert.h"
#include "../src/modules/transformer/distilbert/distilbert_cls.h"
#include "../weightio/distilbert_io.h"
#include "../src/error/rt_error.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --weights <distilbert_sst2.bin> --tokens <tokens.bin>\n"
            "  %s --weights <distilbert_sst2.bin>  (uses built-in demo sentence)\n",
            prog, prog);
}

/* Read token IDs from binary file: int32 seq_len, then int32[seq_len] ids. */
static int32_t* load_tokens(const char* path, int* seq_len_out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open '%s'\n", path); return NULL; }

    int32_t seq_len;
    if (fread(&seq_len, sizeof(int32_t), 1, fp) != 1 || seq_len <= 0) {
        fclose(fp); return NULL;
    }
    int32_t* ids = malloc((size_t)seq_len * sizeof(int32_t));
    if (!ids) { fclose(fp); return NULL; }
    if ((int)fread(ids, sizeof(int32_t), (size_t)seq_len, fp) != seq_len) {
        free(ids); fclose(fp); return NULL;
    }
    fclose(fp);
    *seq_len_out = (int)seq_len;
    return ids;
}

int main(int argc, char* argv[]) {

    const char* weights_path = NULL;
    const char* tokens_path  = NULL;
    int         use_int8     = 0;
    int         use_prof     = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
            weights_path = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            tokens_path = argv[++i];
        else if (strcmp(argv[i], "--int8") == 0)
            use_int8 = 1;
        else if (strcmp(argv[i], "--use_prof") == 0)
            use_prof = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    if (!weights_path) {
        fprintf(stderr, "Error: --weights is required.\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ---- Token IDs ---- */
    int32_t demo_ids[] = { 101, 1045, 2293, 2023, 3185, 999, 102 };
    int32_t* token_data = demo_ids;
    int seq_len = (int)(sizeof(demo_ids) / sizeof(demo_ids[0]));
    int owned_tokens = 0;

    if (tokens_path) {
        int32_t* loaded = load_tokens(tokens_path, &seq_len);
        if (!loaded) {
            fprintf(stderr, "Failed to load tokens from '%s'\n", tokens_path);
            return 1;
        }
        token_data  = loaded;
        owned_tokens = 1;
    } else {
        printf("No --tokens given, using demo: \"I love this movie!\"\n");
    }

    /* ---- Runtime context ---- */
    struct arena root_arena;
    arena_init(&root_arena);
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&root_arena);
    ctx->compute_dtype = TK_F32;
    ctx->use_int8      = use_int8;   /* must be set before tk_distilbert_alloc */
    ctx->use_prof      = use_prof;

    /* ---- DistilBERT Base config ---- */
    struct tk_distilbert_config cfg = {
        .vocab_size      = 30522,
        .max_seq_len     = 512,
        .num_layers      = 6,
        .seq_length      = seq_len,
        .hidden_dim      = 768,
        .n_heads         = 12,
        .inter_dim       = 0,
        .use_qkv_bias    = 1,
        .use_o_proj      = 1,
        .use_o_proj_bias = 1,
        .use_ffn_bias    = 1,
    };

    /* ---- Build base model ---- */
    struct tk_distilbert* model = NULL;
    if (tk_distilbert_config(ctx, cfg, &model) != 0) {
        fprintf(stderr, "distilbert_config failed\n"); return 1;
    }
    if (tk_distilbert_alloc(ctx, cfg, model) != 0) {
        fprintf(stderr, "distilbert_alloc failed\n"); return 1;
    }

    /* ---- Build classification head ---- */
    struct tk_sst2_cls_head* cls_head = tk_sst2_cls_create(ctx);
    tk_sst2_cls_alloc(ctx, cls_head, cfg.hidden_dim);

    printf("Weights allocated: %.1f MB\n",
           (double)arena_get_total_used(ctx->data_arena) / (1024.0 * 1024.0));

    /* ---- Load weights ---- */
    uint32_t file_version = 0;
    printf("Loading weights from '%s'...\n", weights_path);
    if (tk_distilbert_load_weights(weights_path, &cfg, model, &file_version) != 0) {
        fprintf(stderr, "Failed to load base weights\n"); return 1;
    }
    if (file_version != DBERT_VERSION_SST2 && file_version != DBERT_VERSION_INT8) {
        fprintf(stderr,
                "Warning: file version %u is not SST-2 (expected %u or %u).\n"
                "  Export with: python tools/export_distilbert_weights.py "
                "--sst2 distilbert_sst2.bin\n",
                file_version, DBERT_VERSION_SST2, DBERT_VERSION_INT8);
        return 1;
    }
    if (use_int8 && file_version != DBERT_VERSION_INT8) {
        fprintf(stderr, "Error: --int8 flag set but weight file is version %u (need version %u).\n"
                "  Re-export with: python tools/export_distilbert_weights.py --sst2 --int8 ...\n",
                file_version, DBERT_VERSION_INT8);
        return 1;
    }
    if (tk_distilbert_load_cls_weights(weights_path, &cfg, cls_head) != 0) {
        fprintf(stderr, "Failed to load classification head weights\n"); return 1;
    }

    /* ---- Build input tensor ---- */
    int input_shape[1] = { seq_len };
    struct tk_tensor* input_ids = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_I32, input_shape, 1, &input_ids);
    memcpy(input_ids->data, token_data, (size_t)seq_len * sizeof(int32_t));
    if (owned_tokens) free(token_data);

    /* ---- Dry run: size workspace ---- */
    ctx->rt_type = RT_DRYRUN;
    struct tk_tensor* hidden = NULL;
    struct tk_tensor* logits = NULL;
    if (tk_distilbert_forward(ctx, model, input_ids, &hidden) != 0) {
        fprintf(stderr, "dry run (encoder) failed\n"); return 1;
    }
    if (tk_sst2_cls_forward(ctx, cls_head, hidden, &logits) != 0) {
        fprintf(stderr, "dry run (cls head) failed\n"); return 1;
    }
    printf("Peak workspace: %.2f MB\n",
           (double)ctx->ws->peak_offset / (1024.0 * 1024.0));

    /* ---- Inference ---- */
    ctx->rt_type        = RT_INFERENCE;
    ctx->ws->cur_offset = 0;
    hidden = NULL;
    logits = NULL;

    if (tk_distilbert_forward(ctx, model, input_ids, &hidden) != 0) {
        fprintf(stderr, "encoder forward failed\n");
        rt_err_print(stderr);
        return 1;
    }
    if (tk_sst2_cls_forward(ctx, cls_head, hidden, &logits) != 0) {
        fprintf(stderr, "cls head forward failed\n");
        rt_err_print(stderr);
        return 1;
    }

    /* ---- Print result ---- */
    TK_DISPATCH_TYPES(logits->dtype, "sst2_print", {
        scalar_t* p = (scalar_t*)logits->data;
        double neg = (double)p[0];
        double pos = (double)p[1];
        printf("\n--- Sentiment Classification ---\n");
        printf("Negative: %.4f\n", neg);
        printf("Positive: %.4f\n", pos);
        printf("Result:   %s (confidence %.1f%%)\n",
               pos > neg ? "POSITIVE" : "NEGATIVE",
               100.0 * (pos > neg ? pos : neg));
    });

    /* ---- Cleanup ---- */
    arena_destroy(ctx->data_arena);
    arena_destroy(ctx->meta_arena);
    arena_destroy(&root_arena);
    return 0;
}
