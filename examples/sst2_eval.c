/*
 * sst2_eval.c
 *
 * Evaluate DistilBERT SST-2 on a batch binary produced by
 * tools/tokenize_sst2.py.
 *
 * Usage:
 *   sst2_eval --weights distilbert_sst2.bin --data sst2_val.bin
 *   sst2_eval --weights distilbert_sst2.bin --data sst2_val.bin --progress
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
    #include <omp.h>
    #define GET_TIME() omp_get_wtime()
#else
    #define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)

#endif
#include "arena.h"
#include "rt_context.h"
#include "tensor.h"
#include "distilbert.h"
#include "distilbert_cls.h"
#include "distilbert_io.h"
#include "rt_error.h"
#include "rt_context.h"
#include "rt_graph.h"

/* ------------------------------------------------------------------ */
/* Batch binary reader                                                  */
/* ------------------------------------------------------------------ */

struct sst2_batch {
    int32_t  num_samples;
    int32_t  seq_len;
    int32_t* labels;      /* [num_samples] */
    int32_t* token_ids;   /* [num_samples, seq_len] row-major */
};

static int sst2_batch_load(const char* path, struct sst2_batch* out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open '%s'\n", path); return -1; }

    if (fread(&out->num_samples, sizeof(int32_t), 1, fp) != 1 ||
        fread(&out->seq_len,     sizeof(int32_t), 1, fp) != 1 ||
        out->num_samples <= 0 || out->seq_len <= 0) {
        fclose(fp); fprintf(stderr, "Bad batch header\n"); return -1;
    }

    size_t n = (size_t)out->num_samples;
    size_t s = (size_t)out->seq_len;

    out->labels    = malloc(n * sizeof(int32_t));
    out->token_ids = malloc(n * s * sizeof(int32_t));
    if (!out->labels || !out->token_ids) {
        fclose(fp); fprintf(stderr, "OOM\n"); return -1;
    }

    for (size_t i = 0; i < n; ++i) {
        if (fread(&out->labels[i],             sizeof(int32_t), 1, fp) != 1 ||
            fread(out->token_ids + i * s, sizeof(int32_t), s, fp) != s) {
            fclose(fp);
            fprintf(stderr, "Read error at sample %d\n", (int)i);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

static void sst2_batch_free(struct sst2_batch* b) {
    free(b->labels);
    free(b->token_ids);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --weights <sst2.bin> --data <sst2_val.bin> [--limit N] [--progress]\n",
            prog);
}

int main(int argc, char* argv[]) {

    const char* weights_path  = NULL;
    const char* data_path     = NULL;
    int         show_progress = 0;
    int         limit         = 0;   /* 0 = no limit */
    int         use_int8      = 0;
    int         use_prof      = 0;

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--weights")  == 0 && i + 1 < argc) weights_path = argv[++i];
        else if (strcmp(argv[i], "--data")     == 0 && i + 1 < argc) data_path    = argv[++i];
        else if (strcmp(argv[i], "--limit")    == 0 && i + 1 < argc) limit = atoi(argv[++i]);
        else if (strcmp(argv[i], "--progress") == 0) show_progress = 1;
        else if (strcmp(argv[i], "--int8")     == 0) use_int8 = 1;
        else if (strcmp(argv[i], "--use_prof") == 0) use_prof = 1;
        else if (strcmp(argv[i], "--help")     == 0) { print_usage(argv[0]); return 0; }
    }

    if (!weights_path || !data_path) {
        fprintf(stderr, "Error: --weights and --data are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ---- Load batch data ---- */
    struct sst2_batch batch = {0};
    if (sst2_batch_load(data_path, &batch) != 0) return 1;

    int num_eval = (limit > 0 && limit < batch.num_samples) ? limit : batch.num_samples;
    printf("Dataset: %d samples, seq_len=%d  (evaluating %d)\n",
           batch.num_samples, batch.seq_len, num_eval);

    /* ---- Runtime context ---- */
    struct arena root_arena;
    arena_init(&root_arena);

    /* context config */
    struct tk_rt_ctx_config ctx_config = {
        .use_int8 = use_int8,
        .use_prof = use_prof,
        .use_graph_optimize = 1,
        .graph_capacity = 1024,
    };

    struct tk_rt_ctx* ctx = tk_runtime_ctx_create_config(&root_arena, ctx_config);
    // struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&root_arena);
    ctx->compute_dtype = TK_F32;

    /* ---- Model config (seq_len from file) ---- */
    struct tk_distilbert_config cfg = {
        .vocab_size      = 30522,
        .max_seq_len     = 512,
        .num_layers      = 6,
        .seq_length      = batch.seq_len,
        .hidden_dim      = 768,
        .n_heads         = 12,
        .inter_dim       = 0,
        .use_qkv_bias    = 1,
        .use_o_proj      = 1,
        .use_o_proj_bias = 1,
        .use_ffn_bias    = 1,
    };

    /* ---- Build model ---- */
    struct tk_distilbert* model = NULL;
    if (tk_distilbert_config(ctx, cfg, &model) != 0) {
        fprintf(stderr, "distilbert_config failed\n"); return 1;
    }
    if (tk_distilbert_alloc(ctx, cfg, model) != 0) {
        fprintf(stderr, "distilbert_alloc failed\n"); return 1;
    }

    struct tk_sst2_cls_head* cls_head = tk_sst2_cls_create(ctx);
    tk_sst2_cls_alloc(ctx, cls_head, cfg.hidden_dim);

    printf("Weights:  %.1f MB\n",
           (double)arena_get_total_used(ctx->data_arena) / (1024.0 * 1024.0));

    /* ---- Load weights ---- */
    uint32_t file_version = 0;
    if (tk_distilbert_load_weights(weights_path, &cfg, model, &file_version) != 0) {
        fprintf(stderr, "Failed to load base weights\n"); return 1;
    }
    if (file_version != DBERT_VERSION_SST2 && file_version != DBERT_VERSION_INT8) {
        fprintf(stderr, "Weight file is VERSION %u, expected %u or %u (SST-2).\n",
                file_version, DBERT_VERSION_SST2, DBERT_VERSION_INT8);
        return 1;
    }
    if (use_int8 && file_version != DBERT_VERSION_INT8) {
        fprintf(stderr, "Error: --int8 flag set but weight file is version %u (need %u).\n"
                "  Re-export: python tools/export_distilbert_weights.py --sst2 --int8 ...\n",
                file_version, DBERT_VERSION_INT8);
        return 1;
    }
    if (tk_distilbert_load_cls_weights(weights_path, &cfg, cls_head) != 0) {
        fprintf(stderr, "Failed to load cls head weights\n"); return 1;
    }

    /* ---- Allocate reusable input tensor ---- */
    int input_shape[1] = { batch.seq_len };
    struct tk_tensor* input_ids = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_I32, input_shape, 1, &input_ids);

    /* ---- Dry run: size workspace once ---- */
    memcpy(input_ids->data, batch.token_ids, (size_t)batch.seq_len * sizeof(int32_t));
    ctx->rt_type = RT_DRYRUN;
    struct tk_tensor *hidden = NULL, *logits = NULL;
    if (tk_distilbert_forward(ctx, model, input_ids, &hidden) != 0 ||
        tk_sst2_cls_forward(ctx, cls_head, hidden, &logits)   != 0) {
        fprintf(stderr, "dry run failed\n"); return 1;
    }
    printf("Workspace: %.2f MB\n\n",
           (double)ctx->ws->peak_offset / (1024.0 * 1024.0));

    /* ---- static graph record test ---- */
    tk_rt_prepare(ctx);

    /* ---- Evaluation loop ---- */
    int correct = 0;
    int tp = 0, tn = 0, fp_count = 0, fn_count = 0;

    double t_start = GET_TIME();

    for (int s = 0; s < num_eval; ++s) {

        double t_sample_start = GET_TIME();

        memcpy(input_ids->data,
               batch.token_ids + (size_t)s * (size_t)batch.seq_len,
               (size_t)batch.seq_len * sizeof(int32_t));

        ctx->rt_type        = RT_INFERENCE;
        ctx->ws->cur_offset = 0;
        hidden = NULL;
        logits = NULL;

        if (tk_distilbert_forward(ctx, model, input_ids, &hidden) != 0 ||
            tk_sst2_cls_forward(ctx, cls_head, hidden, &logits)   != 0) {
            fprintf(stderr, "inference failed at sample %d\n", s);
            rt_err_print(stderr);
            return 1;
        }
        
        // threads_print(ctx->manager);

        /* argmax of softmax output [1, 2] */
        int pred = 0;
        TK_DISPATCH_TYPES(logits->dtype, "sst2_eval", {
            scalar_t* p = (scalar_t*)logits->data;
            pred = (p[1] > p[0]) ? 1 : 0;
        });

        int label = batch.labels[s];
        if (pred == label) {
            ++correct;
            if (label == 1) ++tp; else ++tn;
        } else {
            if (label == 1) ++fn_count; else ++fp_count;
        }

        double sample_ms = 1000.0 * (GET_TIME() - t_sample_start);

        if (show_progress || (s + 1) % 10 == 0) {
            double elapsed  = (double)(GET_TIME() - t_start);
            double ms_per   = 1000.0 * elapsed / (s + 1);
            double eta_s    = ms_per * (num_eval - s - 1) / 1000.0;
            printf("  [%4d/%4d]  acc: %.1f%%  last: %.0fms  avg: %.0fms  ETA: %.0fs\n",
                   s + 1, num_eval,
                   100.0 * correct / (s + 1),
                   sample_ms, ms_per, eta_s);
        }
    }
    
    // threads_print(ctx->manager);

    double t_end = GET_TIME();
    double total_s  = (double)(t_end - t_start);
    double ms_per   = 1000.0 * total_s / num_eval;

    /* ---- Results ---- */
    double acc = 100.0 * correct / num_eval;
    printf("\n=== SST-2 Evaluation Results ===\n");
    printf("Samples : %d\n",  num_eval);
    printf("Correct : %d\n",  correct);
    printf("Accuracy: %.2f%%\n", acc);
    printf("\nTiming:\n");
    printf("  Total  : %.2f s\n",    total_s);
    printf("  Per sample: %.1f ms\n", ms_per);
    if (num_eval < batch.num_samples)
        printf("  Full val est: %.1f min\n",
               ms_per * batch.num_samples / 60000.0);
    printf("\nConfusion matrix:\n");
    printf("  TP (pos->pos): %d\n", tp);
    printf("  TN (neg->neg): %d\n", tn);
    printf("  FP (neg->pos): %d\n", fp_count);
    printf("  FN (pos->neg): %d\n", fn_count);

    /*
#ifdef PROF
    tk_prof_view_run(ctx->manager);
#endif
    */

    sst2_batch_free(&batch);
    tk_rt_ctx_destroy(ctx);
    
    arena_destroy(&root_arena);
    return 0;
}
