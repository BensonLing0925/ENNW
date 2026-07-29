#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../mem/arena.h"
#include "../src/runtime/rt_context.h"
#include "../src/ops/tensor.h"
#include "../src/modules/transformer/distilbert/distilbert.h"
#include "../weightio/distilbert_io.h"
#include "../src/error/rt_error.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s                          -- run with random weights\n"
            "  %s --weights distilbert.bin -- load real weights\n"
            "  %s --weights distilbert.bin --verify -- print per-layer CLS hidden states\n",
            prog, prog, prog);
}

static void print_verify_layer(int idx, struct tk_tensor* hidden) {
    printf("verify_layer %d:", idx);
    TK_DISPATCH_TYPES(hidden->dtype, "verify_print", {
        scalar_t* d = (scalar_t*)hidden->data;
        for (int i = 0; i < 8; i++)
            printf(" %.10f", (double)d[i]);
    });
    printf("\n");
    fflush(stdout);
}

int main(int argc, char* argv[]) {

    const char* weights_path = NULL;
    int verify_mode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
            weights_path = argv[++i];
        else if (strcmp(argv[i], "--verify") == 0)
            verify_mode = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* ---- Runtime context ---- */
    struct arena root_arena;
    arena_init(&root_arena);

    /* Verify mode: skip graph optimisation, use exec vtable directly */
    struct tk_rt_ctx* ctx;
    if (verify_mode) {
        struct tk_rt_ctx_config vcfg = { .use_int8 = 0,
                                         .use_graph_optimize = 0,
                                         .graph_capacity = 0 };
        ctx = tk_runtime_ctx_create_config(&root_arena, vcfg);
        ctx->rt_type = RT_INFERENCE;
    } else {
        ctx = tk_runtime_ctx_create(&root_arena);
    }

    /* ---- DistilBERT Base config ---- */
    struct tk_distilbert_config cfg = {
        .vocab_size      = 30522,
        .max_seq_len     = 512,
        .num_layers      = 6,
        .seq_length      = 16,   /* tokens in this inference call */
        .hidden_dim      = 768,
        .n_heads         = 12,
        .inter_dim       = 0,    /* 0 = auto (4 * hidden_dim = 3072) */
        .use_qkv_bias    = 1,
        .use_o_proj      = 1,
        .use_o_proj_bias = 1,
        .use_ffn_bias    = 1,
    };

    /* ---- Set compute dtype to match the exported weights (float32) ---- */
    ctx->compute_dtype = TK_F32;

    /* ---- Build model ---- */
    struct tk_distilbert* model = NULL;
    if (tk_distilbert_config(ctx, cfg, &model) != 0) {
        fprintf(stderr, "distilbert_config failed\n");
        return 1;
    }
    if (tk_distilbert_alloc(ctx, cfg, model) != 0) {
        fprintf(stderr, "distilbert_alloc failed\n");
        return 1;
    }

    printf("Weights allocated: %.1f MB\n",
           (double)arena_get_total_used(ctx->data_arena) / (1024.0 * 1024.0));

    /* ---- Optionally load real weights ---- */
    if (weights_path) {
        printf("Loading weights from '%s'...\n", weights_path);
        uint32_t file_ver = 0;
        if (tk_distilbert_load_weights(weights_path, &cfg, model, &file_ver) != 0) {
            fprintf(stderr, "Failed to load weights\n");
            return 1;
        }
    } else {
        printf("No --weights given, using random initialisation.\n");
    }

    /* ---- Sample input: 16 token IDs ---- */
    /* [CLS]=101  hello=7592  world=2088  [SEP]=102  <pad>=0 ... */
    int32_t token_data[16] = { 101, 7592, 2088, 102, 0, 0, 0, 0,
                                  0,    0,    0,   0, 0, 0, 0, 0 };
    int input_shape[1] = { 16 };
    struct tk_tensor* input_ids = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_I32, input_shape, 1, &input_ids);
    memcpy(input_ids->data, token_data, sizeof(token_data));

    if (verify_mode) {
        /* ---- Verify mode: step through each layer, print CLS hidden state ---- */
        struct tk_tensor* hidden = NULL;
        if (tk_distilbert_emb_forward(ctx, model->emb, input_ids, &hidden) != 0) {
            fprintf(stderr, "emb forward failed\n"); return 1;
        }
        print_verify_layer(0, hidden);  /* 0 = after embedding */

        for (int i = 0; i < model->num_layers; ++i) {
            if (tk_tf_block_forward(ctx, model->blocks[i]->base, hidden, ctx->ops->attention) != 0) {
                fprintf(stderr, "block %d forward failed\n", i); return 1;
            }
            print_verify_layer(i + 1, hidden);  /* 1..6 = after each transformer layer */
        }

        arena_destroy(ctx->data_arena);
        arena_destroy(ctx->meta_arena);
        arena_destroy(&root_arena);
        return 0;
    }

    /* ---- Dry run: size the workspace ---- */
    ctx->rt_type = RT_DRYRUN;
    struct tk_tensor* output = NULL;
    if (tk_distilbert_forward(ctx, model, input_ids, &output) != 0) {
        fprintf(stderr, "dry run failed\n");
        return 1;
    }
    printf("Peak workspace:    %.2f MB\n",
           (double)ctx->ws->peak_offset / (1024.0 * 1024.0));

    /* ---- Inference ---- */
    ctx->rt_type        = RT_INFERENCE;
    ctx->ws->cur_offset = 0;
    output = NULL;
    if (tk_distilbert_forward(ctx, model, input_ids, &output) != 0) {
        fprintf(stderr, "inference failed\n");
        rt_err_print(stderr);
        return 1;
    }

    /* ---- Print result ---- */
    printf("Output shape:      [%d, %d]\n", output->shape[0], output->shape[1]);
    printf("[CLS] token hidden state (first 8 dims):\n[");
    TK_DISPATCH_TYPES(output->dtype, "distilbert_print", {
        scalar_t* out = (scalar_t*)output->data;
        for (int i = 0; i < 8; ++i)
            printf("%.6f%s", (double)out[i], i < 7 ? ", " : "]\n");
    });

    /* ---- Cleanup ---- */
    arena_destroy(ctx->data_arena);
    arena_destroy(ctx->meta_arena);
    arena_destroy(&root_arena);
    return 0;
}
