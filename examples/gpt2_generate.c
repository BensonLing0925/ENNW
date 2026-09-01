#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "read_safetensors.h"
#include "gpt2_io.h"
#include "gpt2.h"

/* Parse "464 3797 3332" or "464,3797,3332" into a token id array.
 * Returns a malloc'd array; caller frees. */
static int* parse_prompt(const char* prompt_str, int* out_count) {
    char* str = strdup(prompt_str);
    if (!str) return NULL;

    int cap = 16, n = 0;
    int* ids = malloc(sizeof(int) * cap);
    if (!ids) { free(str); return NULL; }

    for (char* tok = strtok(str, " ,"); tok; tok = strtok(NULL, " ,")) {
        if (n == cap) {
            cap *= 2;
            int* grown = realloc(ids, sizeof(int) * cap);
            if (!grown) { free(ids); free(str); return NULL; }
            ids = grown;
        }
        ids[n++] = atoi(tok);
    }

    free(str);
    *out_count = n;
    return ids;
}

static void run_generate(struct tk_gpt2* model, struct tk_rt_ctx* ctx,
                         const int* prompt_ids, int prompt_len, int max_new_tokens) {
    struct tk_tensor* prompt = NULL;
    int shape[1] = { prompt_len };
    tk_tensor_alloc(ctx->meta_arena, TK_I32, shape, 1, &prompt);
    memcpy(prompt->data, prompt_ids, sizeof(int32_t) * prompt_len);

    int32_t* out_tokens = NULL;
    int out_count = 0;
    int rc = tk_gpt2_generate(ctx, model, prompt, max_new_tokens,
                              /* eos_token_id */ 50256, &out_tokens, &out_count);

    printf("generate rc=%d, tokens: ", rc);
    for (int i = 0; i < out_count; ++i) printf("%d ", out_tokens[i]);
    printf("\n");

    if (ctx->use_prof) tk_prof_summarize(ctx->manager);
}

int main(int argc, char* argv[]) {
    const char* st_path    = "data/gpt2/gpt2_model.safetensors";
    const char* cfg_path   = "model_configs/gpt2_config.json";
    const char* prompt_str = "464 3797 3332 319 262";
    int max_new_tokens = 50;
    int omp_threads    = 4;
    int use_prof       = 0;

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--model")   == 0 && i + 1 < argc) st_path        = argv[++i];
        else if (strcmp(argv[i], "--config")  == 0 && i + 1 < argc) cfg_path       = argv[++i];
        else if (strcmp(argv[i], "--prompt")  == 0 && i + 1 < argc) prompt_str     = argv[++i];
        else if (strcmp(argv[i], "--tokens")  == 0 && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) omp_threads    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--profile") == 0)                 use_prof       = 1;
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 1;
        }
    }

#ifdef _OPENMP
    omp_set_num_threads(omp_threads);
#endif

    int prompt_len = 0;
    int* prompt_ids = parse_prompt(prompt_str, &prompt_len);
    if (!prompt_ids || prompt_len == 0) {
        fprintf(stderr, "failed to parse prompt\n");
        return 1;
    }

    struct arena root_arena;
    arena_init(&root_arena);

    struct tk_rt_ctx_config ctx_config = {
        .use_int8 = 0,
        .use_prof = use_prof,
        .use_graph_optimize = 1,
        .graph_capacity = 1024,
    };
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create_config(&root_arena, ctx_config);
    ctx->compute_dtype = TK_F32;

    struct tk_gpt2_config cfg;
    if (tk_gpt2_config_from_json(cfg_path, &cfg) != 0) {
        fprintf(stderr, "config parse failed\n");
        free(prompt_ids);
        return 1;
    }
    cfg.seq_length = prompt_len;

    struct tk_gpt2* model = NULL;
    RT_CHECK(tk_gpt2_config(ctx, cfg, &model));
    RT_CHECK(tk_gpt2_alloc(ctx, cfg, model));
    RT_CHECK(tk_gpt2_safetensors_load(st_path, cfg_path, &cfg, model));

    run_generate(model, ctx, prompt_ids, prompt_len, max_new_tokens);

    free(prompt_ids);
    tk_rt_ctx_destroy(ctx);
    arena_destroy(&root_arena);
    return 0;
}
