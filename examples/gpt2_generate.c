#include "read_safetensors.h"
#include "gpt2_io.h"
#include "gpt2.h"

static int* prompt_str_to_arr(const char* prompt_str) {
    int token_id;
    char* str = strdup(prompt_str);
    


    free(str);
}

void run_generate(struct tk_gpt2* model, struct tk_rt_ctx* ctx, struct tk_gpt2_config cfg) {
    int prompt_ids[] = {464, 3797, 3332, 319, 262};
    int prompt_len = 5;

    ctx->manager = tk_prof_create(32, 1024);
    if (!ctx->manager) {
        fprintf(stderr, "Failed to create profiler manager\n");
        return;
    }
    tk_prof_bind_manager(ctx->manager);

    struct tk_tensor* prompt = NULL;
    int shape[1] = { prompt_len };
    tk_tensor_alloc(ctx->meta_arena, TK_I32, shape, 1, &prompt);
    memcpy(prompt->data, prompt_ids, sizeof(int32_t) * prompt_len);

    int32_t* out_tokens = NULL;
    int out_count = 0;
    int max_new_token = 50;
    int rc = tk_gpt2_generate(ctx, model, prompt, /* max_new_tokens */ max_new_token,
                               /* eos_token_id */ 50256, &out_tokens, &out_count);
    printf("generate rc=%d, tokens: ", rc);
    for (int i = 0; i < out_count; ++i) printf("%d ", out_tokens[i]);
    printf("\n");

    tk_prof_summarize(ctx->manager);
}

int main(int argc, char* argv[]) {

    // default configurations
	const char* st_path  = (argc > 1) ? argv[1] : "data/gpt2/gpt2_model.safetensors";
    const char* cfg_path = (argc > 2) ? argv[2] : "model_configs/gpt2_config.json";
    const char* prompt_str = "464, 3797, 3332, 319, 262";
    int omp_threads = 4;
    int use_prof = 0;

    for ( int i = 0 ; i < argc ; ++i ) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) st_path = argv[++i];
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) cfg_path = argv[++i];
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt_str = argv[++i];
        if (strcmp(argv[i], "--max_token") == 0 && i + 1 < argc) max_token = argv[++i];
        if (strcmp(argv[i], "--omp_thread") == 0 && i + 1 < argc) omp_threads = argv[++i];
        if (strcmp(argv[i], "--use_prof") == 0 && i + 1 < argc) use_prof = argv[++i]; 
    }

    struct arena root_arena;
    arena_init(&root_arena);
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&root_arena);
    ctx->compute_dtype = TK_F32;
    ctx->use_prof = use_prof;

    struct tk_gpt2_config cfg;
    if (tk_gpt2_config_from_json(cfg_path, &cfg) != 0) {
        fprintf(stderr, "config parse failed\n"); return 1;
    }
    printf("num_layers=%d hidden_dim=%d n_heads=%d vocab_size=%d max_seq_len=%d\n",
           cfg.num_layers, cfg.hidden_dim, cfg.n_heads, cfg.vocab_size, cfg.max_seq_len);

	cfg.seq_length = 5;

    struct tk_gpt2* model = NULL;
    RT_CHECK(tk_gpt2_config(ctx, cfg, &model));
    RT_CHECK(tk_gpt2_alloc(ctx, cfg, model));

    int rc = tk_gpt2_safetensors_load(st_path, cfg_path, &cfg, model);
    printf("load rc=%d\n", rc);

	run_generate(model, ctx, cfg);
	tk_rt_ctx_destroy(ctx);
	arena_destroy(&root_arena);

    return 0;
}

