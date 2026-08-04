#include "read_safetensors.h"
#include "gpt2_io.h"
#include "gpt2.h"

void test_prefill(struct tk_gpt2* model, struct tk_rt_ctx* ctx, struct tk_gpt2_config cfg) {
	int prompt_ids[] = {464, 3797, 3332, 319, 262};
	int prompt_len = 5;
	struct tk_tensor* input = NULL;
	int shape[1] = { prompt_len };
	tk_tensor_alloc(ctx->meta_arena, TK_I32, shape, 1, &input);
	memcpy(input->data, prompt_ids, sizeof(int32_t) * prompt_len);

	struct tk_tensor* logits_dry = NULL;
	tk_rt_ctx_set_mode(ctx, RT_DRYRUN);
	int rc_dry = tk_gpt2_forward(ctx, model, input, &logits_dry, TK_GPT2_PREFILL);
	printf("dry run rc=%d\n", rc_dry);
	if (rc_dry != 0) {
		rt_err_print(stderr); 
    	fprintf(stderr, "dry run failed, aborting\n");
    	return;
	}

	tk_rt_prepare(ctx);

	ctx->ws->cur_offset = 0;
	struct tk_tensor* logits = NULL;
	int rc = tk_gpt2_forward(ctx, model, input, &logits, TK_GPT2_PREFILL);
	printf("rc=%d\n", rc);

	float* last_row = (float*)logits->data + (size_t)(prompt_len - 1) * cfg.vocab_size;
	printf("logits[0..7]: ");
	for (int i = 0; i < 8; ++i) printf("%f ", last_row[i]);
	printf("\n");
}

void test_generate(struct tk_gpt2* model, struct tk_rt_ctx* ctx, struct tk_gpt2_config cfg) {
    int prompt_ids[] = {464, 3797, 3332, 319, 262};
    int prompt_len = 5;

    struct tk_tensor* prompt = NULL;
    int shape[1] = { prompt_len };
    tk_tensor_alloc(ctx->meta_arena, TK_I32, shape, 1, &prompt);
    memcpy(prompt->data, prompt_ids, sizeof(int32_t) * prompt_len);

    int32_t* out_tokens = NULL;
    int out_count = 0;
    int rc = tk_gpt2_generate(ctx, model, prompt, /* max_new_tokens */ 5,
                               /* eos_token_id */ 50256, &out_tokens, &out_count);
    printf("generate rc=%d, tokens: ", rc);
    for (int i = 0; i < out_count; ++i) printf("%d ", out_tokens[i]);
    printf("\n");
}

int main(int argc, char* argv[]) {
	const char* st_path  = (argc > 1) ? argv[1] : "data/gpt2/gpt2_model.safetensors";
    const char* cfg_path = (argc > 2) ? argv[2] : "data/gpt2/gpt2_config.json";

    struct arena root_arena;
    arena_init(&root_arena);
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&root_arena);
    ctx->compute_dtype = TK_F32;

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

	test_prefill(model, ctx, cfg);
	test_generate(model, ctx, cfg);

	tk_rt_ctx_destroy(ctx);
	arena_destroy(&root_arena);

    return 0;
}

