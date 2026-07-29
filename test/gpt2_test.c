#include "../src/modules/transformer/gpt2/gpt2.h"
#include "../mem/arena.h"

void test_emb_build_gpt2();
void test_generate_gpt2();
void test_generate_no_cache_gpt2();

int main() {
	test_emb_build_gpt2();
	// test_generate_gpt2();
	test_generate_no_cache_gpt2();
}

void test_emb_build_gpt2() {
	struct arena root_arena;
	arena_init(&root_arena);

    struct tk_rt_ctx_config ctx_config = {
        .use_int8 = 0,
        .use_prof = 0,
        .use_graph_optimize = 1,
        .graph_capacity = 1024,
    };
	
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create_config(&root_arena, ctx_config);
    ctx->compute_dtype = TK_F32;
	tk_rt_ctx_set_mode(ctx, RT_DRYRUN);

    /* ---- Model config (seq_len from file) ---- */
    struct tk_gpt2_config cfg = {
        .vocab_size      = 100, // use smaller value for test, the real value is 50257
        .max_seq_len     = 16, // 1024 
        .num_layers      = 8, // 12
        .seq_length      = 4,
        .hidden_dim      = 8, // 768
        .n_heads         = 2, // ???
        .inter_dim       = 0,
        .use_qkv_bias    = 1,
        .use_o_proj      = 1,
        .use_o_proj_bias = 1,
        .use_ffn_bias    = 1,
		
		.use_pre_norm	 = 1,
		.use_causal		 = 1,
    };
	
	struct tk_gpt2* model = NULL;
    if (tk_gpt2_config(ctx, cfg, &model) != 0) {
        fprintf(stderr, "gpt2_config failed\n"); return 1;
    }
    if (tk_gpt2_alloc(ctx, cfg, model) != 0) {
        fprintf(stderr, "gpt2_alloc failed\n"); return 1;
    }

	int hidden = cfg.hidden_dim;
	int head = cfg.n_heads;
	int N = cfg.seq_length;
	
	struct tk_tensor* input = NULL;
	int id_shape[1] = {N};
	tk_tensor_alloc(ctx->meta_arena, TK_I32, id_shape, 1, &input);

	int32_t* token_ids = (int32_t*)input->data;
	// set token id to {1, 2, 3, 4}
	for ( int i = 0 ; i < N ; ++i ) {
		token_ids[i] = i;
	}
	
	struct tk_tensor* output = NULL;
	int rc = tk_gpt2_forward(ctx, model, input, &output, TK_GPT2_PREFILL);
	printf("tk_gpt2_forward rc=%d\n", rc);

	tk_rt_prepare(ctx);

	ctx->ws->cur_offset = 0;
	struct tk_tensor* output2 = NULL;
	int rc2 = tk_gpt2_forward(ctx, model, input, &output2, TK_GPT2_PREFILL);
	printf("real inference rc=%d\n", rc2);

	float* out_data = (float*)output2->data;
	for (int i = 0; i < hidden; ++i) printf("%f ", out_data[i]);
	printf("\n");

// ------------------------- Without Optimization -------------------------

    struct tk_rt_ctx_config ctx_direct_config = {
        .use_int8 = 0,
        .use_prof = 0,
        .use_graph_optimize = 0,
        .graph_capacity = 1024,
    };
	
    struct tk_rt_ctx* ctx_2 = tk_runtime_ctx_create_config(&root_arena, ctx_direct_config);
    ctx_2->compute_dtype = TK_F32;
	tk_rt_ctx_set_mode(ctx_2, RT_INFERENCE);
	struct tk_tensor* output3 = NULL;
	int rc3 = tk_gpt2_forward(ctx_2, model, input, &output3, TK_GPT2_PREFILL);
	printf("tk_gpt2_forward rc3=%d\n", rc3);

	float* out_data2 = (float*)output3->data;
	for (int i = 0; i < hidden; ++i) printf("%f ", out_data2[i]);
	printf("\n");


	tk_rt_ctx_destroy(ctx);
	tk_rt_ctx_destroy(ctx_2);
	arena_destroy(&root_arena);
}

void test_generate_gpt2() {

	struct arena root_arena;
	arena_init(&root_arena);

    struct tk_rt_ctx_config ctx_config = {
        .use_int8 = 0,
        .use_prof = 0,
        .use_graph_optimize = 1,
        .graph_capacity = 1024,
    };
	
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create_config(&root_arena, ctx_config);
    ctx->compute_dtype = TK_F32;
	tk_rt_ctx_set_mode(ctx, RT_INFERENCE);

    /* ---- Model config (seq_len from file) ---- */
    struct tk_gpt2_config cfg = {
        .vocab_size      = 100, // use smaller value for test, the real value is 50257
        .max_seq_len     = 16, // 1024 
        .num_layers      = 8, // 12
        .seq_length      = 4,
        .hidden_dim      = 8, // 768
        .n_heads         = 2, // ???
        .inter_dim       = 0,
        .use_qkv_bias    = 1,
        .use_o_proj      = 1,
        .use_o_proj_bias = 1,
        .use_ffn_bias    = 1,
		
		.use_pre_norm	 = 1,
		.use_causal		 = 1,
    };
	
	struct tk_gpt2* model = NULL;
	if (tk_gpt2_config(ctx, cfg, &model) != 0) {
		fprintf(stderr, "gpt2_config failed\n"); return 1;
	}
    if (tk_gpt2_alloc(ctx, cfg, model) != 0) {
        fprintf(stderr, "gpt2_alloc failed\n"); return 1;
    }

	int hidden = cfg.hidden_dim;
	int head = cfg.n_heads;
	int N = cfg.seq_length;

	struct tk_tensor* prompt = NULL;
	int prompt_shape[1] = { 4 };
	tk_tensor_alloc(ctx->meta_arena, TK_I32, prompt_shape, 1, &prompt);
	int32_t* prompt_data = (int32_t*)prompt->data;
	prompt_data[0] = 1; prompt_data[1] = 2; prompt_data[2] = 3; prompt_data[3] = 4;


	/* PREFILL + DECODE */
	int32_t* out_tokens = NULL;
	int out_count = 0;
	int rc4 = tk_gpt2_generate(ctx, model, prompt, /* max_new_tokens= */ 10, /* eos_token_id= */ 99, &out_tokens, &out_count);
	printf("generate rc=%d, total tokens=%d: ", rc4, out_count);
	for (int i = 0; i < out_count; ++i) printf("%d ", out_tokens[i]);
		printf("\n");

	tk_rt_ctx_destroy(ctx);
	arena_destroy(&root_arena);
}

void test_generate_no_cache_gpt2() {

	struct arena root_arena;
	arena_init(&root_arena);

    struct tk_rt_ctx_config ctx_config = {
        .use_int8 = 0,
        .use_prof = 0,
        .use_graph_optimize = 1,
        .graph_capacity = 1024,
    };
	
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create_config(&root_arena, ctx_config);
    ctx->compute_dtype = TK_F32;
	tk_rt_ctx_set_mode(ctx, RT_INFERENCE);

    /* ---- Model config (seq_len from file) ---- */
    struct tk_gpt2_config cfg = {
        .vocab_size      = 100, // use smaller value for test, the real value is 50257
        .max_seq_len     = 16, // 1024 
        .num_layers      = 8, // 12
        .seq_length      = 4,
        .hidden_dim      = 8, // 768
        .n_heads         = 2, // ???
        .inter_dim       = 0,
        .use_qkv_bias    = 1,
        .use_o_proj      = 1,
        .use_o_proj_bias = 1,
        .use_ffn_bias    = 1,
		
		.use_pre_norm	 = 1,
		.use_causal		 = 1,
    };
	
	struct tk_gpt2* model = NULL;
	if (tk_gpt2_config(ctx, cfg, &model) != 0) {
		fprintf(stderr, "gpt2_config failed\n"); return 1;
	}
    if (tk_gpt2_alloc(ctx, cfg, model) != 0) {
        fprintf(stderr, "gpt2_alloc failed\n"); return 1;
    }

	int hidden = cfg.hidden_dim;
	int head = cfg.n_heads;
	int N = cfg.seq_length;

	struct tk_tensor* prompt = NULL;
	int prompt_shape[1] = { 4 };
	tk_tensor_alloc(ctx->meta_arena, TK_I32, prompt_shape, 1, &prompt);
	int32_t* prompt_data = (int32_t*)prompt->data;
	prompt_data[0] = 1; prompt_data[1] = 2; prompt_data[2] = 3; prompt_data[3] = 4;
		
	// since kv cache allocation need to be called explicitly
	// no cache means not calling the function

    ctx->ws->cur_offset = 0;
    ctx->kv_cur_len = 0;
    struct tk_tensor* logits = NULL;

    RT_CHECK(tk_gpt2_forward(ctx, model, prompt, &logits, TK_GPT2_PREFILL));

    /* take the last row of logits [prompt_len, vocab] -> [vocab], argmax */
	int max_new_tokens = 10;
	int prompt_len = prompt->shape[0];
    int32_t* tokens = arena_alloc(ctx->meta_arena, sizeof(int32_t) * (prompt_len + max_new_tokens));
    memcpy(tokens, prompt_data, sizeof(int32_t) * prompt_len);
    int total_count = prompt_len;
    int vocab_size = model->emb->config.vocab_size;
    float* last_row = (float*)logits->data + (size_t)(prompt_len - 1) * vocab_size;
    int next_token = argmax_f32(last_row, vocab_size);
    tokens[total_count++] = next_token;

    ctx->ws->cur_offset = 0;
	int eos_token_id = 99;

    for (int step = prompt_len; step < prompt_len + max_new_tokens - 1; ++step) {
        if (next_token == eos_token_id) break;

		int seq_len_this_step = total_count;
		int in_shape[1] = { seq_len_this_step };
		struct tk_tensor* step_input = NULL;
		RT_CHECK(tk_tensor_alloc(ctx->meta_arena, TK_I32, in_shape, 1, &step_input));
		memcpy(step_input->data, tokens, sizeof(int32_t) * seq_len_this_step);

		ctx->ws->cur_offset = 0;
		RT_CHECK(tk_gpt2_forward(ctx, model, step_input, &logits, TK_GPT2_PREFILL));

		float* last_row = (float*)logits->data + (size_t)(seq_len_this_step - 1) * vocab_size;
		next_token = argmax_f32(last_row, vocab_size);
		tokens[total_count++] = next_token;
    }

	printf("total tokens=%d: ", total_count);
	for (int i = 0; i < total_count; ++i) printf("%d ", tokens[i]);
		printf("\n");

	tk_rt_ctx_destroy(ctx);
	arena_destroy(&root_arena);
}
