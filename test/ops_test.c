#include "arena.h"
#include "tensor.h"
#include "tensor_ops.h"
#include "tf_block.h"
#include "rt_context.h"
#include "tk_ops.h"
#include "math.h"

void test_causal_masking_basic();
void test_causal_no_future_leakage(); 
void test_kv_cache_matches_full_recompute();
void test_kv_cache_multistep();
void test_kv_cache_multilayer();

int main() {
	test_causal_masking_basic();
	test_causal_no_future_leakage(); 
	test_kv_cache_matches_full_recompute();
	test_kv_cache_multistep();
	test_kv_cache_multilayer();
}

void test_causal_masking_basic() {

	struct arena a;
	arena_init(&a);

    int shape[2] = { 4, 4 };
    struct tk_tensor* score = NULL;
    tk_tensor_alloc(&a, TK_F32, shape, 2, &score);

	float* data = (float*)score->data;
    for (int i = 0; i < 16; ++i) data[i] = 1.0f;

    printf("Before mask:\n");
    tk_tensor_print(score);

	// ---- test casual masking ----
    tk_tensor_causal_mask(score);

    printf("After mask (upper triangle should be -inf):\n");
    tk_tensor_print(score);

    int pass = 1;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float val = data[row * 4 + col];
            if (col > row) {
                if (!isinf(val) || val > 0) {
                    printf("FAIL: row=%d col=%d should be -inf, got %f\n", row, col, val);
                    pass = 0;
                }
            } else {
                if (val != 1.0f) {
                    printf("FAIL: row=%d col=%d should be 1.0, got %f\n", row, col, val);
                    pass = 0;
                }
            }
        }
    }

    tk_ops_softmax(score, score);
    printf("After softmax:\n");
    tk_tensor_print(score);

    for (int row = 0; row < 4; ++row) {
        float row_sum = 0.0f;
        for (int col = 0; col < 4; ++col) {
            float val = data[row * 4 + col];
            row_sum += val;
            if (col > row && val > 1e-6f) {
                printf("FAIL: row=%d col=%d should be ~0 after softmax, got %f\n", row, col, val);
                pass = 0;
            }
        }
        if (fabsf(row_sum - 1.0f) > 1e-4f) {
            printf("FAIL: row=%d softmax sum should be 1.0, got %f\n", row, row_sum);
            pass = 0;
        }
    }

	printf("[TEST] %s: ", __func__);
    printf(pass ? "\e[92mPASS\e[0m\n" : "\e[91mFAIL\e[0m\n");
	arena_destroy(&a);
}

void test_causal_no_future_leakage() {

	struct arena a;
	arena_init(&a);

	struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&a);
	tk_rt_ctx_set_mode(ctx, RT_INFERENCE);	

    struct TransformerBlock* tf = tk_tf_block_create(ctx);
    tf->config.seq_length = 4;
    tf->config.hidden_dim = 8;
    tf->config.n_heads    = 2;
    tf->config.dtype      = TK_F32;
    tf->config.use_causal = 1;   // enable causal masking

    tk_tf_block_alloc(ctx, tf);

    int in_shape[2] = { 4, 8 };
    struct tk_tensor* input_a = NULL;
    struct tk_tensor* input_b = NULL;
    tk_tensor_alloc(ctx->data_arena, TK_F32, in_shape, 2, &input_a);
    tk_tensor_alloc(ctx->data_arena, TK_F32, in_shape, 2, &input_b);

	// input_a and input_b's first three tokens are identical, only the last one differs
    float* a_data = (float*)input_a->data;
    float* b_data = (float*)input_b->data;
    for (int i = 0; i < 3 * 8; ++i) {
        a_data[i] = b_data[i] = (float)(i % 5) * 0.1f;
    }
    for (int j = 0; j < 8; ++j) {
        a_data[3 * 8 + j] = 1.0f;
        b_data[3 * 8 + j] = 9999.0f;
    }

    struct tk_tensor* out_a = NULL;
    struct tk_tensor* out_b = NULL;
    int rc_a = tk_tf_attention_forward(ctx, tf, input_a, &out_a);
    int rc_b = tk_tf_attention_forward(ctx, tf, input_b, &out_b);

	printf("rc_a: %d\n", rc_a);
	printf("rc_b: %d\n", rc_b);

    float* oa = (float*)out_a->data;
    float* ob = (float*)out_b->data;

    int pass = 1;
    for (int i = 0; i < 3 * 8; ++i) {
        if (fabsf(oa[i] - ob[i]) > 1e-5f) {
            printf("FAIL: token leaked future info at index %d (a=%f, b=%f)\n", i, oa[i], ob[i]);
            pass = 0;
        }
    }

    int last_row_differs = 0;
    for (int j = 0; j < 8; ++j) {
        if (fabsf(oa[3*8+j] - ob[3*8+j]) > 1e-5f) {
            last_row_differs = 1;
            break;
        }
    }

	int has_nan = 0;
	for (int j = 0; j < 8; ++j) {
		if (isnan(oa[3*8+j]) || isnan(ob[3*8+j])) {
			has_nan = 1;
			break;
		}
	}

	printf("out_a last row: ");
		for (int j = 0; j < 8; ++j) printf("%f ", oa[3*8+j]);
	printf("\nout_b last row: ");
		for (int j = 0; j < 8; ++j) printf("%f ", ob[3*8+j]);
	printf("\n");

	if (has_nan) {
		printf("WARNING: NaN detected in last row output — likely overflow, not a real equality\n");
	} else if (!last_row_differs) {
		printf("WARNING: last row output identical — test may not be sensitive enough\n");
	}

	printf("[TEST] %s: ", __func__);
    printf(pass ? "\e[92mPASS\e[0m\n" : "\e[91mFAIL\e[0m\n");
	
	arena_destroy(&a);
}

void test_kv_cache_matches_full_recompute() {

    struct arena a;
    arena_init(&a);

    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&a);
    tk_rt_ctx_set_mode(ctx, RT_INFERENCE);

    const int N      = 4;   /* total sequence length */
    const int hidden = 8;
    const int heads  = 2;

    /* ---- build one TransformerBlock, weights shared by both paths ---- */
    struct TransformerBlock* tf = tk_tf_block_create(ctx);
    memset(tf, 0, sizeof(struct TransformerBlock));   /* arena_alloc does NOT zero memory */

    tf->config.hidden_dim  = hidden;
    tf->config.n_heads     = heads;
    tf->config.dtype       = TK_F32;
    tf->config.use_causal  = 1;
    tf->config.max_seq_len = 16;
    /* use_qkv_bias / use_o_proj / use_ffn_bias left 0 (memset) to keep this minimal */

    srand(1234);
    tk_tf_block_alloc(ctx, tf);

    tk_tf_kv_cache_alloc(ctx, tf);
    ctx->kv_cur_len = 0;

	tk_rt_ctx_set_mode(ctx, RT_INFERENCE);

    /* ---- deterministic input for all N tokens ---- */
    int full_shape[2] = { N, hidden };
    struct tk_tensor* full_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, full_shape, 2, &full_input);
    float* fdata = (float*)full_input->data;
    for (int i = 0; i < N * hidden; ++i)
        fdata[i] = (float)(i % 7) * 0.1f - 0.3f;

    /* =========================================================
     * Path A: KV cache  (prefill first N-1 tokens, decode last)
     * ========================================================= */

    int prefill_shape[2] = { N - 1, hidden };
    struct tk_tensor* prefill_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, prefill_shape, 2, &prefill_input);
    memcpy(prefill_input->data, fdata, (size_t)(N - 1) * hidden * sizeof(float));

    tf->config.seq_length = N - 1;
    ctx->ws->cur_offset = 0;
    struct tk_tensor* prefill_out = NULL;
    int rc1 = tk_tf_attention_forward(ctx, tf, prefill_input, &prefill_out);

    int decode_shape[2] = { 1, hidden };
    struct tk_tensor* decode_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, decode_shape, 2, &decode_input);
    memcpy(decode_input->data, fdata + (size_t)(N - 1) * hidden, hidden * sizeof(float));

    ctx->ws->cur_offset = 0;
    struct tk_tensor* decode_out = NULL;
    int rc2 = tk_tf_attention_forward_decode(ctx, tf, decode_input, &decode_out);

    /* copy result out NOW — the next forward call will reuse/overwrite workspace memory */
    float kv_result[64];
    memcpy(kv_result, decode_out->data, (size_t)hidden * sizeof(float));

    /* =========================================================
     * Path B: full recompute (all N tokens in one shot, no cache)
     * ========================================================= */

    struct tk_tensor* saved_k_cache = tf->k_cache;
    struct tk_tensor* saved_v_cache = tf->v_cache;
    tf->k_cache = NULL;   /* detach cache so this pass doesn't touch/overwrite it */
    tf->v_cache = NULL;

    tf->config.seq_length = N;
    ctx->ws->cur_offset = 0;
    struct tk_tensor* full_out = NULL;
    int rc3 = tk_tf_attention_forward(ctx, tf, full_input, &full_out);

    tf->k_cache = saved_k_cache;
    tf->v_cache = saved_v_cache;

    printf("rc1=%d rc2=%d rc3=%d\n", rc1, rc2, rc3);

    float* full_data     = (float*)full_out->data;
    float* full_last_row = full_data + (size_t)(N - 1) * hidden;

    /* =========================================================
     * Compare: KV-cache decode output vs. last row of full recompute
     * ========================================================= */

    int pass = 1;
    printf("kv_result:      ");
    for (int j = 0; j < hidden; ++j) printf("%f ", kv_result[j]);
    printf("\nfull_last_row:  ");
    for (int j = 0; j < hidden; ++j) printf("%f ", full_last_row[j]);
    printf("\n");

    for (int j = 0; j < hidden; ++j) {
        if (fabsf(kv_result[j] - full_last_row[j]) > 1e-4f) {
            printf("FAIL: index %d differs (kv=%f, full=%f)\n",
                   j, kv_result[j], full_last_row[j]);
            pass = 0;
        }
    }

	printf("[TEST] %s: ", __func__);
    printf(pass ? "\e[92mPASS\e[0m\n" : "\e[91mFAIL\e[0m\n");

    arena_destroy(&a);
}

void test_kv_cache_multistep() {
 
    struct arena a;
    arena_init(&a);
 
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&a);
    tk_rt_ctx_set_mode(ctx, RT_INFERENCE);
 
    const int N       = 6;   /* total sequence length */
    const int P       = 2;   /* prefill length */
    const int hidden  = 8;
    const int heads   = 2;
 
    struct TransformerBlock* tf = tk_tf_block_create(ctx);
    memset(tf, 0, sizeof(struct TransformerBlock));
 
    tf->config.hidden_dim  = hidden;
    tf->config.n_heads     = heads;
    tf->config.dtype       = TK_F32;
    tf->config.use_causal  = 1;
    tf->config.max_seq_len = 16;
 
    srand(1234);
    tk_tf_block_alloc(ctx, tf);
 
    tk_tf_kv_cache_alloc(ctx, tf);
    ctx->kv_cur_len = 0;
 
    /* ---- deterministic input for all N tokens ---- */
    int full_shape[2] = { N, hidden };
    struct tk_tensor* full_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, full_shape, 2, &full_input);
    float* fdata = (float*)full_input->data;
    for (int i = 0; i < N * hidden; ++i)
        fdata[i] = (float)(i % 7) * 0.1f - 0.3f;
 
    /* =========================================================
     * Path A: KV cache — prefill P tokens, then decode N-P times
     * ========================================================= */
 
    int prefill_shape[2] = { P, hidden };
    struct tk_tensor* prefill_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, prefill_shape, 2, &prefill_input);
    memcpy(prefill_input->data, fdata, (size_t)P * hidden * sizeof(float));
 
    tf->config.seq_length = P;
    ctx->ws->cur_offset = 0;
    struct tk_tensor* prefill_out = NULL;
    int rc_prefill = tk_tf_attention_forward(ctx, tf, prefill_input, &prefill_out);
    printf("rc_prefill=%d  kv_cur_len after prefill=%d\n", rc_prefill, ctx->kv_cur_len);
 
    /* stash each decode step's output here for later comparison */
    float kv_results[N][64];
 
    for (int step = 0; step < N - P; ++step) {
        int token_idx = P + step;
 
        int decode_shape[2] = { 1, hidden };
        struct tk_tensor* decode_input = NULL;
        tk_tensor_alloc(ctx->meta_arena, TK_F32, decode_shape, 2, &decode_input);
        memcpy(decode_input->data, fdata + (size_t)token_idx * hidden, hidden * sizeof(float));
 
        ctx->ws->cur_offset = 0;
        struct tk_tensor* decode_out = NULL;
        int rc = tk_tf_attention_forward_decode(ctx, tf, decode_input, &decode_out);
 
        printf("step=%d token_idx=%d rc=%d kv_cur_len(before increment)=%d\n",
               step, token_idx, rc, ctx->kv_cur_len);
 
        /* copy result out before the next call reuses workspace memory */
        memcpy(kv_results[token_idx], decode_out->data, (size_t)hidden * sizeof(float));
 
        ctx->kv_cur_len++;   /* advance cache position for the next step */
    }
 
    /* =========================================================
     * Path B: full recompute — all N tokens in one shot, no cache
     * ========================================================= */
 
    struct tk_tensor* saved_k_cache = tf->k_cache;
    struct tk_tensor* saved_v_cache = tf->v_cache;
    tf->k_cache = NULL;
    tf->v_cache = NULL;
 
    tf->config.seq_length = N;
    ctx->ws->cur_offset = 0;
    struct tk_tensor* full_out = NULL;
    int rc_full = tk_tf_attention_forward(ctx, tf, full_input, &full_out);
 
    tf->k_cache = saved_k_cache;
    tf->v_cache = saved_v_cache;
 
    printf("rc_full=%d\n", rc_full);
    float* full_data = (float*)full_out->data;
 
    /* =========================================================
     * Compare every decoded token's output against the matching
     * row of the full recompute — not just the last one
     * ========================================================= */
 
    int pass = 1;
    for (int token_idx = P; token_idx < N; ++token_idx) {
        float* full_row = full_data + (size_t)token_idx * hidden;
        float* kv_row   = kv_results[token_idx];
 
        int row_pass = 1;
        for (int j = 0; j < hidden; ++j) {
            if (fabsf(kv_row[j] - full_row[j]) > 1e-4f) {
                printf("FAIL: token_idx=%d index=%d differs (kv=%f, full=%f)\n",
                       token_idx, j, kv_row[j], full_row[j]);
                row_pass = 0;
                pass = 0;
            }
        }
        printf("token_idx=%d: %s\n", token_idx, row_pass ? "match" : "MISMATCH");
    }
 
	printf("[TEST] %s: ", __func__);
    printf(pass ? "\e[92mPASS\e[0m\n" : "\e[91mFAIL\e[0m\n");
 
    arena_destroy(&a);
}

void test_kv_cache_multilayer() {

    struct arena a;
    arena_init(&a);

    struct tk_rt_ctx* ctx = tk_runtime_ctx_create(&a);
    tk_rt_ctx_set_mode(ctx, RT_INFERENCE);

    const int NUM_LAYERS = 2;
    const int N          = 5;   /* total sequence length */
    const int P          = 2;   /* prefill length */
    const int hidden     = 8;
    const int heads      = 2;

    /* ---- build NUM_LAYERS independent TransformerBlocks, each with its own cache ---- */
    struct TransformerBlock* layers[NUM_LAYERS];
    srand(1234);   /* single seed, weights differ naturally as rand() sequence advances per layer */
    for (int l = 0; l < NUM_LAYERS; ++l) {
        layers[l] = tk_tf_block_create(ctx);
        memset(layers[l], 0, sizeof(struct TransformerBlock));
        layers[l]->config.hidden_dim  = hidden;
        layers[l]->config.n_heads     = heads;
        layers[l]->config.dtype       = TK_F32;
        layers[l]->config.use_causal  = 1;
        layers[l]->config.max_seq_len = 16;

        tk_tf_block_alloc(ctx, layers[l]);
        tk_tf_kv_cache_alloc(ctx, layers[l]);
    }
    ctx->kv_cur_len = 0;

    /* ---- deterministic input for all N tokens (fed into layer 0) ---- */
    int full_shape[2] = { N, hidden };
    struct tk_tensor* full_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, full_shape, 2, &full_input);
    float* fdata = (float*)full_input->data;
    for (int i = 0; i < N * hidden; ++i)
        fdata[i] = (float)(i % 7) * 0.1f - 0.3f;

    /* =========================================================
     * Path A: KV cache — prefill P tokens through all layers,
     * then decode N-P times, chaining layer0 -> layer1 -> ...
     * ========================================================= */

    int prefill_shape[2] = { P, hidden };
    struct tk_tensor* prefill_input = NULL;
    tk_tensor_alloc(ctx->meta_arena, TK_F32, prefill_shape, 2, &prefill_input);
    memcpy(prefill_input->data, fdata, (size_t)P * hidden * sizeof(float));

    for (int l = 0; l < NUM_LAYERS; ++l)
        layers[l]->config.seq_length = P;

    ctx->ws->cur_offset = 0;   /* reset ONCE for the whole prefill pass across all layers */
    struct tk_tensor* layer_input = prefill_input;
    struct tk_tensor* layer_out   = NULL;
    for (int l = 0; l < NUM_LAYERS; ++l) {
        int rc = tk_tf_attention_forward(ctx, layers[l], layer_input, &layer_out);
        printf("prefill layer=%d rc=%d\n", l, rc);
        layer_input = layer_out;   /* chain: this layer's output feeds the next */
    }
    printf("kv_cur_len after prefill=%d\n", ctx->kv_cur_len);

    float kv_results[N][64];

    for (int step = 0; step < N - P; ++step) {
        int token_idx = P + step;

        int decode_shape[2] = { 1, hidden };
        struct tk_tensor* decode_input = NULL;
        tk_tensor_alloc(ctx->meta_arena, TK_F32, decode_shape, 2, &decode_input);
        memcpy(decode_input->data, fdata + (size_t)token_idx * hidden, hidden * sizeof(float));

        ctx->ws->cur_offset = 0;   /* reset ONCE per token, before layer 0 — not between layers */
        struct tk_tensor* dl_input = decode_input;
        struct tk_tensor* dl_out   = NULL;
        int rc = 0;
        for (int l = 0; l < NUM_LAYERS; ++l) {
            rc = tk_tf_attention_forward_decode(ctx, layers[l], dl_input, &dl_out);
            dl_input = dl_out;
        }
        printf("decode step=%d token_idx=%d rc=%d\n", step, token_idx, rc);

        /* copy the FINAL layer's output before the next call reuses workspace memory */
        memcpy(kv_results[token_idx], dl_out->data, (size_t)hidden * sizeof(float));

        ctx->kv_cur_len++;   /* advance once per token, after all layers processed it */
    }

    /* =========================================================
     * Path B: full recompute — all N tokens, all layers, no cache
     * ========================================================= */

    struct tk_tensor* saved_k[NUM_LAYERS];
    struct tk_tensor* saved_v[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; ++l) {
        saved_k[l] = layers[l]->k_cache;
        saved_v[l] = layers[l]->v_cache;
        layers[l]->k_cache = NULL;
        layers[l]->v_cache = NULL;
        layers[l]->config.seq_length = N;
    }

    ctx->ws->cur_offset = 0;
    struct tk_tensor* full_layer_input = full_input;
    struct tk_tensor* full_layer_out   = NULL;
    int rc_full = 0;
    for (int l = 0; l < NUM_LAYERS; ++l) {
        rc_full = tk_tf_attention_forward(ctx, layers[l], full_layer_input, &full_layer_out);
        full_layer_input = full_layer_out;
    }
    printf("rc_full=%d\n", rc_full);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        layers[l]->k_cache = saved_k[l];
        layers[l]->v_cache = saved_v[l];
    }

    float* full_data = (float*)full_layer_out->data;

    /* =========================================================
     * Compare every decoded token's final-layer output against
     * the matching row of the full recompute
     * ========================================================= */

    int pass = 1;
    for (int token_idx = P; token_idx < N; ++token_idx) {
        float* full_row = full_data + (size_t)token_idx * hidden;
        float* kv_row   = kv_results[token_idx];

        int row_pass = 1;
        for (int j = 0; j < hidden; ++j) {
            if (fabsf(kv_row[j] - full_row[j]) > 1e-3f) {
                printf("FAIL: token_idx=%d index=%d differs (kv=%f, full=%f)\n",
                       token_idx, j, kv_row[j], full_row[j]);
                row_pass = 0;
                pass = 0;
            }
        }
        printf("token_idx=%d: %s\n", token_idx, row_pass ? "match" : "MISMATCH");
    }

	printf("[TEST] %s: ", __func__);
    printf(pass ? "\e[92mPASS\e[0m\n" : "\e[91mFAIL\e[0m\n");

    arena_destroy(&a);
}
