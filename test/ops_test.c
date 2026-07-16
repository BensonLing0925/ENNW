#include "../mem/arena.h"
#include "../src/ops/tensor.h"
#include "../src/ops/tensor_ops.h"
#include "../src/modules/transformer/tf_block.h"
#include "../src/runtime/rt_context.h"
#include "../src/ops/tk_ops.h"
#include "math.h"

int main() {
	test_causal_masking_basic();
	test_causal_no_future_leakage(); 
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
