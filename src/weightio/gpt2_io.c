#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpt2_io.h"
#include "read_safetensors.h"
#include "rt_error.h"
#include "tensor.h"
#include "gpt2.h"
#include "tf_block.h"

/* ------------------------------------------------------------------ */
/* Low-level helpers                                                    */
/* ------------------------------------------------------------------ */

static int read_bytes(FILE* fp, void* ptr, size_t n) {
    return (fread(ptr, 1, n, fp) == n) ? 0 : -1;
}

static int read_u32(FILE* fp, uint32_t* v) {
    return read_bytes(fp, v, sizeof(*v));
}

static int read_u64(FILE* fp, uint64_t* v) {
    return read_bytes(fp, v, sizeof(*v));
}

static int read_f64(FILE* fp, double* v) {
    return read_bytes(fp, v, sizeof(*v));
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

static int move_fp(FILE* fp, uint64_t abs_offset) {
	return fseek(fp, abs_offset, SEEK_SET) == 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

// caller must allocate space for out_cfg
int tk_gpt2_config_from_json(const char* cfg_path, struct tk_gpt2_config* out_cfg) {
    FILE* f = NULL;
    char* cfg_str = NULL;
    uint64_t cfg_size;
    size_t cfg_sz;

    f = st_cfg_read(cfg_path, &cfg_size, &cfg_str, &cfg_sz);
    if (!f) return -1;

    cJSON* cfg_json = str_to_json(cfg_str, cfg_sz);
    if (!cfg_json) { free(cfg_str); fclose(f); return -1; }

    out_cfg->num_layers  = (int)cJSON_GetObjectItemCaseSensitive(cfg_json, "n_layer")->valuedouble;
    out_cfg->n_heads     = (int)cJSON_GetObjectItemCaseSensitive(cfg_json, "n_head")->valuedouble;
    out_cfg->hidden_dim  = (int)cJSON_GetObjectItemCaseSensitive(cfg_json, "n_embd")->valuedouble;
    out_cfg->vocab_size  = (int)cJSON_GetObjectItemCaseSensitive(cfg_json, "vocab_size")->valuedouble;
    out_cfg->max_seq_len = (int)cJSON_GetObjectItemCaseSensitive(cfg_json, "n_positions")->valuedouble;
	out_cfg->inter_dim = 0;	// 0 means 4 * hidden_dim

    out_cfg->use_qkv_bias    = 1;
    out_cfg->use_o_proj      = 1;
    out_cfg->use_o_proj_bias = 1;
    out_cfg->use_ffn_bias    = 1;
    out_cfg->use_pre_norm    = 1;
    out_cfg->use_causal      = 1;

    cJSON_Delete(cfg_json);
    free(cfg_str);
    fclose(f);
    return 0;
}

struct tk_tensor* find_dest_tensor (struct tk_gpt2* model,
   				      struct Weight_Meta* wm) {

	char* name = wm->name;
	struct tk_tensor* out_tensor = NULL;

	if (!strcmp(name, "wte.weight")) {
		 out_tensor = model->emb->word_emb->weights;
	}
	else if (!strcmp(name, "wpe.weight")) {
		out_tensor = model->emb->pos_emb->weights;
	}
	else {
		int layer_idx = 0;
		sscanf(wm->name, "h.%d.", &layer_idx);
		struct TransformerBlock* tf = model->blocks[layer_idx]->base;
		if (strstr(wm->name, "attn.c_proj.weight") != NULL) {
			out_tensor = tf->o_proj_weights;	
		}
		else if (strstr(wm->name, "attn.c_proj.bias") != NULL) {
			out_tensor = tf->o_proj_bias;
		}
		else if (strstr(wm->name, "mlp.c_fc.weight") != NULL) {
			out_tensor = tf->ffn_up_weights;
		}
		else if (strstr(wm->name, "mlp.c_proj.weight") != NULL) {
			out_tensor = tf->ffn_down_weights;
		}
		else if (strstr(wm->name, "mlp.c_fc.bias") != NULL) {
			out_tensor = tf->ffn_up_bias;
		}
		else if (strstr(wm->name, "mlp.c_proj.bias") != NULL) {
			out_tensor = tf->ffn_down_bias;
		}
		else if (strstr(wm->name, "ln_1.weight") != NULL) {
			out_tensor = tf->ln1_gamma;
		}
		else if (strstr(wm->name, "ln_1.bias") != NULL) {
			out_tensor = tf->ln1_beta;
		}
		else if (strstr(wm->name, "ln_2.weight") != NULL) {
			out_tensor = tf->ln2_gamma;
		}
		else if (strstr(wm->name, "ln_2.bias") != NULL) {
			out_tensor = tf->ln2_beta;
		}
		else if (strstr(wm->name, "ln_f.bias") != NULL) {
			out_tensor = model->final_ln_beta;
		}
		else if (strstr(wm->name, "ln_f.weight") != NULL) {
			out_tensor = model->final_ln_gamma;
		}
	}
	return out_tensor;
}

int load_qkv_split(struct TransformerBlock* tf, FILE* f, struct Weight_Meta* wm) {
    /* c_attn.weight shape: [hidden_dim, 3 * hidden_dim]，row-major
     * each row contains 3 * hidden dim floats, the first 1/3 of hidden_dim is Q then K then V*/
    int hidden_dim = tf->config.hidden_dim;
    size_t row_floats     = (size_t)hidden_dim * 3;
    size_t part_floats    = (size_t)hidden_dim;
    size_t row_bytes      = row_floats * sizeof(float);
    size_t part_bytes     = part_floats * sizeof(float);

    float* q_dst = (float*)tf->q_weights->data;
    float* k_dst = (float*)tf->k_weights->data;
    float* v_dst = (float*)tf->v_weights->data;

    for (int r = 0; r < hidden_dim; ++r) {
        if (fread(q_dst + (size_t)r * part_floats, 1, part_bytes, f) != part_bytes) return -1;
        if (fread(k_dst + (size_t)r * part_floats, 1, part_bytes, f) != part_bytes) return -1;
        if (fread(v_dst + (size_t)r * part_floats, 1, part_bytes, f) != part_bytes) return -1;
    }
    return 0;
}

int load_qkv_bias_split(struct TransformerBlock* tf, FILE* f, struct Weight_Meta* wm) {
    int hidden_dim = tf->config.hidden_dim;
    size_t part_bytes = (size_t)hidden_dim * sizeof(float);

    if (fread(tf->q_bias->data, 1, part_bytes, f) != part_bytes) return -1;
    if (fread(tf->k_bias->data, 1, part_bytes, f) != part_bytes) return -1;
    if (fread(tf->v_bias->data, 1, part_bytes, f) != part_bytes) return -1;
    return 0;
}

int tk_gpt2_safetensors_load(const char* path,
						 const char* cfg_path,
						 struct tk_gpt2_config* out_config,
                         struct tk_gpt2* model) {

    if (!path || !out_config || !model)
        RT_FAIL(RT_EINVAL, "tk_gpt2_load_weights: NULL argument");

	cJSON* json = NULL;
	cJSON* cfg_json = NULL;
	FILE* f = NULL;
	FILE* cfg_f = NULL;
	char* header_str = NULL;
	char* cfg_str = NULL;
	struct Weight_Meta** wm_list = NULL;

	int ok = 0;

	uint64_t header_size;
	size_t sz;
	f = st_header_read(path, &header_size, &header_str, &sz);
	if (!f)		return -1;
	json = str_to_json(header_str, sz);
	if (!json)	{
		ok = -1;
		goto clean_up;
	}

	uint64_t cfg_size;
	size_t cfg_sz;
	cfg_f = st_cfg_read(cfg_path, &cfg_size, &cfg_str, &cfg_sz);
	cfg_json = str_to_json(cfg_str, cfg_sz);

	// safetensors format, first 8 bytes represent the size of json
	uint64_t data_start = 8 + header_size;

	// --- cfg and json check ---
	int n_items = 0;
	int num_layers = layer_count(json, &n_items);
	cJSON* item = cJSON_GetObjectItemCaseSensitive(cfg_json, "n_layer");
	if (!item) {
		printf("[FAIL] Missing cJSON item \"n_layer\"\n");
		ok = -1;
		goto clean_up;
	}
	if (num_layers == item->valuedouble)
		printf("[OK] Number of layer matched\n");

	wm_list = wm_list_create(json, n_items);
	if (!wm_list) {
		ok = -1;
		goto clean_up;
	}


	// --- read weights ---

    struct tk_gpt2_emb* emb = model->emb;

	for ( int i = 0 ; i < n_items ; ++i ) {
		struct Weight_Meta* wm = wm_list[i];
		if (strstr(wm->name, "c_attn.weight") != NULL) {
			int layer_idx = 0;
			sscanf(wm->name, "h.%d.", &layer_idx);
			move_fp(f, data_start + wm->offset_arr[0]);
			load_qkv_split(model->blocks[layer_idx]->base, f, wm);
			continue;
		}
		if (strstr(wm->name, "c_attn.bias") != NULL) {
			int layer_idx = 0;
			sscanf(wm->name, "h.%d.", &layer_idx);
			move_fp(f, data_start + wm->offset_arr[0]);
			load_qkv_bias_split(model->blocks[layer_idx]->base, f, wm);
			continue;
		}
		if (strstr(wm->name, "attn.bias") != NULL) continue;
		struct tk_tensor* dest = find_dest_tensor(model, wm);
		if (!dest) {
			fprintf(stderr, "[WARN] unhandled tensor: %s\n", wm->name);
			continue;
		}
		size_t count = shape_size_calc(dest->shape, dest->ndims);
		move_fp(f, data_start + wm->offset_arr[0]);
		if (load_or_skip(f, dest, count) < 0) {
			ok = -1;
			goto clean_up;
		}
	}
    printf("[gpt2_io] %d tensors loaded\n", n_items);
    printf("[gpt2_io] layer %d loaded\n", num_layers);
    printf("[gpt2_io] all weights loaded from '%s'\n", path);

	/*
	printf("q_weights[0..7]: ");
	float* qw = (float*)model->blocks[0]->base->q_weights->data;
	for (int i = 0; i < 8; ++i) printf("%f ", qw[i]);
	printf("\nk_weights[0..7]: ");
	float* kw = (float*)model->blocks[0]->base->k_weights->data;
	for (int i = 0; i < 8; ++i) printf("%f ", kw[i]);
	printf("\nv_weights[0..7]: ");
	float* vw = (float*)model->blocks[0]->base->v_weights->data;
	for (int i = 0; i < 8; ++i) printf("%f ", vw[i]);
	printf("\nq_bias[0..7]: ");
	float* qb = (float*)model->blocks[0]->base->q_bias->data;
	for (int i = 0; i < 8; ++i) printf("%f ", qb[i]);
	printf("\nk_bias[0..7]: ");
	float* kb = (float*)model->blocks[0]->base->k_bias->data;
	for (int i = 0; i < 8; ++i) printf("%f ", kb[i]);
	printf("\nv_bias[0..7]: ");
	float* vb = (float*)model->blocks[0]->base->v_bias->data;
	for (int i = 0; i < 8; ++i) printf("%f ", vb[i]);
	printf("\n");
	*/

clean_up:
	if (json) cJSON_Delete(json);
	if (cfg_json) cJSON_Delete(cfg_json);
	if (wm_list) {
	for (size_t item_idx = 0; item_idx < n_items; ++item_idx)
		wm_free(wm_list[item_idx]);
		free(wm_list);
	}
	if (header_str)	free(header_str);
	if (cfg_str)	free(cfg_str);
	if (f) fclose(f);
	if (cfg_f) fclose(cfg_f); 
	
	if (ok == 0) {
    	return 0;
	}
	else {
		printf("[gpt2_io] failed to load weight from '%s'\n", path);
	}
	return -1;
}
