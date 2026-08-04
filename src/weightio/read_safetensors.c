#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "cJSON.h"
#include "read_safetensors.h"



void print_first_n_element(FILE* f, struct Weight_Meta* item, size_t n, uint64_t data_start) {	
	printf("%-25s: ", item->name);
	float value[n];
	uint64_t file_offset = data_start + item->offset_arr[0];
	// printf("File offset: %" PRIu64 "\n", file_offset);
	fseek(f, (long)file_offset, SEEK_SET);
	fread(value, sizeof(float), n, f);
	for ( size_t i = 0 ; i < n-1 ; ++i ) {
		printf("%f, ", value[i]);
	}
	printf("%f\n", value[n-1]);
}

void wm_free(struct Weight_Meta* wm) {
	// we do not need to free dtype, it is a shallow copy from cJSON object
	free(wm->offset_arr);
	free(wm);
}

struct Weight_Meta* wm_create(cJSON* item) {

	cJSON* offsets = cJSON_GetObjectItemCaseSensitive(item, "data_offsets");
	char* dtype = cJSON_GetObjectItemCaseSensitive(item, "dtype")->valuestring;	
	if (!offsets) {
		fprintf(stderr, "Fail to find json field: \"data_offsets\"\n");
		return NULL;
	}
	if (!dtype) {
		fprintf(stderr, "Fail to find json field: \"dtype\"\n");
		return NULL;
	}
	int offset_ndims = cJSON_GetArraySize(offsets);
	int offset_arr[offset_ndims];

	int data_idx = 0;
	cJSON* offset = NULL;
	cJSON_ArrayForEach(offset, offsets) {
		offset_arr[data_idx++] = offset->valueint;
	}
	struct Weight_Meta* meta = malloc(sizeof(struct Weight_Meta));
	meta->item = item;
	meta->offset_arr = malloc(offset_ndims * sizeof(int));
	for ( int i = 0 ; i < data_idx ; ++i )
		meta->offset_arr[i] = offset_arr[i];
	meta->data_size = (offset_arr[1] - offset_arr[0]);
	meta->dtype = dtype;	
	meta->name = item->string;
	return meta;
}

struct Weight_Meta** wm_list_create(cJSON* item, size_t n_items) {
	struct Weight_Meta** list = malloc(sizeof(struct Weight_Meta*) * n_items);
	int n_items_idx = 0;
	cJSON* entry = NULL;
	cJSON_ArrayForEach(entry, item) {
    	const char* name = entry->string;
    	if (strcmp(name, "__metadata__") == 0) continue;
		list[n_items_idx++] = wm_create(entry);
	}
	return list;
}

static size_t calc_shape_size(int* shape, int ndims) {
	int size = 1;
	for ( int i = 0 ; i < ndims ; ++i ) size *= shape[i];
	return size;
}

// return bytes
static size_t dtype_size(char* dtype) {
	if (strcmp(dtype, "F32") == 0) {
		return 4;
	}
}

// item is a element like "h.3...."
void weight_size_check(cJSON* item) {
	char* name = item->string;
	char* dtype = cJSON_GetObjectItemCaseSensitive(item, "dtype")->valuestring;
	cJSON* shape = cJSON_GetObjectItemCaseSensitive(item, "shape");

	int ndims = cJSON_GetArraySize(shape);
	int shape_arr[ndims];
	int shape_idx = 0;

	cJSON* entry = NULL;
	cJSON_ArrayForEach(entry, shape) {
		shape_arr[shape_idx++] = entry->valueint;
	}

	size_t shape_size = calc_shape_size(shape_arr, ndims);

	cJSON* offsets = cJSON_GetObjectItemCaseSensitive(item, "data_offsets");
	int offset_ndims = cJSON_GetArraySize(offsets);
	int offset_arr[offset_ndims];

	int data_idx = 0;
	cJSON* offset = NULL;
	cJSON_ArrayForEach(offset, offsets) {
		offset_arr[data_idx++] = offset->valueint;
	}

	size_t byte_offset = offset_arr[1] - offset_arr[0];

	if (shape_size != byte_offset / dtype_size(dtype)) {
		printf("[FAIL] %s size mismatch\n", name);
	}
}

cJSON* str_to_json (char* buf, size_t sz) {
	cJSON* json = NULL;
	json = cJSON_Parse(buf);
	buf[sz] = '\0';
	return json;
}

// this function will automatically free the string
void json_tree_print(cJSON* json) {
	char* json_tree = cJSON_Print(json);
	printf("%s\n", json_tree);
	free(json_tree);
}

FILE* st_header_read(const char* st_path, uint64_t* header_sz, 
					 char** header_str, size_t* nread) {
	FILE* file = fopen(st_path, "rb");
	if (!file) {
	    fprintf(stderr, "fail to open file: \"%s\"\n", st_path);
		*header_sz = 0xA;
        return NULL;
	}
	uint64_t header_size = 0;
	uint64_t safetensor_offset = 8;
	fread(&header_size, sizeof(uint64_t), 1, file);
	printf("Header size: %" PRIu64 "\n", header_size);
	*header_sz = header_size;
	*header_str = malloc(header_size+1);
	size_t _nread = fread(*header_str, 1, (size_t)header_size, file);
	(*header_str)[_nread] = '\0';
	*nread = _nread;
	return file;
}

FILE* st_cfg_read(const char* st_cfg_path, uint64_t* cfg_sz, 
					 char** cfg_str, size_t* nread) {
	FILE* file = fopen(st_cfg_path, "rb");
	if (!file) {
	    fprintf(stderr, "fail to open file: \"%s\"\n", st_cfg_path);
		*cfg_sz = 0xA;
        return NULL;
	}
	fseek(file, 0L, SEEK_END);
	long sz = ftell(file);
	rewind(file);
	printf("config size: %" PRIu64 "\n", (uint64_t)sz);
	*cfg_sz = sz;
	*cfg_str = malloc(sz+1);
	size_t _nread = fread(*cfg_str, 1, (size_t)sz, file);
	(*cfg_str)[_nread] = '\0';
	*nread = _nread;
	return file;
}

int layer_count(cJSON* item, int* n_items) {
	int max_layer = -1;
	*n_items = 0;
	cJSON* entry = NULL;
	cJSON_ArrayForEach(entry, item) {
    	const char* name = entry->string;
    	if (strcmp(name, "__metadata__") == 0) continue;
		int layer_idx;
		(*n_items)++;
		if (sscanf(name, "h.%d.", &layer_idx) == 1) {
			if (layer_idx > max_layer) max_layer = layer_idx;
		}
		weight_size_check(entry);
	}
	return max_layer+1;
}

int safetensors_read(const char* path, const char* cfg_path) {

	cJSON* json = NULL;
	cJSON* cfg_json = NULL;
	FILE* f = NULL;
	FILE* cfg_f = NULL;
	char* header_str = NULL;
	char* cfg_str = NULL;
	struct Weight_Meta** wm_list = NULL;

	uint64_t header_size;
	size_t sz;
	f = st_header_read(path, &header_size, &header_str, &sz);
	if (!f)		return -1;
	json = str_to_json(header_str, sz);
	if (!json)	goto clean_up;

	uint64_t cfg_size;
	size_t cfg_sz;
	cfg_f = st_cfg_read(cfg_path, &cfg_size, &cfg_str, &cfg_sz);
	cfg_json = str_to_json(cfg_str, cfg_sz);

	uint64_t data_start = 8 + header_size;

	// --- cfg and json check ---
	int n_items = 0;
	int num_layers = layer_count(json, &n_items);
	cJSON* item = cJSON_GetObjectItemCaseSensitive(cfg_json, "n_layer");
	if (!item) {
		printf("[FAIL] Missing cJSON item \"n_layer\"\n");
		goto clean_up;
	}
	if (num_layers == item->valuedouble)
		printf("[OK] Number of layer matched\n");

	wm_list = wm_list_create(json, n_items);
	if (!wm_list) goto clean_up;

	size_t n_items_print = 8;
	for ( int wm_list_idx = 0 ; wm_list_idx < n_items ; ++wm_list_idx )
		print_first_n_element(f, wm_list[wm_list_idx], n_items_print, data_start);


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
}
