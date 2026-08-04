#ifndef TK_READST_H
#define TK_READST_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "cJSON.h"
#define MAX_STR_LEN (1 << 14)	// 16384 > 14283

struct Weight_Meta {
	cJSON* item;
	int* offset_arr;
	size_t data_size;
	char* dtype;
	char* name;
};

void print_first_n_element(FILE* f, struct Weight_Meta* item, size_t n, uint64_t data_start);
struct Weight_Meta* wm_create(cJSON* item);
void wm_free(struct Weight_Meta* wm);
struct Weight_Meta** wm_list_create(cJSON* item, size_t n_items);
void weight_size_check(cJSON* item);
FILE* st_header_read(const char* st_path, uint64_t* header_sz, char** header_str, size_t* nread);
void json_tree_print(cJSON* json);
cJSON* str_to_json (char* buf, size_t sz);
FILE* st_cfg_read(const char* st_cfg_path, uint64_t* cfg_sz, char** cfg_str, size_t* nread);
int layer_count(cJSON* item, int* n_items);
int safetensors_read(const char* path, const char* cfg_path);

#endif
