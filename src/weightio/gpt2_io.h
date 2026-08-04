#ifndef TK_GPT2_IO_H
#define TK_GPT2_IO_H

#include "read_safetensors.h"                              /* struct Weight_Meta */
#include "gpt2.h"           /* struct tk_gpt2, struct tk_gpt2_config */
#include "tf_block.h"            /* struct TransformerBlock */

int tk_gpt2_config_from_json(const char* cfg_path, struct tk_gpt2_config* out_cfg);
struct tk_tensor* find_dest_tensor (struct tk_gpt2* model, struct Weight_Meta* wm);
int load_qkv_split(struct TransformerBlock* tf, FILE* f, struct Weight_Meta* wm);
int load_qkv_bias_split(struct TransformerBlock* tf, FILE* f, struct Weight_Meta* wm);
int tk_gpt2_safetensors_load(const char* path, const char* cfg_path, struct tk_gpt2_config* out_config, struct tk_gpt2* model);

#endif
