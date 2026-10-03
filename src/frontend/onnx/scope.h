#ifndef TK_ONNX_SCOPE_H
#define TK_ONNX_SCOPE_H

#include "hashmap.h"

struct tk_onnx_scope {
    struct tk_onnx_scope* parent;
    struct tk_hashmap symbols;
};

int tk_scope_init(struct tk_onnx_scope* sc, struct tk_onnx_scope* parent);
int tk_scope_symbol_insert(struct tk_onnx_scope* sc, const char* key, size_t key_len, void* data);
void* tk_scope_symbol_find(struct tk_onnx_scope* sc, const char* key, size_t key_len);
void tk_scope_destroy(struct tk_onnx_scope* sc);


#endif
