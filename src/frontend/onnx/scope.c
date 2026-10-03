#include "scope.h"
#include "hashmap.h"
#include "rt_error.h"

int tk_scope_init(struct tk_onnx_scope* sc, struct tk_onnx_scope* parent) {
    if (!sc) {
        RT_FAIL(RT_EINVAL, "Scope is NULL\n");
    }
    sc->symbols = (struct tk_hashmap){0};
    tk_hashmap_init(&sc->symbols, DEFAULT_SIZE);
    sc->parent = parent;
    return 0;
}

int tk_scope_symbol_insert(struct tk_onnx_scope* sc, const char* key, size_t key_len, void* data) {
    if (!sc) {
        RT_FAIL(RT_EINVAL, "Scope is NULL\n");
    }
    RT_CHECK(tk_hashmap_insert(&sc->symbols, key, key_len, data));
    return 0;
}

// for onnx, inner scope can find the key in outer scope
// but not vice versa
// caller is responsible for type-changing
void* tk_scope_symbol_find(struct tk_onnx_scope* sc, const char* key, size_t key_len) {

    struct tk_onnx_scope* cur_sc = sc;
    // remember to move the scope upward
    struct tk_hashmap_entry* ent = NULL;
    do {
        ent = tk_hashmap_get(&cur_sc->symbols, key, key_len);
        cur_sc = cur_sc->parent;
    } while (cur_sc && !ent);
    if (!ent)
        return NULL;
    return ent->val;
}

void tk_scope_destroy(struct tk_onnx_scope* sc) {
    tk_hashmap_destroy(&sc->symbols);
}
