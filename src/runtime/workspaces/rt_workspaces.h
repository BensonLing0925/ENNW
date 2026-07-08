#ifndef RT_WORKSPACE_H
#define RT_WORKSPACE_H

#include <stdint.h>
#include "../../ops/tensor.h"

#define TK_WS_BEGIN(ctx) size_t _ws_save = (ctx)->ws->cur_offset
#define TK_WS_END(ctx) (ctx)->ws->cur_offset = _ws_save

struct tk_workspace {
    void* arena_base;
    size_t capacity;
    size_t cur_offset;
    size_t peak_offset;
    int is_dryrun;
    int last_err;
};

struct tk_workspace* tk_ws_create(struct arena* root_arena, void* data_ptr);
int tk_ws_alloc(struct tk_workspace* ws, size_t size, void** out);
int tk_ws_tensor_alloc(struct tk_workspace* ws, struct arena* meta_arena, enum tk_dtype dtype, int* shape, int ndims, struct tk_tensor** out);
#endif
