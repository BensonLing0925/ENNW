#include "../../error/rt_error.h"
#include "../../ops/tensor.h"
#include "rt_workspaces.h"

struct tk_workspace* tk_ws_create(struct arena* root_arena, void* data_ptr) {
    struct tk_workspace* ws = arena_alloc(root_arena, sizeof(struct tk_workspace));
    
    uintptr_t addr = (uintptr_t)data_ptr;
    uintptr_t aligned_addr = (addr + 63) & ~63;
    
    ws->arena_base = (void*)aligned_addr;
    // ws->capacity = cap - (aligned_addr - addr); // deduct wasted space for alignment
    ws->cur_offset = 0;
    ws->peak_offset = 0;
    
    return ws;
}

// repeatedly called for both dry run and actual run
void* tk_ws_alloc(struct tk_workspace* ws, size_t size) {
    
    // calculate alignment (64-alignment)
    size_t aligned_size = (size + 63) & ~63;
    void* result_addr = NULL;

    // used during actual run
    if (!ws->is_dryrun) {
        // something very wrong here
        // meaning dry run's estimated size is incorrect
        if (ws->cur_offset + aligned_size > ws->capacity) {
            RT_FAIL(RT_EOOM, "Insufficient workspace memory during inference");
            return NULL;
        }
        result_addr = (uint8_t*)ws->arena_base + ws->cur_offset;
    }

    // move offset
    ws->cur_offset += aligned_size;

    if (ws->cur_offset > ws->peak_offset) {
        ws->peak_offset = ws->cur_offset;
    }

    return result_addr;
}

struct tk_tensor* tk_ws_tensor_alloc(struct tk_workspace* ws, struct arena* meta_arena, enum tk_dtype dtype, int* shape, int ndims) {
    // meta data stored in meta_arena
    struct tk_tensor* tk = arena_alloc(meta_arena, sizeof(struct tk_tensor));
    
    tk->shape = arena_alloc(meta_arena, sizeof(int) * ndims);
    tk->strides = arena_alloc(meta_arena, sizeof(int) * ndims);
    memcpy(tk->shape, shape, sizeof(int) * ndims);
    strides_calc(tk->strides, shape, ndims);
    tk->ndims = ndims;
    tk->dtype = dtype;

    // alloc actual data from data_arena
    uint64_t total_bytes = shape_size_calc(shape, ndims) * tk_get_dtype_size(dtype);
    
    // for dry run, tk_ws_alloc return NULL
    // for actual run, tk_ws_alloc return actual address
    // both move offset
    tk->data = tk_ws_alloc(ws, total_bytes); 

    return tk;
}





