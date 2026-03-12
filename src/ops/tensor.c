#include "../error/rt_error.h"
#include "arena.h"
#include "tensor.h"
#include <inttypes.h>

// stack alloc
#define TK_TENSOR_INIT(tensor) \
    { TK_NONE, NULL, 0, NULL, NULL }

#define TK_TENSOR_CREATE(name) \
    struct tk_tensor name = TK_TENSOR_INIT(name)

static uint64_t shape_size_calc(int* shape, int ndims) {
    uint64_t size = 1;
    for ( int i = 0 ; i < ndims ; ++i )
        size *= shape[i];
    return size;
}

static void strides_calc(int* strides, int* shape, int ndims) {
    int s = 1;
    for ( int i = ndims-1 ; i >= 0 ; --i ) {
        strides[i] = s;
        s *= shape[i];
    }
}

int tk_tensor_is_contiguous(struct tk_tensor* tk) {
    if (tk->ndims == 1) return 1;
    if (tk->strides[tk->ndims-1] != 1) return 0;
    int expected_stride = 1;
    for ( int i = tk->ndims - 1 ; i > 0 ; --i ) {
        if (tk->shape[i] > 1) {
            if (expected_stride != tk->strides[i]) 
                return 0;
        }
        expected_stride *= tk->shape[i];
    }
    return 1;
}

static void tk_tensor_data_copy(struct tk_tensor* src, struct tk_tensor* dest) {

    int* indices = calloc(src->ndims, sizeof(int));    // eg: [0, 0, 0]
    uint64_t total_size = shape_size_calc(src->shape, src->ndims);
    TK_DISPATCH_TYPES(src_dtype, __func__, {
        scalar_t* src_ptr = (scalar_t*)src->data;
        scalar_t* dest_ptr = (scalar_t*)dest->data;
        for ( uint64_t n = 0 ; n < total_size ; ++n ) {
            // calculate offset first
            uint64_t src_offset = 0;
            for ( int d = src->ndims-1 ; d >= 0 ; --d ) {
                src_offset += (indices[d] * src->strides[d]);
            }

            dest_ptr[n] = src_ptr[src_offset];

            for ( int d = ndims - 1 ; d >= 0 ; --d ) {
                indices[d]++;
                if (indices[d] < src->shape[d])
                    break;
                else
                    indices[d] = 0;
            } 
        }
    });
    free(indices);
}

// heap alloc
struct tk_tensor* tk_tensor_alloc(struct arena* a,
                                  enum tk_dtype dtype,
                                  int* shape,
                                  int ndims) {
    struct tk_tensor* tk = arena_alloc(a, sizeof(struct tk_tensor));
    if (!tk) {
        RT_FAIL(RT_EOOM, "Out of memory"); 
    }

    uint64_t size = shape_size_calc(shape, ndims);

    TK_DISPATCH_TYPES(dtype, __func__, {
                tk->data = (void*) arena_alloc(a, sizeof(scalar_t) * size); 
                tk->strides = (int*) arena_alloc(a, sizeof(int) * ndims);
                strides_calc(tk->strides, shape, ndims);
                tk->dtype = dtype;
                });

    tk->ndims = ndims;
    tk->shape = arena_alloc(a, sizeof(int) * ndims);
    memcpy(tk->shape, shape, sizeof(int) * ndims);

    return tk;
}

struct tk_tensor* tk_tensor_reshape(struct arena* a, struct tk_tensor* src, int* new_shape, int new_ndims) {
    // allocate a new struct tk_tensor but data points to src->data (Zero-copy)
    // recalculate shape and strides
    uint64_t new_size = shape_size_calc(new_shape, new_dims);
    uint64_t size = shape_size_calc(src->shape, src->ndims);
    if (new_size != size) {
        RT_FAIL(RT_EINVAL, "Tensor and reshape mismatch %PRIu64 and %PRIu64",
                            size, new_size);
    }

    struct tk_tensor* dest = NULL;
    // eg. [2, 3, 4] -> [2, 12]
    // eg. [2, 3, 4] -> [8, 3]

    // transposed, need realloc and copy data
    if (!tk_tensor_is_contiguous(src)) {
        dest = tk_tensor_alloc(a, src->dtype, src->shape, src->ndims);
        tk_tensor_data_copy(src, dest); 
    }
    else {
        dest = arena_alloc(a, sizeof(struct tk_tensor));
        dest->dtype = src->dtype;
        dest->ndims = new_ndims;
        dest->shape = arena_alloc(a, sizeof(int) * new_ndims);
        dest->strides = arena_alloc(a, sizeof(int) * new_ndims);
        memcpy(dest->shape, new_shape, sizeof(int) * new_ndims);
        strides_calc(dest->strides, dest->shape, dest->ndims);
        dest->data = src->data; // Zero-copy
    }
    
    return dest;
}

int tk_tensor_transpose(struct arena* a,
                        struct tk_tensor* tk_tensor, 
                        int dim1,
                        int dim2) {
    // check index and stride length
    if (tk_tensor->ndims < dim1 || tk_tensor->ndims < dim2)
        RT_FAIL(RT_EINVAL, "Out of bound dimension index. tk_tensor->ndims: %d, dim1: %d, dim2: %d\n",
                            tk_tensor->ndims, dim1, dim2);
        
    // eager transpose
    // make tensor incontiguous
    int tmp_stride = tk_tensor->strides[dim1];
    tk_tensor->strides[dim1] = tk_tensor->strides[dim2];
    tk_tensor->strides[dim2] = tmp_stride;

    int tmp_shape = tk_tensor->shape[dim1];
    tk_tensor->shape[dim1] = tk_tensor->shape[dim2];
    tk_tensor->shape[dim2] = tmp_shape;

    // we use arena, just update the pointer and we are done
    struct tk_tensor* result = tk_tensor_reshape(a, tk_tensor, tk_tensor->shape, tk_tensor->ndims);
    *tk_tensor = result;
    return 0;
}

