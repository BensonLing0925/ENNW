#include "../error/rt_error.h"
#include "arena.h"
#include "tensor.h"
#include <inttypes.h>

// stack alloc
#define TK_TENSOR_INIT(tensor) \
    { TK_NONE, NULL, 0, NULL, NULL }

#define TK_TENSOR_CREATE(name) \
    struct tk_tensor name = TK_TENSOR_INIT(name)

uint64_t shape_size_calc(int* shape, int ndims) {
    uint64_t size = 1;
    for ( int i = 0 ; i < ndims ; ++i )
        size *= shape[i];
    return size;
}

void strides_calc(int* strides, int* shape, int ndims) {
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

size_t tk_get_dtype_size(enum tk_dtype dtype) {
    switch(dtype) {
        case TK_F64: return sizeof(double);
        case TK_F32: return sizeof(float);
        case TK_I16: return sizeof(int16_t);
        case TK_I8:  return sizeof(int8_t);
        case TK_U8:  return sizeof(uint8_t);
        default:     return 0;
    }
}


void tk_tensor_data_reorder(struct tk_tensor* src, struct tk_tensor* dest) {

    int* indices = calloc(src->ndims, sizeof(int));    // eg: [0, 0, 0]
    int ndims = src->ndims;
    uint64_t total_size = shape_size_calc(src->shape, src->ndims);
    TK_DISPATCH_TYPES(src->dtype, __func__, {
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

int tk_tensor_reshape(struct arena* a, struct tk_tensor* src, struct tk_tensor** dest_ptr, int* new_shape, int new_ndims) {
    // allocate a new struct tk_tensor but data points to src->data (Zero-copy)
    // recalculate shape and strides
    uint64_t new_size = shape_size_calc(new_shape, new_ndims);
    uint64_t size = shape_size_calc(src->shape, src->ndims);
    if (new_size != size) {
        RT_FAIL(RT_EINVAL, "Tensor and reshape mismatch %PRIu64 and %PRIu64",
                            size, new_size);
    }

    struct tk_tensor* dest;

    // eg. [2, 3, 4] -> [2, 12]
    // eg. [2, 3, 4] -> [8, 3]

    // transposed, need realloc and copy data
    if (!tk_tensor_is_contiguous(src)) {
        dest = tk_tensor_alloc(a, src->dtype, src->shape, src->ndims);
        tk_tensor_data_reorder(src, dest); 
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

    *dest_ptr = dest;
    
    return 0;
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
    tk_tensor_reshape(a, tk_tensor, &tk_tensor, tk_tensor->shape, tk_tensor->ndims);
    return 0;
}

struct tk_tensor* tk_tensor_view(struct arena* a, struct tk_tensor* src, int* new_shape, int new_ndims) {

    if (!src)
        return NULL;

    // 只在 meta_arena 拿一個載體
    struct tk_tensor* view = arena_alloc(a, sizeof(struct tk_tensor));
    
    view->dtype = src->dtype;
    view->shape = new_shape;
    view->ndims = new_ndims;
    
    // 載體指向同一個 data，不需要調用 tk_ws_alloc，所以不增加 peak_offset
    view->data = src->data; 
    
    return view;
}

struct tk_tensor* tk_tensor_copy(struct arena* meta_a, struct arena* data_a, struct tk_tensor* src) {
    struct tk_tensor* dst = arena_alloc(meta_a, sizeof(struct tk_tensor));
    
    memcpy(dst, src, sizeof(struct tk_tensor));

    uint64_t num_elements = shape_size_calc(src->shape, src->ndims);
    uint64_t bytes = num_elements * tk_get_dtype_size(src->dtype);

    dst->data = arena_alloc(data_a, bytes);

    if (src->data && dst->data) {
        memcpy(dst->data, src->data, bytes);
    }

    return dst;
}

void tk_tensor_fill_zero(struct tk_tensor* tensor) {
    uint64_t size = shape_size_calc(tensor->shape, tensor->ndims);
    TK_DISPATCH_TYPES(tensor->dtype, __func__, {
                memset(tensor->data, 0, sizeof(scalar_t) * size);
                });
}

static void tk_tensor_padding_data_move(struct tk_tensor* dest, struct tk_tensor* src,
                                        int pad_h, int pad_w) {
    int src_h = src->shape[src->ndims-2];
    int src_w = src->shape[src->ndims-1];
    int dest_h = dest->shape[dest->ndims-2];
    int dest_w = dest->shape[dest->ndims-1];
    for ( int inner_h = pad_h-1 ; inner_h < dest_h - pad_h ; ++inner_h ) {
        for ( int inner_w = pad_w-1 ; inner_w < dest_w - pad_w; ++inner_w ) {
            TK_DISPATCH_TYPES(src->dtype, __func__, {
                scalar_t* src_ptr = src->data;
                scalar_t* dest_ptr = dest->data;
                dest_ptr[inner_h * (pad_w + src_w) + inner_w] = src_ptr[inner_h * src_w + inner_w];
                src_w++;
            });
        } 
        src_h++;
        src_w = 0;
    }
}

void tk_tensor_padding(struct tk_tensor* src, struct tk_tensor* dest, int pad_h, int pad_w) {
    int in_h = src->shape[src->ndims-1];
    int in_w = src->shape[src->ndims-2];
    int out_h = in_h + 2 * pad_h;
    int out_w = in_w + 2 * pad_w;

    int out_shape[src->ndims];
    memcpy(out_shape, src->shape, sizeof(int) * src->ndims);
    out_shape[src->ndims-1] = out_h;
    out_shape[src->ndims-2] = out_w;
    tk_tensor_fill_zero(dest);
    tk_tensor_padding_data_move(dest, src, pad_h, pad_w);
}

static void _tk_tensor_relu_2d(struct tk_tensor* src) {

    int src_h = src->shape[src->ndims-2];
    int src_w = src->shape[src->ndims-1];

	for ( int i = 0 ; i < src_h ; ++i ) {
		for ( int j = 0 ; j < src_w; ++j ) {
            TK_DISPATCH_TYPES(src->dtype, __func__, {
                scalar_t* src_ptr = src->data;
                if ( src_ptr[i * src_w + j] < 0.000 )
                    src_ptr[i * src_w + j] = 0.000;
            });
		}		
	}		
}	

void tk_tensor_relu(struct tk_tensor* src) {
    int planes = 1;
    for (int i = 0; i < src->ndims - 2; ++i) {
        planes *= src->shape[i];
    }

    int pic_plane_size = src->shape[src->ndims-2] * src->shape[src->ndims-1];

    for (int p = 0; p < planes; ++p) {
        struct tk_tensor pic_view = *src;
        pic_view.data = src->data + (p * pic_plane_size);

        _tk_tensor_relu_2d(&pic_view);
    }
}

int tk_tensor_load_data(struct tk_tensor* dest, FILE* fp, size_t size) {
    int ndims = dest->ndims;
    int* shape = dest->shape;

    uint64_t cur_size = shape_size_calc(shape, ndims);
    if (cur_size != size) {
        RT_FAIL(RT_EINVAL, "Tensor and reshape mismatch %PRIu64 and %PRIu64",
                            cur_size, size);
    }

    fread(dest->data, tk_get_dtype_size(dest->dtype), size, fp);
    return 0;
}
