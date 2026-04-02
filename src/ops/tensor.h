#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include "arena.h"

#define TK_DISPATCH_TYPES(dtype, name, ...) \
    do { \
        switch(dtype) { \
            case TK_F64: { typedef double scalar_t; __VA_ARGS__ break; } \
            case TK_F32: { typedef float  scalar_t; __VA_ARGS__ break; } \
            case TK_I16:  { typedef int16_t scalar_t; __VA_ARGS__ break; } \
            case TK_I8:  { typedef int8_t scalar_t; __VA_ARGS__ break; } \
            case TK_U8:  { typedef int8_t scalar_t; __VA_ARGS__ break; } \
            default: printf("Unsupported type in %s", name); \
        } \
    } while (0)

enum tk_dtype {
    TK_NONE,
    TK_F64,
    TK_F32,
    TK_I16,
    TK_I8,
    TK_U8
};

struct tk_tensor {
    enum tk_dtype dtype;
    void* data;
    int ndims;
    int* shape;
    int* strides;
};

uint64_t shape_size_calc(int* shape, int ndims);
void strides_calc(int* strides, int* shape, int ndims);
struct tk_tensor* tk_tensor_alloc(struct arena* a,
                                  enum tk_dtype dtype,
                                  int* shape,
                                  int ndims);

int tk_tensor_reshape(struct arena* a, struct tk_tensor* src, struct tk_tensor** dest_ptr, int* new_shape, int new_ndims);
int tk_tensor_is_contiguous(struct tk_tensor* tk);
size_t tk_get_dtype_size(enum tk_dtype dtype);
void tk_tensor_data_reorder(struct tk_tensor* src, struct tk_tensor* dest);
struct tk_tensor* tk_tensor_view(struct arena* a, struct tk_tensor* src, int* new_shape, int new_ndims);
struct tk_tensor* tk_tensor_copy(struct arena* meta_a, struct arena* data_a, struct tk_tensor* src);
void tk_tensor_fill_zero(struct tk_tensor* tensor);
void tk_tensor_padding(struct tk_tensor* src, struct tk_tensor* dest, int pad_h, int pad_w);
int tk_tensor_transpose(struct arena* a, struct tk_tensor* tk_tensor, int dim1, int dim2);
void tk_tensor_relu(struct tk_tensor* src);
int tk_tensor_load_data(struct tk_tensor* dest, FILE* fp, size_t size);

#endif
