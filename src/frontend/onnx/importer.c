#include "arena.h"
#include "fileio.h"
#include "rt_context.h"
#include "tensor.h"
#include "onnx.proto3.pb-c.h"
#include "onnx/importer.h"
#include <stdio.h>
#include <stdlib.h>

// call tk_onnx_model_free to free
Onnx__ModelProto* tk_onnx_model_get(struct tk_file_data* f) {
    if (!f) {
        fprintf(stderr, "Input is NULL\n");
        return NULL;
    }
    Onnx__ModelProto* model = onnx__model_proto__unpack(
            NULL, f->bytes, f->data);

    if (!model) {
        fprintf(stderr, "Failed to unpack ONNX model\n");
        return NULL;
    }
    return model;
}

void tk_onnx_model_free(Onnx__ModelProto* model) {
    onnx__model_proto__free_unpacked(model, NULL);
}

static enum tk_dtype tk_onnx_to_tk_dtype(int32_t onnx_dtype) {
    switch (onnx_dtype) {
        case 0:
            fprintf(stderr, "Undefined ONNX data type\n");
            break;
        case 1:
            return TK_F32;
        case 2:
            return TK_U8;
        case 3:
            return TK_I8;
        case 5:
            return TK_I16;
        case 6:
            return TK_I32;
        case 7:
            return TK_I64;
        case 11:
            return TK_F64;
        default:
            fprintf(stderr, "Unsupported TK data type\n");
    }
}

// specifically for raw_data copy
static int tk_onnx_raw_copy(Onnx__TensorProto* t, struct tk_tensor* tensor) {
    if (!t || !tensor || !t->raw_data.data) {
        RT_FAIL(RT_EINVAL, "Invalid tensor/raw data");
    }
    if (t->raw_data.len == 0) {
        RT_FAIL(RT_EINVAL, "ONNX tensor raw_data is empty");
    }
    // validate size
    size_t expected = shape_size_calc(tensor->shape, tensor->ndims) * tk_dtype_size(tensor->dtype);
    if (expected != t->raw_data.len) {
        RT_FAIL(RT_EINVAL, "Number of bytes mismatched between onnx and tk tensor");
    }
    memcpy(tensor->data, t->raw_data.data, t->raw_data.len);
    return 0;
}

// use specifically for debug and
// figuring out the field in which the tensor's data stored
int tk_onnx_tensor_data_copy(const Onnx__TensorProto *t, struct tk_tensor* tensor) {

    if (t->data_location == ONNX__TENSOR_PROTO__DATA_LOCATION__EXTERNAL) {
        fprintf(stderr, "Unsupported Data Location\n");
        return;
    }
    // explicitly set means store in raw_data field
    // DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
    else if (t->raw_data.len > 0) {
        RT_CHECK(tk_onnx_raw_copy(t, tensor)); 
        return 0;
    }
    // type field check
    else {
        size_t numel = shape_size_calc(tensor->shape, tensor->ndims);

        switch (t->data_type) {

            case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
                if (t->n_float_data != numel)
                    RT_FAIL(RT_EINVAL, "FLOAT element count mismatch");
                memcpy(tensor->data,
                       t->float_data,
                       numel * sizeof(float));
                break;

            case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
                if (t->n_double_data != numel)
                    RT_FAIL(RT_EINVAL, "DOUBLE element count mismatch");
                memcpy(tensor->data,
                       t->double_data,
                       numel * sizeof(double));
                break;

            case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
                if (t->n_int32_data != numel)
                    RT_FAIL(RT_EINVAL, "INT32 element count mismatch");
                memcpy(tensor->data,
                       t->int32_data,
                       numel * sizeof(int32_t));
                break;

            case ONNX__TENSOR_PROTO__DATA_TYPE__INT8: {
                if (t->n_int32_data != numel)
                    RT_FAIL(RT_EINVAL, "INT8 element count mismatch");
                int8_t *dst = tensor->data;
                for (size_t i = 0; i < numel; ++i)
                    dst[i] = (int8_t)t->int32_data[i];
                break;
            }

            case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8: {
                if (t->n_int32_data != numel)
                    RT_FAIL(RT_EINVAL, "UINT8 element count mismatch");
                uint8_t *dst = tensor->data;
                for (size_t i = 0; i < numel; ++i)
                    dst[i] = (uint8_t)t->int32_data[i];
                break;
            }

            case ONNX__TENSOR_PROTO__DATA_TYPE__INT16: {
                if (t->n_int32_data != numel)
                    RT_FAIL(RT_EINVAL, "INT16 element count mismatch");
                int16_t *dst = tensor->data;
                for (size_t i = 0; i < numel; ++i)
                    dst[i] = (int16_t)t->int32_data[i];
                break;
            }

            case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
                if (t->n_int64_data != numel)
                    RT_FAIL(RT_EINVAL, "INT64 element count mismatch");
                memcpy(tensor->data,
                       t->int64_data,
                       numel * sizeof(int64_t));
                break;

            default:
                RT_FAIL(RT_EINVAL,
                        "Unsupported ONNX tensor dtype %d",
                        t->data_type);
        }

        return 0;
    }
}

// called when parsing initializers
// callee allocate tensors
// out only needs to be a valid pointer
int tk_onnx_to_tensor(struct tk_rt_ctx* ctx, Onnx__TensorProto* t, struct tk_tensor** out) {
    struct tk_tensor* tensor = NULL;
    enum tk_dtype dtype = tk_onnx_to_tk_dtype(t->data_type);
    RT_CHECK(tk_tensor_alloc(ctx->data_arena, dtype, t->dims, t->n_dims, &tensor));
    RT_CHECK(tk_onnx_tensor_data_copy(t, tensor));
    *out = tensor;
    return 0;
}

// ONNX defined specific schema for different node->op_type
// the schema also differs according to different opset version
// currently using version 13 to first run the gpt2_q8_Xenova model
static int tk_onnx_constant_to_tensor(struct tk_rt_ctx* ctx, Onnx__NodeProto* node, struct tk_onnx_scope* cur_scope) {
    // for Constant, there should be only one output
    if (!node->attribute)
        RT_FAIL(RT_EINVAL, "Constant Node Missing Attribute\n");
    Onnx__AttributeProto* attr = node->attribute[0];
    struct tk_tensor* tensor = NULL;
    switch (attr->type) {
        case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
            RT_CHECK(tk_onnx_to_tensor(ctx, attr->t, &tensor));
            break;
        default:
            break;

    }
    RT_CHECK(tk_scope_symbol_insert(cur_scope, node->output[0], strlen(node->output[0]), (void*)tensor));
    return 0;
}

static int tk_onnx_shape_import(struct tk_rt_ctx* ctx, Onnx__NodeProto* node, struct tk_onnx_scope* cur_scope) {
    // shape should have one input and one output
    // input a tensor, output its shape with int64_t[]
    if (node->n_input != 1 || node->n_output != 1)
        RT_FAIL(RT_EINVAL, "Shape Node has wrong amount of input/output");
    struct tk_tensor *input = tk_scope_symbol_find(cur_scope, node->input[0], strlen(node->input[0]));
    
    if (!input)
        RT_FAIL(RT_EINVAL, "Input is not registered in the symbol table");

    // since not all tensor's metadata is known(static)
    // only adds entry but we are not done yet

    /* if input:
     * input->shape = [-1, -1];
     * input->ndims = 2 (decided);
     */
    struct tk_tensor* tensor = NULL;
    // shape output an 1D, int64 vector
    int output_shape[] = {input->ndims};
    RT_CHECK(tk_tensor_create(ctx->data_arena, TK_I64, output_shape, 1, &tensor));
    RT_CHECK(tk_scope_symbol_insert(cur_scope, node->output[0], strlen(node->output[0]), (void*)tensor));
    return 0;
}

int tk_onnx_node_import(struct tk_rt_ctx* ctx, Onnx__NodeProto* node, struct tk_onnx_scope* cur_scope) {
    if (!node)
        RT_FAIL(RT_EINVAL, "node is NULL");

    if (!cur_scope)
        RT_FAIL(RT_EINVAL, "cur_scope is NULL");
    
    if (!strcmp(node->op_type, "Constant"))
        RT_CHECK(tk_onnx_constant_to_tensor(ctx, node, cur_scope));

}

// recursively called to handle nested graphs
// each time called, create a new scope
int tk_onnx_graph_import(struct tk_rt_ctx* ctx, Onnx__GraphProto* g, struct tk_onnx_scope* parent) {

    struct tk_onnx_scope sc = {0};
    RT_CHECK(tk_scope_init(&sc, parent));
    int rc = 0;
    
    // initializer to scope's hashmap
    for ( size_t t = 0 ; t < g->n_initializer ; ++t ) {
        Onnx__TensorProto* tp = g->initializer[t];
        struct tk_tensor* tensor = NULL;
        RT_CHECK_GOTO(tk_onnx_to_tensor(ctx, tp, &tensor), rc, cleanup);
        RT_CHECK_GOTO(tk_scope_symbol_insert(&sc, tp->name, strlen(tp->name), (void*)tensor), rc, cleanup);
    }
    // walk the graph and create nodes, recursively
    for ( size_t n = 0 ; n < g->n_node ; ++n ) {
        Onnx__NodeProto* node = g->node[n];
        // attribute proto can only have one field with content
        // enforcing C Union equivalent
        tk_onnx_node_import(ctx, node, &sc);
        /*
        if (node->n_attribute >= 1) {
            for ( size_t a = 0 ; a < node->n_attribute ; ++a ) {
                Onnx__AttributeProto* attr = node->attribute[a];
                switch (attr->type) {
                    case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH:
                        RT_CHECK_GOTO(tk_onnx_graph_import(ctx, attr->g, &sc), rc, cleanup);
                        break;
                    default:
                        break;
                }
            }
        }
        */
        // build the node
        // ...
    }
    return 0;

cleanup:
    tk_scope_destroy(&sc);
    return rc;
}
