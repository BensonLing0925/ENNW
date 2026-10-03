#include "onnx.proto3.pb-c.h"
#include "onnx/importer.h"
#include "fileio.h"
#include "arena.h"
#include "rt_context.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

void test_constant_scalar_i64(void) {
    struct arena arena;
    arena_init(&arena);

    struct tk_rt_ctx ctx = {0};
    ctx.data_arena = &arena;

    int64_t value = 42;

    Onnx__TensorProto tp = ONNX__TENSOR_PROTO__INIT;
    tp.data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;
    tp.n_dims = 0;
    tp.dims = NULL;
    tp.n_int64_data = 1;
    tp.int64_data = &value;

    Onnx__AttributeProto attr = ONNX__ATTRIBUTE_PROTO__INIT;
    attr.name = "value";
    attr.type =
        ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR;
    attr.t = &tp;

    Onnx__AttributeProto *attrs[] = {
        &attr
    };

    char *outputs[] = {
        "/test/Constant_output_0"
    };

    struct tk_onnx_scope sc = {0};
    RT_CHECK(tk_scope_init(&sc, NULL));

    Onnx__NodeProto node = ONNX__NODE_PROTO__INIT;
    node.op_type = "Constant";
    node.n_attribute = 1;
    node.attribute = attrs;
    node.n_output = 1;
    node.output = outputs;

    struct tk_tensor *tensor = NULL;

    int rc = tk_onnx_node_import(
        &ctx,
        &node,
        &sc
        );

    assert(rc == 0);

    // find the tensor via scope's hashmap
    tensor = (struct tk_tensor*)tk_scope_symbol_find(&sc, outputs[0], strlen(outputs[0]));

    assert(tensor != NULL);

    assert(tensor->dtype == TK_I64);
    assert(tensor->ndims == 0);
    assert(tensor->shape == NULL);
    assert(tensor->strides == NULL);

    assert(*(int64_t *)tensor->data == 42);

    tk_scope_destroy(&sc);
    arena_destroy(&arena);

    printf("test_constant_scalar_i64 passed\n");
}

void test_constant_vector_i64(void) {
    struct arena arena;
    arena_init(&arena);

    struct tk_rt_ctx ctx = {0};
    ctx.data_arena = &arena;

    int64_t value = 42;
    int64_t dims[] = {1};

    Onnx__TensorProto tp = ONNX__TENSOR_PROTO__INIT;
    tp.data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;
    tp.n_dims = 1;
    tp.dims = dims;
    tp.n_int64_data = 1;
    tp.int64_data = &value;

    Onnx__AttributeProto attr = ONNX__ATTRIBUTE_PROTO__INIT;
    attr.name = "value";
    attr.type =
        ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR;
    attr.t = &tp;

    Onnx__AttributeProto *attrs[] = {
        &attr
    };

    char *outputs[] = {
        "/test/Constant_vector_i64_output_0"
    };

    struct tk_onnx_scope sc = {0};
    assert(tk_scope_init(&sc, NULL) == 0);

    Onnx__NodeProto node = ONNX__NODE_PROTO__INIT;
    node.op_type = "Constant";
    node.n_attribute = 1;
    node.attribute = attrs;
    node.n_output = 1;
    node.output = outputs;

    int rc = tk_onnx_node_import(
        &ctx,
        &node,
        &sc
    );

    assert(rc == 0);

    struct tk_tensor *tensor =
        tk_scope_symbol_find(
            &sc,
            outputs[0],
            strlen(outputs[0])
        );

    assert(tensor != NULL);

    assert(tensor->dtype == TK_I64);

    // [1], not scalar []
    assert(tensor->ndims == 1);
    assert(tensor->shape != NULL);
    assert(tensor->shape[0] == 1);

    assert(tensor->strides != NULL);
    assert(tensor->strides[0] == 1);

    assert(tensor->data != NULL);
    assert(((int64_t *)tensor->data)[0] == 42);

    tk_scope_destroy(&sc);
    arena_destroy(&arena);

    printf("test_constant_vector_i64 passed\n");
}

void test_constant_empty_vector_i64(void) {
    struct arena arena;
    arena_init(&arena);

    struct tk_rt_ctx ctx = {0};
    ctx.data_arena = &arena;

    int64_t dims[] = {0};

    Onnx__TensorProto tp = ONNX__TENSOR_PROTO__INIT;
    tp.data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;

    tp.n_dims = 1;
    tp.dims = dims;

    tp.n_int64_data = 0;
    tp.int64_data = NULL;

    Onnx__AttributeProto attr = ONNX__ATTRIBUTE_PROTO__INIT;
    attr.name = "value";
    attr.type =
        ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR;
    attr.t = &tp;

    Onnx__AttributeProto *attrs[] = {
        &attr
    };

    char *outputs[] = {
        "/test/Constant_empty_vector_output_0"
    };

    struct tk_onnx_scope sc = {0};
    assert(tk_scope_init(&sc, NULL) == 0);

    Onnx__NodeProto node = ONNX__NODE_PROTO__INIT;
    node.op_type = "Constant";
    node.n_attribute = 1;
    node.attribute = attrs;
    node.n_output = 1;
    node.output = outputs;

    int rc = tk_onnx_node_import(
        &ctx,
        &node,
        &sc
    );

    assert(rc == 0);

    struct tk_tensor *tensor =
        tk_scope_symbol_find(
            &sc,
            outputs[0],
            strlen(outputs[0])
        );

    assert(tensor != NULL);

    assert(tensor->dtype == TK_I64);

    assert(tensor->ndims == 1);
    assert(tensor->shape != NULL);
    assert(tensor->shape[0] == 0);

    assert(tensor->strides != NULL);

    /*
     * Important:
     * numel should be 0.
     */
    assert(shape_size_calc(
        tensor->shape,
        tensor->ndims
    ) == 0);

    tk_scope_destroy(&sc);
    arena_destroy(&arena);

    printf("test_constant_empty_vector_i64 passed\n");
}

void test_constant_missing_value(void) {
    struct arena arena;
    arena_init(&arena);

    struct tk_rt_ctx ctx = {0};
    ctx.data_arena = &arena;

    char *outputs[] = {
        "/test/Constant_missing_value_output_0"
    };

    struct tk_onnx_scope sc = {0};
    assert(tk_scope_init(&sc, NULL) == 0);

    Onnx__NodeProto node = ONNX__NODE_PROTO__INIT;
    node.op_type = "Constant";

    // Deliberately malformed Constant.
    node.n_attribute = 0;
    node.attribute = NULL;

    node.n_output = 1;
    node.output = outputs;

    int rc = tk_onnx_node_import(
        &ctx,
        &node,
        &sc
    );

    assert(rc < 0);

    // Failed import should not register an output symbol.
    struct tk_tensor *tensor =
        tk_scope_symbol_find(
            &sc,
            outputs[0],
            strlen(outputs[0])
        );

    assert(tensor == NULL);

    tk_scope_destroy(&sc);
    arena_destroy(&arena);

    printf("test_constant_missing_value passed\n");
}

void test_constant_vector_f32(void) {
    struct arena arena;
    arena_init(&arena);

    struct tk_rt_ctx ctx = {0};
    ctx.data_arena = &arena;

    float values[] = {
        1.5f,
        2.5f
    };

    int64_t dims[] = {2};

    Onnx__TensorProto tp = ONNX__TENSOR_PROTO__INIT;
    tp.data_type = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
    tp.n_dims = 1;
    tp.dims = dims;
    tp.n_float_data = 2;
    tp.float_data = values;

    Onnx__AttributeProto attr = ONNX__ATTRIBUTE_PROTO__INIT;
    attr.name = "value";
    attr.type =
        ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR;
    attr.t = &tp;

    Onnx__AttributeProto *attrs[] = {
        &attr
    };

    char *outputs[] = {
        "/test/Constant_vector_f32_output_0"
    };

    struct tk_onnx_scope sc = {0};
    assert(tk_scope_init(&sc, NULL) == 0);

    Onnx__NodeProto node = ONNX__NODE_PROTO__INIT;
    node.op_type = "Constant";
    node.n_attribute = 1;
    node.attribute = attrs;
    node.n_output = 1;
    node.output = outputs;

    int rc = tk_onnx_node_import(
        &ctx,
        &node,
        &sc
    );

    assert(rc == 0);

    struct tk_tensor *tensor =
        tk_scope_symbol_find(
            &sc,
            outputs[0],
            strlen(outputs[0])
        );

    assert(tensor != NULL);

    assert(tensor->dtype == TK_F32);

    assert(tensor->ndims == 1);
    assert(tensor->shape != NULL);
    assert(tensor->shape[0] == 2);

    assert(tensor->data != NULL);

    float *data = (float *)tensor->data;

    assert(data[0] == 1.5f);
    assert(data[1] == 2.5f);

    tk_scope_destroy(&sc);
    arena_destroy(&arena);

    printf("test_constant_vector_f32 passed\n");
}

void test_constant_scalar_i64_raw_data(void) {
    struct arena arena;
    arena_init(&arena);

    struct tk_rt_ctx ctx = {0};
    ctx.data_arena = &arena;

    int64_t value = 42;

    uint8_t raw[sizeof(int64_t)];
    memcpy(raw, &value, sizeof(value));

    Onnx__TensorProto tp = ONNX__TENSOR_PROTO__INIT;
    tp.data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;

    // scalar
    tp.n_dims = 0;
    tp.dims = NULL;

    // No typed int64_data.
    tp.n_int64_data = 0;
    tp.int64_data = NULL;

    // Use raw_data instead.
    tp.raw_data.data = raw;
    tp.raw_data.len = sizeof(raw);

    Onnx__AttributeProto attr = ONNX__ATTRIBUTE_PROTO__INIT;
    attr.name = "value";
    attr.type =
        ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR;
    attr.t = &tp;

    Onnx__AttributeProto *attrs[] = {
        &attr
    };

    char *outputs[] = {
        "/test/Constant_scalar_i64_raw_output_0"
    };

    struct tk_onnx_scope sc = {0};
    assert(tk_scope_init(&sc, NULL) == 0);

    Onnx__NodeProto node = ONNX__NODE_PROTO__INIT;
    node.op_type = "Constant";
    node.n_attribute = 1;
    node.attribute = attrs;
    node.n_output = 1;
    node.output = outputs;

    int rc = tk_onnx_node_import(
        &ctx,
        &node,
        &sc
    );

    assert(rc == 0);

    struct tk_tensor *tensor =
        tk_scope_symbol_find(
            &sc,
            outputs[0],
            strlen(outputs[0])
        );

    assert(tensor != NULL);

    assert(tensor->dtype == TK_I64);
    assert(tensor->ndims == 0);
    assert(tensor->shape == NULL);

    assert(tensor->data != NULL);
    assert(*(int64_t *)tensor->data == 42);

    tk_scope_destroy(&sc);
    arena_destroy(&arena);

    printf("test_constant_scalar_i64_raw_data passed\n");
}

int main(int argc, char* argv[]) {
    test_constant_scalar_i64();
    test_constant_vector_i64();
    test_constant_missing_value();
    test_constant_vector_f32();
    test_constant_scalar_i64_raw_data();
    test_constant_empty_vector_i64();
    /*
    if (argc != 2) {
        printf("usage: ./executable model.onnx\n");
        return 0;
    }

    struct arena root_arena;
    arena_init(&root_arena);
    // we don't care about the context config for now
    struct tk_rt_ctx_config ctx_config = {
        .use_int8           = 0,
        .use_prof           = 1,
        .use_graph_optimize = 1,
        .graph_capacity     = 1024,
    };
    struct tk_rt_ctx* ctx = tk_runtime_ctx_create_config(&root_arena, ctx_config);
    ctx->compute_dtype = TK_F32;

    struct tk_file_data* r = tk_file_read_all(argv[1]);
    Onnx__ModelProto* model = onnx__model_proto__unpack(
            NULL, r->bytes, r->data); 
    printf("ir_version: %ld\n", model->ir_version);

    for (size_t i = 0; i < model->n_opset_import; ++i) {
        Onnx__OperatorSetIdProto *opset = model->opset_import[i];

        printf("opset domain=%s version=%ld\n",
               opset->domain ? opset->domain : "",
               (long)opset->version);
    }

    printf("nodes: %zu\n", model->graph->n_node);
    printf("initializers: %zu\n", model->graph->n_initializer);

    printf("Printing Initializers...\n");
    Onnx__TensorProto** tensors = model->graph->initializer;
    for ( size_t t = 0 ; t < model->graph->n_initializer ; ++t ) {
        Onnx__TensorProto* tensor = model->graph->initializer[t];
        // printf("tensor [%zu]: %s\n", t, tensor->name);
        // initializer register(table)
        // tk_onnx_to_tensor(ctx, tensor, tensor);
        // graph walk

        printf("\n");

    }

    for (size_t i = 0; i < model->graph->n_node; ++i) {
        Onnx__NodeProto *node = model->graph->node[i];

        printf("node[%zu]: %s\n",
               i,
               node->op_type ? node->op_type : "(null)");
        for (size_t i = 0; i < node->n_input; ++i)
            printf("  input[%zu]: %s\n", i, node->input[i]);

        for (size_t i = 0; i < node->n_output; ++i)
            printf("  output[%zu]: %s\n", i, node->output[i]);


        for (size_t a = 0; a < node->n_attribute; ++a) {
            Onnx__AttributeProto *attr = node->attribute[a];

            printf("  attr[%zu]: %s type=%d\n",
                   a, attr->name, attr->type);

            if (attr->type ==
                ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR) {

                Onnx__TensorProto *t = attr->t;

                printf("    dtype=%d\n", t->data_type);
                printf("    dims=[");

                for (size_t d = 0; d < t->n_dims; ++d)
                    printf("%s%ld", d ? ", " : "", t->dims[d]);

                printf("]\n");

                if (t->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT64) {
                    if (t->n_int64_data == 1) {
                        printf("    value=%" PRId64 "\n",
                               t->int64_data[0]);
                    } else if (t->raw_data.len == sizeof(int64_t)) {
                        int64_t value;
                        memcpy(&value, t->raw_data.data, sizeof(value));
                        printf("    value=%" PRId64 "\n", value);
                    }
                }
            }
        }
    }


    onnx__model_proto__free_unpacked(model, NULL);

    free(r->data);
    free(r);

    arena_destroy(&root_arena);
    */
}

