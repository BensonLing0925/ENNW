#ifndef RT_CONTEXT_H
#define RT_CONTEXT_H

#define RESERVED 2048

#include "../../mem/arena.h"
#include "../error/rt_error.h"
#include "workspaces/rt_workspaces.h"

enum rt_type {
    RT_TRAIN,
    RT_INFERENCE,
    RT_DRYRUN
};

struct tk_rt_ctx {

    enum rt_type rt_type;
    struct arena* meta_arena;
    struct arena* data_arena;
    struct tk_workspace* ws;
    struct Model* model;

};

struct tk_rt_ctx* tk_runtime_ctx_create(struct arena* root_arena);

#endif
