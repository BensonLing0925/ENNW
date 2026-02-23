#ifndef ARENA_H
#define ARENA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <errno.h>
#include <stdint.h>
#include <stdalign.h>

#define BLOCK_SIZE (1024u * 64u)
#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

struct block {
    struct block* next;
    size_t cap;
    size_t used;
    alignas(max_align_t) unsigned char data[]; // flexible array member
};

struct arena {
    struct block* head;
    struct block* current_block;
};

void arena_init(struct arena* a);
void arena_destroy(struct arena* a);
void* arena_alloc(struct arena* a, size_t size);
void* arena_alloc_align(struct arena* a, size_t size, size_t align);
void* arena_calloc(struct arena* a, size_t count, size_t size);
unsigned char* arena_strdup(struct arena *a, const char *s);
void arena_reset(struct arena* a);
struct block* allocBlock(size_t block_size);
void freeBlock(struct block* b);

#endif
