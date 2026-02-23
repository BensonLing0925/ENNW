#include "arena.h"
#include <inttypes.h> // PRIuPTR
#include <stdint.h>
#include <stdio.h>

// ---- Debug ----
#ifdef ARENA_DEBUG
static void dbg(struct block* b, size_t align, size_t used_align, void* p) {
    printf("[arena] align=%zu used=%zu used_align=%zu cap=%zu\n",
           align, b->used, used_align, b->cap);
    printf("[arena] data=%p data%%align=%zu\n",
           (void*)b->data, (size_t)((uintptr_t)b->data & (align - 1)));
    printf("[arena] p   =%p p%%align   =%zu\n",
           p, (size_t)((uintptr_t)p & (align - 1)));
}
#endif

void arena_init(struct arena* a) {
    a->head = NULL;
    a->current_block = NULL;
}

void arena_destroy(struct arena* a) {
    struct block* prev = NULL;
    struct block* b = a->head;
    while (b) {
        prev = b;
        b = b->next;
        freeBlock(prev);
    }
    a->head = NULL;
    a->current_block = NULL;
}

void* arena_alloc(struct arena* a, size_t size) {

    if (size == 0) {
        size = 1;
    }

    void* mem = arena_alloc_align(a, size, alignof(max_align_t));
    if (mem == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    return mem;
}

void* arena_alloc_align(struct arena* a, size_t size, size_t align) {

    if (size == 0) {
        size = 1;
    }

    if (align == 0 || (align & (align - 1)) != 0) {
        perror("alignment should be the power of two");
        return NULL;
    }

    if (align < alignof(max_align_t)) {
        align = alignof(max_align_t);
    }

    struct block* b = a->current_block;

    if (b == NULL) {
        b = allocBlock(BLOCK_SIZE);
        if (b == NULL) {
            errno = ENOMEM;
            return NULL;
        }
        a->head = b;
        a->current_block = b;
    }

    size_t used_align = ALIGN_UP(b->used, align);
    
    if (used_align + size > b->cap) {
        if (size > SIZE_MAX - align) {
            perror("memory overflow at arena_alloc_align");
            return NULL;
        }
        size_t need = size + align;
        size_t cap = (need > BLOCK_SIZE) ? need : BLOCK_SIZE;
        b = allocBlock(cap);
        if (b == NULL) {
            errno = ENOMEM;
            return NULL;
        }
        b->next = a->head;
        a->head = b;
        a->current_block = b;

        used_align = ALIGN_UP(a->current_block->used, align);
    }
    unsigned char* p = b->data + used_align;
#ifdef ARENA_DEBUG
    dbg(b, align, used_align, p);
#endif 
    b->used = used_align + size;
    return p;
}

void* arena_calloc(struct arena* a, size_t count, size_t size) {
    if (count && size > SIZE_MAX / count) return NULL;
    size_t n = count * size;
    void* p = arena_alloc(a, n);
    if (p) memset(p, 0, n);
    return p;
}

unsigned char* arena_strdup(struct arena *a, const char *s) {
    size_t n = strlen(s) + 1;
    unsigned char* p = (unsigned char*)arena_alloc_align(a, n, 1);
    if (p == NULL) {
        return NULL;
    }
    memcpy(p, s, n);
    return p;
}

void arena_reset(struct arena* a) {
    if (!a->head) {
        return;
    }
    struct block* b = a->head->next;
    while (b) {
        struct block* next = b->next;
        freeBlock(b);
        b = next;
    }
    a->head->next = NULL;
    a->current_block = a->head;
    a->current_block->used = 0;
}

struct block* allocBlock(size_t block_size) {

    struct block* b = (struct block*)malloc(sizeof(struct block) + block_size);
    if (b == NULL) {
        return NULL;
    }

#ifdef ARENA_DEBUG
    printf("b=%p, data=%p, offsetof(data)=%zu, data%%16=%zu\n",
       (void*)b, (void*)b->data,
       offsetof(struct block, data),
       (size_t)((uintptr_t)b->data & 15));

    if (block_size < BLOCK_SIZE) {
        block_size = BLOCK_SIZE;
    }
#endif

    b->cap = block_size;
    b->used = 0;
    b->next = NULL;
    return b;
}

void freeBlock(struct block* b) {
    free(b);
}
