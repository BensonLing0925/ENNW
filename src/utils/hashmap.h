// This open-addressing hash table implementation is heavily inspired by and referenced to chibicc's hashmap implementation
// https://github.com/rui314/chibicc/blob/main/hashmap.c

#ifndef TK_HASHMAP_H
#define TK_HASHMAP_H

#define DEFAULT_SIZE (1 << 6)
#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME        1099511628211ULL
#define LOW_WATERMARK 50
#define HIGH_WATERMARK 70

#include <inttypes.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

struct tk_hashmap_entry {
    char* key;
    void* val;
    size_t key_len;
    uint64_t hash;
};

struct tk_hashmap {
    struct tk_hashmap_entry* buckets;
    size_t cap;
    size_t used;
};

void tk_hashmap_init(struct tk_hashmap* hmap, size_t num_buckets);
int tk_hashmap_insert(struct tk_hashmap* hmap, const char* key, size_t key_len, void* val);
void tk_hashmap_destroy(struct tk_hashmap* hmap);
struct tk_hashmap_entry* tk_hashmap_get(struct tk_hashmap* hmap, const char* key, size_t key_len);
#endif
