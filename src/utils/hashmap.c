#include "hashmap.h"
#include "rt_error.h"
#include <string.h>

void tk_hashmap_init(struct tk_hashmap* hmap, size_t num_buckets) {
    if (num_buckets < DEFAULT_SIZE)
        num_buckets = DEFAULT_SIZE;
    hmap->cap = num_buckets;
    hmap->buckets = calloc(hmap->cap, sizeof(struct tk_hashmap_entry));
}

static int tk_hashmap_rehash(struct tk_hashmap* hmap) {
    size_t num_keys = hmap->used;
    size_t cap = hmap->cap;
    while ((num_keys * 100) / cap >= LOW_WATERMARK)
        cap = cap * 2;
    assert(cap > 0);
    struct tk_hashmap hmap_new = {0};
    hmap_new.cap = cap;
    hmap_new.buckets = calloc(cap, sizeof(struct tk_hashmap_entry));
    if (!hmap_new.buckets)
        RT_FAIL(RT_EOOM, "Failed to allocate hashmap buckets");
    for ( size_t i = 0 ; i < hmap->cap ; ++i ) {
        struct tk_hashmap_entry* ent = &hmap->buckets[i];
        if (ent->key) {
            RT_CHECK(tk_hashmap_insert(&hmap_new, ent->key, ent->key_len, ent->val));
        }
    } 
    free(hmap->buckets);
    *hmap = hmap_new;
}

static uint64_t tk_hashmap_hash(const char* key, size_t key_len) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for ( size_t b = 0 ; b < key_len ; ++b ) {
        // FNV-1a
        hash ^= (uint8_t)key[b];
        hash *= FNV_PRIME;
    }
    return hash;
}

static int tk_hashmap_match(struct tk_hashmap_entry* ent, const char* key, size_t key_len, uint64_t hash) {
    return ent->key &&
       ent->hash == hash &&
       ent->key_len == key_len &&
       memcmp(ent->key, key, key_len) == 0;
}

int tk_hashmap_insert(struct tk_hashmap* hmap, const char* key, size_t key_len, void* val) {
    if ((hmap->used * 100 / hmap->cap) >= HIGH_WATERMARK) {
        tk_hashmap_rehash(hmap);
    }
    uint64_t hash = tk_hashmap_hash(key, key_len);
    for ( size_t i = 0 ; i < hmap->cap ; ++i ) {
        size_t idx = (hash + i) % hmap->cap;
        struct tk_hashmap_entry *ent = &hmap->buckets[idx];
        // not occupied
        if (!ent->key) {
            ent->key = key;
            ent->val = val;
            ent->key_len = key_len;
            ent->hash = hash;
            hmap->used++;
            return 0;
        }

        if (tk_hashmap_match(ent, key, key_len, hash)) {
            ent->val = val;
            return 0;
        }
    }
}

void tk_hashmap_destroy(struct tk_hashmap* hmap) {
    free(hmap->buckets);
    hmap->cap = 0;
    hmap->used = 0;
    hmap->buckets = NULL;
}

struct tk_hashmap_entry* tk_hashmap_get(struct tk_hashmap* hmap, const char* key, size_t key_len) {
    uint64_t hash = tk_hashmap_hash(key, key_len);
    for (size_t i = 0; i < hmap->cap; ++i) {
        size_t idx = (hash + i) % hmap->cap;
        struct tk_hashmap_entry *ent = &hmap->buckets[idx];

        if (!ent->key)
            return NULL;

        if (tk_hashmap_match(ent, key, key_len, hash))
            return ent;
    }
    return NULL;
}
