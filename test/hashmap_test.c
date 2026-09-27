#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "hashmap.h"

static void test_basic_insert_get(void)
{
    struct tk_hashmap map = {0};
    tk_hashmap_init(&map, 8);

    int value = 123;
    const char *key = "hello";

    assert(tk_hashmap_insert(
        &map, key, strlen(key), &value) == 0);

    struct tk_hashmap_entry *ent =
        tk_hashmap_get(&map, key, strlen(key));

    assert(ent != NULL);
    assert(ent->val == &value);
    assert(*(int *)ent->val == 123);
    assert(ent->key_len == strlen(key));
    assert(map.used == 1);

    tk_hashmap_destroy(&map);

    printf("test_basic_insert_get passed\n");
}

static void test_missing_key(void)
{
    struct tk_hashmap map = {0};
    tk_hashmap_init(&map, 8);

    int value = 42;
    const char *key = "exists";

    assert(tk_hashmap_insert(
        &map, key, strlen(key), &value) == 0);

    struct tk_hashmap_entry *ent =
        tk_hashmap_get(
            &map,
            "does_not_exist",
            strlen("does_not_exist"));

    assert(ent == NULL);

    tk_hashmap_destroy(&map);

    printf("test_missing_key passed\n");
}

static void test_overwrite(void)
{
    struct tk_hashmap map = {0};
    tk_hashmap_init(&map, 8);

    const char *key = "tensor";

    int value1 = 100;
    int value2 = 200;

    assert(tk_hashmap_insert(
        &map, key, strlen(key), &value1) == 0);

    assert(map.used == 1);

    assert(tk_hashmap_insert(
        &map, key, strlen(key), &value2) == 0);

    /*
     * Updating an existing key must not create
     * another entry.
     */
    assert(map.used == 1);

    struct tk_hashmap_entry *ent =
        tk_hashmap_get(&map, key, strlen(key));

    assert(ent != NULL);
    assert(ent->val == &value2);
    assert(*(int *)ent->val == 200);

    tk_hashmap_destroy(&map);

    printf("test_overwrite passed\n");
}

static void test_rehash(void)
{
    enum { N = 512 };

    struct tk_hashmap map = {0};

    /*
     * Your init() currently clamps this to DEFAULT_SIZE.
     */
    tk_hashmap_init(&map, 8);

    size_t initial_cap = map.cap;

    /*
     * Important:
     * hashmap only borrows key pointers, so these strings
     * must remain alive throughout the test.
     */
    char keys[N][32];
    int values[N];

    for (int i = 0; i < N; ++i) {
        snprintf(keys[i], sizeof(keys[i]),
                 "tensor_%d", i);

        values[i] = i * 10;

        assert(tk_hashmap_insert(
            &map,
            keys[i],
            strlen(keys[i]),
            &values[i]) == 0);
    }

    assert(map.used == N);

    /*
     * With 512 entries this should definitely have forced
     * at least one resize.
     */
    assert(map.cap > initial_cap);

    /*
     * Verify that rehash preserved every entry.
     */
    for (int i = 0; i < N; ++i) {
        struct tk_hashmap_entry *ent =
            tk_hashmap_get(
                &map,
                keys[i],
                strlen(keys[i]));

        assert(ent != NULL);
        assert(ent->val == &values[i]);
        assert(*(int *)ent->val == i * 10);
    }

    tk_hashmap_destroy(&map);

    printf("test_rehash passed\n");
}

static size_t bucket_of(const char *key, size_t len, size_t cap)
{
    uint64_t hash = FNV_OFFSET_BASIS;

    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint8_t)key[i];
        hash *= FNV_PRIME;
    }

    return hash % cap;
}

static void test_collision(void)
{
    struct tk_hashmap map = {0};
    tk_hashmap_init(&map, DEFAULT_SIZE);

    char key1[32] = {0};
    char key2[32] = {0};

    int found = 0;

    for (int i = 0; i < 1000 && !found; ++i) {
        snprintf(key1, sizeof(key1), "key_%d", i);

        for (int j = i + 1; j < 1000; ++j) {
            snprintf(key2, sizeof(key2), "key_%d", j);

            if (bucket_of(key1, strlen(key1), map.cap) ==
                bucket_of(key2, strlen(key2), map.cap)) {
                found = 1;
                break;
            }
        }
    }

    assert(found);

    int value1 = 111;
    int value2 = 222;

    assert(tk_hashmap_insert(
        &map, key1, strlen(key1), &value1) == 0);

    assert(tk_hashmap_insert(
        &map, key2, strlen(key2), &value2) == 0);

    struct tk_hashmap_entry *ent1 =
        tk_hashmap_get(&map, key1, strlen(key1));

    struct tk_hashmap_entry *ent2 =
        tk_hashmap_get(&map, key2, strlen(key2));

    assert(ent1 != NULL);
    assert(ent2 != NULL);

    assert(*(int *)ent1->val == 111);
    assert(*(int *)ent2->val == 222);

    assert(map.used == 2);

    tk_hashmap_destroy(&map);

    printf("test_collision passed\n");
}

int main(void)
{
    test_basic_insert_get();
    test_missing_key();
    test_overwrite();
    test_rehash();
    test_collision();

    printf("All hashmap tests passed\n");

    return 0;
}
